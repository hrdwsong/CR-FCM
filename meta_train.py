import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.backbones import BackBone
from dataloader.samplers import CategoriesSampler
from utils import ensure_path, Averager, count_acc
from tensorboardX import SummaryWriter
# ===========================================
from logger import get_logger
from tqdm import tqdm
from methods.crdcov import Crdcov
import numpy as np
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_mul', type=float, default=100)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='miniImageNet')
    parser.add_argument('--init_weights', type=str, default='./initialization/miniimagenet/checkpoint1600.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--method', type=str, default='cpea', choices=['crdcov'])
    parser.add_argument('--num_episodes_per_epoch', type=int, default=100,
                        help="""Number of episodes used for 1 epoch of meta fine tuning. """)

    parser.add_argument('--l2', type=float, default=0.4)  # L2 from 0.04--->0.4
    args = parser.parse_args()

    save_path = '-'.join([args.method, args.dataset, args.model_type, '{}w{}s'.format(args.way, args.shot)])
    args.save_path = osp.join('./results', save_path)
    ensure_path(args.save_path)

    logger = get_logger(osp.join(args.save_path, 'meta_train_log.txt'))
    args.logger = logger

    logger.info(vars(args))

    if args.dataset == 'miniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'tieredImageNet':
        from dataloader.tiered_in import TieredImagenet as Dataset
    elif args.dataset == 'cifar-fs':
        from dataloader.cifarfs import CIFARFS as Dataset
    elif args.dataset == 'fc100':
        from dataloader.fc100 import FC100 as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, args.num_episodes_per_epoch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 500, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    model = BackBone(args)
    if args.method == 'crdcov':
        args.include_cls = True
        fsl_header = Crdcov(in_dim=384, args=args)
    else:
        raise ValueError('Unknown method. Please check your selection!')

    #
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters()}], lr=args.lr, weight_decay=args.l2)  #
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)
    fsl_header_optim = torch.optim.AdamW(fsl_header.parameters(), lr=args.lr * args.lr_mul, weight_decay=args.l2)
    fsl_header_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fsl_header_optim, T_max=args.max_epoch, eta_min=1e-6)

    # load pre-trained model (no FC weights)
    model_dict = model.state_dict()
    logger.info(model_dict.keys())
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights, map_location='cpu')['teacher']
        logger.info(pretrained_dict.keys())
        pretrained_dict = {k.replace('backbone', 'encoder'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        logger.info(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        fsl_header = fsl_header.cuda()

    #
    frozen_layers = ['module.encoder.cls_token', 'module.encoder.pos_embed',
                     'module.encoder.patch_embed',
                     'module.encoder.blocks.0.',
                     'module.encoder.blocks.1.',
                     ]
    for name, param in model.named_parameters():
        # print(name)
        for item in frozen_layers:
            if item in name:
                param.requires_grad = False
                # print('{} is frozen.'.format(name))

    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
        torch.save(dict(params=fsl_header.state_dict()), osp.join(args.save_path, name + '_fsl_header.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    global_count = 0
    writer = SummaryWriter(logdir=args.save_path)

    for epoch in range(1, args.max_epoch + 1):
        train_tqdm_gen = tqdm(train_loader)
        model.train()
        fsl_header.train()
        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_tqdm_gen, 1):
            # zero gradient
            optimizer.zero_grad()
            fsl_header_optim.zero_grad()

            # forward and backward
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            data = data.reshape(args.way, -1, *data.size()[1:])
            data_shot = data[:, :args.shot].reshape(args.way*args.shot, *data.size()[2:])
            data_query = data[:, args.shot:].reshape(args.way*args.query, *data.size()[2:])
            feat_shot, feat_query = model(data_shot, data_query, args.include_cls)
            results = fsl_header(feat_shot, feat_query)
            label = torch.arange(args.way).repeat_interleave(args.query).long().to('cuda')

            loss = F.cross_entropy(results, label)

            acc = count_acc(results.data, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)

            tl.add(loss.item())
            ta.add(acc)
            train_tqdm_gen.set_description(
                'Training: Ep {} | bt {}/{}: Loss epi:{:.2f} avg: {:.4f} | Acc: epi:{:.2f} avg: {:.4f}'.format(
                    epoch, i,
                    len(train_loader),
                    loss.item(),
                    tl.item(),
                    acc,
                    ta.item()))

            loss_total = loss
            loss_total.backward()
            optimizer.step()
            fsl_header_optim.step()

        logger.info('Training: Ep {}: Loss avg: {:.4f} | Acc avg: {:.4f}'.format(epoch, tl.item(), ta.item()))
        logger.info('The weight of locations is: {:.3f}'.format(fsl_header.weight.data.item()))
        lr_scheduler.step()
        fsl_header_scheduler.step()

        current_wd = args.l2 + (0.1 * args.l2 - args.l2) * (1 + math.cos(math.pi * (epoch-1) / (args.max_epoch-1))) / 2

        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = current_wd
        for param_group in fsl_header_optim.param_groups:
            param_group['weight_decay'] = current_wd
        logger.info('Current learning rate: {}; Current weight decay: {}'.format(lr_scheduler.get_lr()[0], current_wd))

        tl = tl.item()
        ta = ta.item()

        model.eval()
        fsl_header.eval()

        vl = Averager()
        va = Averager()

        logger.info('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                data = data.reshape(args.way, -1, *data.size()[1:])
                data_shot = data[:, :args.shot].reshape(args.way * args.shot, *data.size()[2:])
                data_query = data[:, args.shot:].reshape(args.way * args.query, *data.size()[2:])
                feat_shot, feat_query = model(data_shot, data_query, args.include_cls)
                results = fsl_header(feat_shot, feat_query)
                label = torch.arange(args.way).repeat_interleave(args.query).long().to('cuda')

                loss = F.cross_entropy(results, label)
                acc = count_acc(results.data, label)
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)
        logger.info('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va >= trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        save_model('epoch-last')

    writer.close()

