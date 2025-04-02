import argparse
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.backbones import BackBone
from dataloader.samplers import CategoriesSampler
from utils import Averager, count_acc, compute_confidence_interval
from logger import get_logger
from tqdm import tqdm
from methods.crdcov import Crdcov

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='miniImageNet')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--method', type=str, default='cpea', choices=['crdcov'])
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()

    logger = get_logger(osp.join(args.save_path, 'eval_log.txt'))
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

    model = BackBone(args)
    if args.method == 'crdcov':
        args.include_cls = True
        fsl_header = Crdcov(in_dim=384, args=args)
    else:
        raise ValueError('Unknown method. Please check your selection!')

    logger.info('Using {}'.format(args.model_type))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        fsl_header = fsl_header.cuda()

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 1000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=4, pin_memory=True)
    test_acc_record = np.zeros((1000,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])  #
    model.eval()

    fsl_header.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '_fsl_header.pth'))['params'])  #
    fsl_header.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)

    loader_tqdm_gen = tqdm(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader_tqdm_gen, 1):
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

            acc = count_acc(results.data, label)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            loader_tqdm_gen.set_description('batch {}: avg acc {:.2f} - acc epi {:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    logger.info('Val Best Epoch {}, Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'] * 100))
    logger.info('Test Acc {:.4f} +- {:.4f}'.format(m*100, pm*100))
