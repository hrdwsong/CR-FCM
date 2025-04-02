import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.bdc_module import BDC


class Crdcov(nn.Module):
    def __init__(self, in_dim, args):
        super(Crdcov, self).__init__()

        self.args = args
        h = w = 14
        reduce_dim = None
        self.feat_dim = (in_dim, w, h)
        self.dcov = BDC(is_vec=True, input_dim=self.feat_dim, dimension_reduction=reduce_dim)
        self.n_way = args.way
        self.n_shot = args.shot
        self.n_query = args.query
        self.weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, feat_shot, feat_query):
        _, n, c = feat_query.size()

        query_class = feat_query[:, 0, :].unsqueeze(1)  # Q x 1 x C
        query_image = feat_query[:, 1:, :]  # Q x L x C

        support_class = feat_shot[:, 0, :].unsqueeze(1)  # KS x 1 x C
        support_image = feat_shot[:, 1:, :]  # KS x L x C


        # dcov metric
        query_image = F.normalize(query_image, p=2, dim=2)
        query_image = query_image - torch.mean(query_image, dim=2, keepdim=True)
        support_image = F.normalize(support_image, p=2, dim=2)
        support_image = support_image - torch.mean(support_image, dim=2, keepdim=True)
        support_image = support_image.transpose(1, 2)  # NK x 196 x 384
        query_image = query_image.transpose(1, 2)
        x = torch.concat([support_image, query_image], dim=0)
        z_all = self.dcov(x.reshape(-1, *self.feat_dim))
        z_proto = z_all[:self.n_way*self.n_shot].reshape(self.n_way, self.n_shot, -1).mean(1)
        z_query = z_all[self.n_way*self.n_shot:]
        bdc_scores = self.metric(z_query, z_proto, use_prod=(self.n_shot == 1))

        support_class = support_class.reshape(self.n_way, self.n_shot, c).transpose(-2, -1)
        query_class = query_class.transpose(-1, -2)
        support_class = self.dcov(support_class)
        query_class = self.dcov(query_class)
        cls_scores = self.metric(query_class, support_class, use_prod=True)
        scores = self.weight * bdc_scores + (1 - self.weight) * cls_scores
        return scores

    def metric(self, x, y, use_prod=True):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if use_prod:
            score = (x * y).sum(2)
        else:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist

        return score
