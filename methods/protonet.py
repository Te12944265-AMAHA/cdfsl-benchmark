# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from utils import guassian_kernel


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, discriminator=None, cosine=False, adaptive_classifier=None):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()
        if (discriminator is not None):
            self.discriminator = discriminator
            if (torch.cuda.device_count() > 1):
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.discriminator = nn.DataParallel(self.discriminator)
        self.adaptive_classifier = adaptive_classifier
        self.cosine_dist = cosine
        if self.cosine_dist: self.temperature = 10


    def set_forward(self,x,is_feature = False):
        # we want to find the prototypes using the support set
        # and assign labels to query set examples using a distance function
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        if self.cosine_dist:
            dists = cosine_dist(z_query, z_proto, self.temperature)
        else:
            dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        if is_adversarial:
            z_set = z_proto.view(-1).unsqueeze(0)
            return scores, z_set

        elif is_adaptFinetune:
            assert (is_adversarial==False)
            y = torch.range(0,self.n_way-1).unsqueeze(1).type(torch.long).repeat(1, self.n_way)
            return scores, z_support.view(self.n_way*self.n_support, -1), y.view(self.n_way*self.n_support)
        else:
            return scores


    def discriminator_score(self, zS, zT, adv_loss_fn, epoch):
        logitS = self.discriminator(zS)
        logitT = self.discriminator(zT)
        ls = adv_loss_fn(logitS, torch.ones(zS.shape[0], dtype=torch.long).cuda())
        lt = adv_loss_fn(logitT, torch.zeros(zT.shape[0], dtype=torch.long).cuda())
        loss = ls+lt

        # SPL: reweighting of loss
        corr = torch.dot((zS / torch.norm(zS)).squeeze(), (zT / torch.norm(zT)).squeeze()).unsqueeze(0)
        wt = torch.norm(corr)

        return wt*loss


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


    def set_forward_loss(self, xS, xT=None, params = None, epoch=None):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        #y_query = y_query.cuda()

        scores, z_source = self.set_forward(xS, is_adversarial=True)
        proto_loss = self.loss_fn(scores, y_query) # classifier loss

        # if self.discriminator is not None:
        if params is not None:
            if params.adversarial:
                if params.n_shot_test != -1: self.n_support=params.n_shot_test
                _, z_target = self.set_forward(xT, is_adversarial=True)
                adverasrial_loss = self.discriminator_score(z_source, z_target, self.adv_loss_fn, epoch)
                domain_reg = params.gamma
                loss = proto_loss + domain_reg*adverasrial_loss
                return loss, proto_loss, adverasrial_loss
            elif params.adaptFinetune:
                assert (params.adversarial==False)
                _, z_target, y_target = self.set_forward(xT, is_adaptFinetune = True)
                logitT = self.adaptive_classifier(z_target)
                adaptive_loss = self.adv_loss_fn(logitT, y_target.cuda())
                adaptive_wt = 1.0
                loss = proto_loss + adaptive_wt*adaptive_loss
                return loss, proto_loss, adaptive_loss
        else:
            return proto_loss


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine_dist( x, y, tau):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x/torch.norm(x, p=2, dim=1).unsqueeze(1).repeat(1,d)
    y = y/torch.norm(y, p=2, dim=1).unsqueeze(1).repeat(1,d)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return tau*torch.sum(x*y, dim=2)
