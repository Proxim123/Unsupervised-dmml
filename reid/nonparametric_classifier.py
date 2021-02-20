import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math
import types
import random


class ExemplarMemory(Function):
    # def __init__(self, em, mu=0.01):
    #     super(ExemplarMemory, self).__init__()
    #     self.em = em
    #     self.mu = mu

    @staticmethod
    def forward(ctx, input1, input2, em, mu=0.01):
        ctx.em = em
        ctx.mu = mu
        ctx.save_for_backward(input1, input2)
        outputs = input1.mm(em.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        input1, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.mm(ctx.em)
        for x, y in zip(input1, targets):
            ctx.em[y] = ctx.mu * ctx.em[y] + (1. - ctx.mu) * x
            ctx.em[y] /= ctx.em[y].norm()
        return grad_inputs, None, None, None


class nonparametric_classifier(nn.Module):
    def __init__(self, num_features, num_classes, tau=0.05, lambda0=1, delta=1, mu=0.4):
        super(nonparametric_classifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.mu = mu
        self.tau = tau
        self.lambda0 = lambda0
        self.delta = delta
        self.em = nn.Parameter(torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets, epoch=None):
        mu = min(self.mu / 60 * (epoch + 1), 1.0)
        # inputs = ExemplarMemory(self.em, mu=mu)(inputs, targets)
        inputs = ExemplarMemory.apply(inputs, targets, self.em, mu)
        inputs /= self.tau

        if epoch > 4:
            #loss, ks = self.target_loss(inputs, targets)
            loss, ks = self.cross_entropy2d(inputs, targets)
            return loss, ks
        else:
            loss = F.cross_entropy(inputs, targets)
            ks = torch.FloatTensor(128).zero_()
        return loss, ks

    def target_loss(self, inputs, targets):
        targets_line, ks = self.adaptive_selection(inputs.detach().clone(), targets.detach().clone())
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets_line * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss, ks

    def adaptive_selection(self, inputs, targets):
        ###minimize distances between all person images
        targets_onehot = (inputs > self.lambda0 / self.tau).float()
        ks = (targets_onehot.sum(dim=1)).float()
        ks1 = ks.cpu()
        ks_mask = (ks > 1).float()
        ks = ks * ks_mask + (1 - ks_mask) * 2
        ks = self.delta / (ks * torch.log(ks))
        ks = (ks * ks_mask).view(-1, 1)
        targets_onehot = targets_onehot * ks

        ###maximize distances between similar person images
        targets = torch.unsqueeze(targets, 1)
        targets_onehot.scatter_(1, targets, float(1))

        return targets_onehot, ks1

	def cross_entropy2d(self, inputs, targets, weight=None, size_average=True):
		# inputs: (n, c, h, w), targets: (n, h, w)
		n, c, h, w = inputs.size()
		# log_p: (n, c, h, w)
		log_p = F.log_softmax(inputs)
		# log_p: (n*h*w, c)
		log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
		log_p = log_p[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
		log_p = log_p.view(-1, c)
		# targets: (n*h*w,)
		mask = targets >= 0
		targets = targets[mask]
		loss = F.nll_loss(log_p, targets, weight=weight, size_average=False)
		if size_average:
			loss /= mask.data.sum()
		return loss
