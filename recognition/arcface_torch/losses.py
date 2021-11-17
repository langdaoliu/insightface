import torch
from torch import nn
import torchshard as ts


def get_loss(name):
    if name == "cosface":
        return CosFace()
    elif name == "arcface":
        return ArcFace()
    elif name == "magface":
        return MagFace()
    elif name == "magcos":
        return MagCos()
    else:
        raise ValueError()


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine

class MagCos(nn.Module):
    def __init__(self, lambda_g=20, s=30.0, u_margin=0.8, l_margin=0.45, u_a=110, l_a=10):
        super(MagCos, self).__init__()
        # self.in_features = in_features
        # self.out_features = (out_features + world_size - 1) // world_size
        self.lambda_g = lambda_g
        self.s = s
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.l_a = l_a
        self.u_a = u_a
    # def _margin(self, x):
    #     """generate adaptive margin
    #     """
    #     # ori
    #     margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (x - self.l_a) + self.l_margin
    #     return margin
    # def g_loss(self, x_norm):
    #     #ori
    #     g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
    #     return torch.mean(g)

    def forward(self, cosine: torch.Tensor, label):
        # x_norm = torch.norm(input, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        # ada_margin = self._margin(x_norm)
        # ada_margin = 0.6/(1+torch.exp(cosine.mul(9)-4))+0.4
        ada_margin = (1-cosine)*0.45+0.45
        ada_margin.clamp(-1, 1)
        index = torch.where(label != -1)[0]
        # m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        # m_hot.scatter_(1, label[index, None], 1)
        # ori_cos = cosine
        cosine.acos_()
        cosine[index] += ada_margin[index]
        cosine.cos_().mul_(self.s)
        # cosine_m.cos_()
        # similarity = torch.where(ori_cos > 0, cosine, ori_cos)
        # m_hot = torch.zeros_like(similarity, device=cosine.device)
        # m_hot.scatter_(1, label[index, None], 1)
        # cosine[index] = (m_hot.mul_(similarity[index]) + (1.0 - m_hot).mul_(cosine[index])).mul_(self.s)
        # cosine.cos_().mul_(self.s)
        # gloss
        # g_loss = self.g_loss(x_norm)
        # return cosine,g_loss*self.lambda_g
        return cosine

class MagFace(nn.Module):
    def __init__(self, lambda_g=20, s=30.0, u_margin=0.8, l_margin=0.45, u_a=110, l_a=10):
        super(MagFace, self).__init__()
        # self.in_features = in_features
        # self.out_features = (out_features + world_size - 1) // world_size
        self.lambda_g = lambda_g
        self.s = s
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.l_a = l_a
        self.u_a = u_a
    def _margin(self, x):
        """generate adaptive margin
        """
        # ori
        margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (x - self.l_a) + self.l_margin
        return margin
    def g_loss(self, x_norm):
        #ori
        g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        return torch.mean(g)

    def forward(self, cosine: torch.Tensor, label):
        x_norm = torch.norm(input, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        ada_margin = self._margin(x_norm)
        # ada_margin = 0.6/(1+torch.exp(cosine.mul(9)-4))+0.4
        # ada_margin = (1-cosine)*0.45+0.45
        ada_margin.clamp(-1, 1)
        index = torch.where(label != -1)[0]
        # m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        # m_hot.scatter_(1, label[index, None], 1)
        ori_cos = cosine
        # cosine.acos_()
        # cosine[index] += ada_margin[index]
        # cosine.cos_().mul_(self.s)
        # cosine_m.cos_()
        similarity = torch.where(ori_cos > 0, cosine, ori_cos)
        m_hot = torch.zeros_like(similarity, device=cosine.device)
        m_hot.scatter_(1, label[index, None], 1)
        # logits =  (m_hot * similarity + (1.0 - m_hot) * cos_theta)
        cosine[index] = m_hot*similarity[index] + (1.0 - m_hot)*(cosine[index])
        cosine.cos_().mul_(self.s)
        #gloss
        g_loss = self.g_loss(x_norm)
        return cosine,g_loss*self.lambda_g
        # return cosine