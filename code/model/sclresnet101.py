import torch
import torch.nn as nn
import torch.nn.functional as F
from model import resnet

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values

# stage one ,unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRStage1, self).__init__()

        self.f = []
        # se_resnet101
        for name, module in resnet.resnet(in_channels=3, num_classes=2, mode='resnet101', pretrained=False).named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(out, dim=-1)


# stage two ,supervised learning
class SimCLRStage2(torch.nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f
        self.SA = SelfAttention(2048)
        # classifier
        self.fc1 = nn.Linear(2048, 512, bias=True)
        self.fc2 = nn.Linear(512, num_class, bias=True)

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        feature = self.f(x)
        feature = torch.unsqueeze(feature, dim=1)
        out = self.SA(feature)
        out = torch.squeeze(out)

        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self, out_1, out_2, batch_size, temperature=0.5):
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


if __name__ == "__main__":
    shape = (3, 224, 224)
    tensor1 = torch.unsqueeze(torch.rand(shape), dim=0)
    tensor2 = torch.unsqueeze(torch.rand(shape), dim=0)
    tensor = torch.cat((tensor1, tensor2), dim=0)

    # model1 = SimCLRStage1()
    # result = model1(tensor)

    model2 = SimCLRStage2(2)
    result = model2(tensor)
    print(result)


