import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

class Cross_attention_dl(nn.Module):
    def __init__(self, embed_dim1, embed_dim2):
        super(Cross_attention_dl, self).__init__()
        self.SA1 = SelfAttention(embed_dim1)
        self.SA2 = SelfAttention(embed_dim2)
        self.weight1 = torch.ones(1, requires_grad=True)
        self.weight2 = torch.ones(1, requires_grad=True)

    def forward(self, x, y):
        x1 = self.SA1(x)
        x1 = x1 + y * self.weight2

        y1 = self.SA1(y)
        y1 = y1 + x * self.weight1

        temp = torch.cat((x1, y1), dim=2)

        result = self.SA2(temp)

        return result

class Cross_attention_dl_other(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3):
        super(Cross_attention_dl_other, self).__init__()
        self.SA1 = SelfAttention(embed_dim1)
        self.SA2 = SelfAttention(embed_dim2)
        self.SA3 = SelfAttention(embed_dim3)
        self.weight1 = torch.ones(1, requires_grad=True)
        self.weight2 = torch.ones(1, requires_grad=True)

    def forward(self, x, y):
        x1 = self.SA1(x)
        x1 = x1 * self.weight1

        y1 = self.SA2(y)
        y1 = y1 * self.weight2

        temp = torch.cat((x1, y1), dim=2)
        result = self.SA3(temp)

        return result

class Deep_fusion_classifier(nn.Module):
    def __init__(self, num_classes):
        super(Deep_fusion_classifier, self).__init__()
        # self.CA1 = Cross_attention_dl_other(embed_dim1=4, embed_dim2=2048, embed_dim3=2052)
        self.CA2 = Cross_attention_dl(embed_dim1=2048, embed_dim2=4096)
        # self.CA3 = Cross_attention_dl_other(embed_dim1=2052, embed_dim2=4096, embed_dim3=6148)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, y, z):
        # attended_values1 = self.CA1(x, y)
        attended_values = self.CA2(y, z)
        # attended_values = self.CA3(attended_values1, attended_values2)

        attended_values = torch.flatten(attended_values, 1)
        result = self.fc1(attended_values)
        result = torch.relu(result)
        result = self.fc2(result)
        return result

if __name__ == '__main__':

    # shape1 = (1, 4)
    shape2 = (1, 2048)
    shape3 = (1, 2048)

    # tensor1 = torch.unsqueeze(torch.rand(shape1), dim=0)
    tensor2 = torch.unsqueeze(torch.rand(shape2), dim=0)
    tensor3 = torch.unsqueeze(torch.rand(shape3), dim=0)

    model = Deep_fusion_classifier(2)
    result = model(tensor2, tensor3)
    print(result)

