import torch
import os
import argparse
from dataloader import Dataset
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from model import densenet, resnet, vgg, sclresnet101
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

def data_list(datapath: str):
    train_images_path = []
    val_images_path = []
    test_images_path = []
    img_path = [datapath + '/train', datapath + '/valid', datapath + '/itest']
    for path in img_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if path == datapath + '/train':
                    train_images_path.append(os.path.join(root, file))
                elif path == datapath + '/valid':
                    val_images_path.append(os.path.join(root, file))
                else:
                    test_images_path.append(os.path.join(root, file))
    return train_images_path, val_images_path, test_images_path


def train_on_epochs(train_dataset: Dataset, val_dataset: Dataset, pre_train: str):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # net = densenet.densenet121(isattention=False, pretrained=False, num_classes=2)
    # net = resnet.resnet(in_channels=3, num_classes=2, mode='resnet18', isattention=True, pretrained=True)
    # net = vgg.vgg(model_name="vgg19", isattention=False, pretrained=True)

    net = sclresnet101.SimCLRStage2(2)
    net.load_state_dict(torch.load("./result/sclresnet101_stage1/model_stage1_epoch.pth"), strict=False)

    # if pre_train != '':
    #     net.load_state_dict(torch.load(pre_train), strict=False)

    net.to(device)

    model_params = net.parameters()
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate,
                                 weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma,
                                                last_epoch=-1)

    loss_function = nn.CrossEntropyLoss()

    writer = SummaryWriter('./result')

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_loader = DataLoader(train_dataset, **config.dataset_params)
    val_loader = DataLoader(val_dataset, **config.dataset_params)

    train_steps = len(train_loader)
    val_steps = len(val_loader)

    for epoch in range(config.epoches):
        net.train()
        running_loss = 0.0
        test_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, (imgs, labs) in enumerate(train_bar):
            images = imgs.float()
            labels = labs.to(torch.int64)
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     config.epoches,
                                                                     loss)
        scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for imgs, labs in val_bar:
                val_images = imgs.float()
                val_labels = labs.to(torch.int64)
                outputs = net(val_images.to(device))
                val_loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                test_loss += val_loss.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           config.epoches)
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_loss: %.3f accurate: %.3f' %
                  (epoch + 1, running_loss / train_steps, test_loss / val_steps, val_accurate))
            ValLoss = test_loss / val_steps
            writer.add_scalar("TrainLoss", running_loss / train_steps, epoch)
            writer.add_scalar("ValLoss", test_loss / val_steps, epoch)
            writer.add_scalar("acc", val_accurate, epoch)
            torch.save(net.state_dict(), config.save_path + 'stage2 - ' + str(epoch) + '.pth')


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='./data')
    parser.add_argument('-l', '--label_path', help='path to your datasets label', default='./data/3y.csv')
    parser.add_argument('-r', '--pre_train', help='path to the pretrain weights', default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    label_path = args.label_path
    pre_train = args.pre_train

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.01651567, 0.01632006, 0.01664748], std=[0.06470487, 0.06384874, 0.06503657])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.01651567, 0.01632006, 0.01664748], std=[0.06470487, 0.06384874, 0.06503657])
    ])

    train_images_path, val_images_path, test_images_path = data_list(data_path)
    train_on_epochs(Dataset(train_images_path, label_path, transform=train_transform),
                    Dataset(val_images_path, label_path, transform=valid_transform), pre_train)
