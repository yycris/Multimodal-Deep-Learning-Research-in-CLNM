import torch
from dsaf_dataloader import CSVDataset
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from model import dsaf
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def train_on_epochs(train_dataset: CSVDataset, val_dataset: CSVDataset, pre_train: str):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    net = dsaf.Deep_fusion_classifier(2)
    if pre_train != '':
        net.load_state_dict(torch.load(pre_train), state_dict=False)
    net.to(device)

    model_params = net.parameters()
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate, weight_decay=1e-2)
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
        # train
        net.train()
        running_loss = 0.0
        test_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, (tumor_dl, fat_dl, labs) in enumerate(train_bar):
            # clinical_feature = clinical_feature.float()
            tumor_dl = tumor_dl.float()
            fat_dl = fat_dl.float()
            labels = labs.to(torch.int64)

            optimizer.zero_grad()
            logits = net(tumor_dl.to(device), fat_dl.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     config.epoches,
                                                                     loss)
        scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for (tumor_dl, fat_dl, labs) in val_bar:
                # val_clinical_feature = clinical_feature.float()
                val_tumor_dl = tumor_dl.float()
                val_fat_dl = fat_dl.float()
                val_labels = labs.to(torch.int64)

                outputs = net(val_tumor_dl.to(device), val_fat_dl.to(device))
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
            torch.save(net.state_dict(), config.save_path + 'dsaf_2 - ' + str(epoch) + '.pth')

if __name__ == "__main__":
    # train_clinic_path = ''
    train_tumor_dl_path = ''
    train_fat_dl_path = ''

    # val_clinic_path = ''
    val_tumor_dl_path = ''
    val_fat_dl_path = ''

    train_on_epochs(CSVDataset(train_tumor_dl_path, train_fat_dl_path),
                    CSVDataset(val_tumor_dl_path, val_fat_dl_path), pre_train='')
