import torch, os
import config_sclresnet101_stage1, sclresnet101_stage1_dataloader
from model import sclresnet101

# train stage one
def train():
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')

    train_dataset = sclresnet101_stage1_dataloader.SIMCLRDataset(data_path=config_sclresnet101_stage1.train_images_path, transform=config_sclresnet101_stage1.train_transform)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=config_sclresnet101_stage1.batch_size, shuffle=True, num_workers=config_sclresnet101_stage1.num_workers, drop_last=True)

    model = sclresnet101.SimCLRStage1().to(DEVICE)
    lossLR = sclresnet101.Loss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_sclresnet101_stage1.lr, weight_decay=1e-6)

    os.makedirs(config_sclresnet101_stage1.save_path, exist_ok=True)
    for epoch in range(1, config_sclresnet101_stage1.epoches+1):
        model.train()
        total_loss = 0
        for batch, (imgL, imgR) in enumerate(train_data):
            imgL, imgR = imgL.to(DEVICE), imgR.to(DEVICE)

            pre_L = model(imgL)
            pre_R = model(imgR)

            loss = lossLR(pre_L, pre_R, config_sclresnet101_stage1.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()

        print("epoch loss:", total_loss/len(train_dataset)*config_sclresnet101_stage1.batch_size)

        with open(os.path.join(config_sclresnet101_stage1.save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss/len(train_dataset)*config_sclresnet101_stage1.batch_size) + " ")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config_sclresnet101_stage1.save_path, 'model_stage1_epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    train()


