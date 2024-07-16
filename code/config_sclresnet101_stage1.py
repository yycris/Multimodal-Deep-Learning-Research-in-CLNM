import os
from torchvision import transforms
def data_list(datapath:str):
    train_images_path = []
    val_images_path = []
    test_images_path = []
    img_path = [datapath+'/train', datapath+'/valid', datapath+'/itest']
    for path in img_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if path == datapath+'/train':
                    train_images_path.append(os.path.join(root, file))
                elif path == datapath+'/valid':
                    val_images_path.append(os.path.join(root, file))
                else:
                    test_images_path.append(os.path.join(root, file))
    return train_images_path, val_images_path, test_images_path

train_images_path, val_images_path, test_images_path = data_list("./data")

lr = 1e-4
batch_size = 16
num_workers = 0
epoches = 200

save_path = "./result/sclresnet101_stage1"

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.Resize((224, 224)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=11,sigma=(0.06, 0.07))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.01651567, 0.01632006, 0.01664748], std=[0.06470487, 0.06384874, 0.06503657])
    ])


