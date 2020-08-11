import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F
from skimage import exposure
import os

def filterEmpty(img_files):
    new_files = []
    for fi in img_files:
        fl = fi.rstrip().replace('.jpg', '.txt').replace('images', 'labels')
        if len(open(fl).readlines()) > 0:
            new_files.append(fi)
    img_files = new_files
    return img_files

class dataGen(Dataset):
    def __init__(self, list_path, device, size):
        with open(list_path, "r") as file:
            self.img_files = [f.rstrip() for f in file.readlines()]
            # TODO get rid off all empty labels
            self.img_files = filterEmpty(self.img_files)

        self.label_files = [
            path.replace("images", "labels8").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.size = size
        self.device = device

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)]

        # Extract image as PyTorch tensor
        img_t = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(self.device)

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        data = torch.tensor(np.loadtxt(label_path).astype(np.float32)).to(self.device)

        target = {}

        target['points'] = data[:, 3:].view(-1, 4, 2).contiguous()
        target['class'] = data[:, 2]
        target['type'] = data[:, 0]
        target['order'] = data[:, 1]
        target['img_path'] = img_path

        img_patch, class_num = self.extract(img_t, target, rand=True, size=self.size)

        return img_patch, class_num

    def __len__(self):
        return len(self.img_files)

    def extract(self, img_t, target, rand, size=128, gamma=0.003, keep=1.0):
        W, H = img_t.size(2), img_t.size(1)
        points = target['points']

        assert rand

        # Select just one
        I_keep = torch.where((target['class'] <= 9) * (target['class'] >= 0))[0]
        I_keep = I_keep[torch.randperm(len(I_keep))[:int(keep * len(I_keep))]]
        points = points[I_keep]
        class_num = target['class'][I_keep]

        # Perturbation
        points = points + gamma * torch.randn(points.shape).to(self.device)

        # Produce points
        a = points[:, 0, :].view(-1, 1, 2)
        b = points[:, 1, :].view(-1, 1, 2)
        c = points[:, 2, :].view(-1, 1, 2)
        d = points[:, 3, :].view(-1, 1, 2)
        M = size
        la = torch.arange(0, 1, step=1 / M).view(1, -1, 1).to(self.device)
        e = a + la * (b - a)
        f = d + la * (c - d)
        e = e.view(-1, 1, 2)
        f = f.view(-1, 1, 2)
        nu = torch.arange(0, 1, step=1 / M).view(1, -1, 1).to(self.device)
        ij = f + nu * (e - f)
        ij = ij.view(len(points), -1, 2).clamp(0.001, 0.999)

        # Convert to int
        ij = (ij * torch.tensor([W, H]).to(self.device)).type(torch.LongTensor)
        ij_patch = torch.arange(M).repeat(M, 1).to(self.device)
        ij_patch = torch.stack([ij_patch.t(), ij_patch], dim=2).view(-1, 2)

        ij_patch[:, 0] = (M - 1) - ij_patch[:, 0]

        img_patch = torch.zeros(3, len(points), M, M).to(self.device)

        img_patch[:, :, ij_patch[:, 1], ij_patch[:, 0]] = img_t[:, ij[:, :, 1], ij[:, :, 0]]

        img_patch.transpose_(1, 0)

        return img_patch, class_num

if __name__ == "__main__":
    print('Perform some dataset selfcheck')
    train_path = '/home/marat/dataset/photo_birka/part1/train.txt'
    out_path = '/mnt/hugedisk/data/ganbirka128_2/'

    os.makedirs(out_path, exist_ok=True)

    dataset = dataGen(train_path, 'cpu', size=128)

    cnt = np.zeros(10)

    for i in range(len(dataset) * 3):
        d = dataset[i]
        for j, (img_t, label) in enumerate(zip(d[0], d[1])):
            if i >= len(dataset) and cnt[int(label.item())] > cnt.mean():
                continue

            clnum = str(int(label.item()))

            os.makedirs(os.path.join(out_path, 'class' + clnum), exist_ok=True)

            out_file = os.path.join(out_path, 'class' + clnum + '/'
                                    + clnum + '_' + str(i) + '_' + str(j) + '.jpg')
            print('Saving:', out_file)
            transforms.ToPILImage()(img_t).save(out_file)

            cnt[int(label.item())] += 1

    print(cnt)