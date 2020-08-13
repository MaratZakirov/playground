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
    def __init__(self, list_path, device, size, G=None, D=None, latent_dim=None):
        self.latent_dim = latent_dim
        self.G = G
        self.D = D

        self.transform = transforms.Compose([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

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

    def __len__(self):
        return len(self.img_files)

    def extract(self, img_t, target, rand, size=128, gamma=0.003, keep=1.0):
        W, H = img_t.size(2), img_t.size(1)
        points = target['points']

        assert rand

        # Select just one
        I_keep = torch.where((target['class'] <= 9) * (target['class'] >= 0))[0]
        I_keep = I_keep[torch.randperm(len(I_keep))[:int(keep * len(I_keep))]]
        nonaff_points = None#points[~I_keep]
        points = points[I_keep]
        aff_points = torch.clone(points)
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

        return img_patch, class_num, aff_points, nonaff_points

    def dextract(self, img_t, fake_patch, aff_points):
        W, H = img_t.size(2), img_t.size(1)

        # Perturbation
        points = aff_points

        # Produce points
        a = points[:, 0, :].view(-1, 1, 2)
        b = points[:, 1, :].view(-1, 1, 2)
        c = points[:, 2, :].view(-1, 1, 2)
        d = points[:, 3, :].view(-1, 1, 2)
        M = self.size
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

        # Several rotaions of fake patches
        fake_patch = fake_patch.permute(1, 0, 3, 2).contiguous()
        fake_patch = torch.flip(fake_patch, dims=[2]).contiguous()
        fake_patch = fake_patch.view(3, -1, self.size ** 2)

        img_t[:, ij[:, :, 1], ij[:, :, 0]] = fake_patch

        return img_t

    def tuneImg(self, fake, real):
        nB = len(real)
        assert fake.shape == real.shape
        fake = fake.contiguous().view(nB * 3, -1)
        real = real.contiguous().view(nB * 3, -1)
        fake = (fake - fake.mean(1).view(-1, 1)) / fake.std(1).view(-1, 1)
        fake = (fake * real.std(1).view(-1, 1)) + real.mean(1).view(-1, 1)
        return torch.clamp(fake.view(nB, 3, 128, 128), 0, 1)

    def getDigit(self, index):
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

        img_patch, class_num = self.extract(img_t, target, rand=True)

        return img_patch, class_num

    def getImg(self, index):
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

        img_patch, class_num, aff_points, _ = self.extract(img_t, target, rand=True, size=self.size)

        wmetric_real, _ = self.D(img_patch)

        list_fake_imgs = []
        list_fake_metric = []

        for i in range(60):
            z = torch.randn(len(img_patch), self.latent_dim, device=self.device)
            fake_patch = self.G(z, class_num.type(torch.LongTensor))
            wmetric_fake, _ = self.D(fake_patch)

            fake_patch = self.tuneImg(fake_patch, img_patch)

            list_fake_imgs.append(fake_patch.cpu())
            list_fake_metric.append(wmetric_fake.squeeze().cpu())

        list_fake_imgs = torch.stack(list_fake_imgs)
        list_fake_metric = torch.stack(list_fake_metric)

        best_I = list_fake_metric.argmin(dim=0)
        fake_patch = list_fake_imgs[best_I, torch.arange(list_fake_imgs.size(1))].to(self.device)

        print('Real threshold')
        print(wmetric_real.squeeze())
        print(list_fake_metric[best_I, torch.arange(list_fake_imgs.size(1))])

        img_t = self.dextract(img_t, fake_patch, aff_points)

        return img_t, target

    def __getitem__(self, index):
        if self.G == None:
            return self.getDigit(index)
        else:
            return self.getImg(index)

def produceNewBirkas(generator, discriminator, device, latent_dim, num):
    print('Generating birkas with new digits')
    train_path = '/home/marat/dataset/photo_birka/part1/train.txt'
    out_path = '/mnt/hugedisk/data/imagesGEN/'

    os.makedirs(out_path, exist_ok=True)

    dataset = dataGen(train_path, 'cuda', size=128, G=generator, D=discriminator, latent_dim=latent_dim)

    for i in range(20):#len(dataset)):
        img_t, traget = dataset[i]
        img = transforms.ToPILImage()(img_t.cpu())
        img_fname = os.path.join(out_path, 'gen_' + traget['img_path'].split('/')[-1])
        img.save(img_fname)

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