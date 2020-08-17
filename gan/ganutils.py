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

def imageSharpness(img_t):
    nB = len(img_t)
    grayscale = img_t.mean(dim=1).unsqueeze(1)
    lap_kernel = torch.tensor([[0, 1.0, 0],
                               [1, -4,  1],
                               [0, 1.0, 0]], device=img_t.device).view(1, 1, 3, 3)
    laplacian = F.conv2d(grayscale, lap_kernel)
    return laplacian.view(nB, -1).std(dim=1)

class dataGen(Dataset):
    def __init__(self, list_path, device, size, G=None, D=None, latent_dim=None):
        self.latent_dim = latent_dim
        self.G = G
        self.D = D

        self.normimg = lambda x : (x - 0.5) / 0.5

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

    def extract(self, img_t, target, rand, size=128, gamma=0.003):
        W, H = img_t.size(2), img_t.size(1)
        points = target['points']

        assert rand

        # Select just one
        I_keep = torch.where((target['class'] <= 9) * (target['class'] >= 0))[0]
        I_keep = I_keep[torch.randperm(len(I_keep))[:int(len(I_keep))]]
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

    def dextract(self, img_t, fake_patch, aff_points, gamma=0.003):
        W, H = img_t.size(2), img_t.size(1)

        # Perturbation
        points = aff_points + gamma * torch.randn(aff_points.shape).to(self.device)

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

        img_patch, class_num, _, _ = self.extract(img_t, target, rand=True)

        return img_patch, class_num

    def getImg(self, index, trys=100):
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
        class_num = class_num.type(torch.LongTensor).to(self.device)

        nB = len(img_patch)

        # TODO do I need reference values at all?
        #metrs_ref = torch.zeros(3, nB)
        wm_real, class_real = self.D(self.normimg(img_patch))
        #class_fake[torch.arange(nB), class_num]
        #sharpness_real = imageSharpness(self.normimg(img_patch))

        if (torch.argmax(class_real, dim=1) != class_num).any():
            print('All classes of real images are good')
        else:
            print('Note: Some classes are different of real images')

        fake_imgs = torch.zeros(trys, nB, 3, self.size, self.size, device=self.device)
        metrs = torch.zeros(3, nB, trys)

        # Measure 3 things of a fake image
        # 1. Wessertain metric
        # 2. Class confidence
        # 3. Image sharpness (not noise)
        for i in range(trys):
            z = torch.randn(nB, self.latent_dim, device=self.device)

            fake_imgs[i] = self.G(z, class_num)

            wm_fake, class_fake = self.D(fake_imgs[i])
            fake_imgs[i] = self.tuneImg(fake_imgs[i], img_patch)

            metrs[0, :, i] = -wm_fake.squeeze().cpu()
            metrs[1, :, i] = class_fake[torch.arange(nB), class_num]
            metrs[2, :, i] = imageSharpness(fake_imgs[i]).cpu()


        # TODO Randomly selected values
        coef = torch.tensor([0.6, 0.2, 0.2]).view(3, 1, 1)

        # Normalizing to standard distribution
        metrs = (coef * (metrs - metrs.mean(2).unsqueeze(2)) / metrs.std(2).unsqueeze(2)).mean(0).transpose(0, 1)

        # TODO argmin looks more reasonable why?!!
        best_I = metrs.argmax(dim=0)
        best_fakes = fake_imgs[best_I, torch.arange(nB)].to(self.device)

        print('Metrics compare')
        print(wm_real.squeeze())
        print(metrs[best_I, torch.arange(nB)])

        img_t = self.dextract(img_t, best_fakes, aff_points)

        return img_t, target

    def __getitem__(self, index):
        if self.G == None:
            return self.getDigit(index)
        else:
            return self.getImg(index)

def produceNewBirkas(generator, discriminator, device, latent_dim):
    print('Generating birkas with new digits')
    train_path = '/home/marat/dataset/photo_birka/part1/train.txt'
    out_path = '/mnt/hugedisk/data/imagesGEN/'

    os.makedirs(out_path, exist_ok=True)

    dataset = dataGen(train_path, device=device, size=128, G=generator, D=discriminator, latent_dim=latent_dim)

    for i in range(len(dataset)):
        img_t, target = dataset[i]
        img = transforms.ToPILImage()(img_t.cpu())
        print('Generate i:', i, 'clone of', target['img_path'].split('/')[-1])
        img_fname = os.path.join(out_path, 'gen_' + target['img_path'].split('/')[-1])
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