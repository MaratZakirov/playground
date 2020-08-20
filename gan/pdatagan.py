import argparse
import os
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

import torch.nn as nn
import torch.autograd as autograd
import torch

from PIL import Image
import numpy as np

torch.manual_seed(50)
colors = torch.rand(11, 3)
colors[0, :] = 0

os.makedirs("pdata", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--dataroot', default='', required=False, help='path to dataset')
opt = parser.parse_args()
print(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def filterEmpty(img_files):
    new_files = []
    for fi in img_files:
        fl = fi.rstrip().replace('.jpg', '.txt').replace('images', 'labels')
        if len(open(fl).readlines()) > 0:
            new_files.append(fi)
    img_files = new_files
    return img_files

class pdataData(Dataset):
    def __init__(self, list_path, size, nclass):
        self.size = size
        self.nclass = nclass

        with open(list_path, "r") as file:
            self.img_files = [f.rstrip() for f in file.readlines()]
            # TODO get rid off all empty labels
            self.img_files = filterEmpty(self.img_files)

        self.label_files = [
            path.replace("images", "labels8").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

    def __len__(self):
        return len(self.label_files)

    def showImgTensor(self, t):
        transforms.ToPILImage()(colors[t.argmax(0)].permute(2, 0, 1)).resize((512, 512)).show()

    def __getitem__(self, item):
        data = torch.tensor(np.loadtxt(self.label_files[item]).astype(np.float32))
        nB = len(data)

        ij = (self.size * data[:, 3:].view(nB, 4, 2).mean(dim=1))
        cl = data[:, 2].type(torch.LongTensor)

        # Filter out
        ij = ij[cl <= 9]
        cl = cl[cl <= 9]

        # Random rotation
        radian =  (3.14 / 2) * torch.randint(-2, 3, (1, ))
        rot_mat = torch.tensor([[torch.cos(radian), -torch.sin(radian)],
                                [torch.sin(radian), torch.cos(radian)]])
        ij = torch.mm(ij, rot_mat)

        ij = ij.type(torch.LongTensor)
        r = torch.zeros(self.nclass + 1, self.size, self.size)
        r[0, :, :] = 1.0
        r[0, ij[:, 1], ij[:, 0]] = 0.0
        r[cl + 1, ij[:, 1], ij[:, 0]] = 1.0

        if 0:
            Image.open(self.label_files[item].replace('labels8', 'images')
                       .replace('.txt', '.jpg')).resize((512, 512)).show()
            self.showImgTensor(r)
            exit()

        return r

class Generator(nn.Module):
    def __init__(self, ngf, nz, n_classes):
        super(Generator, self).__init__()

        self.n_classes = n_classes
        self.upsample = lambda x : nn.Upsample(scale_factor=2)(x[:, :x.size(1) // 2])
        self.block0 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                                    nn.BatchNorm2d(ngf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block1 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block3 = nn.Sequential(nn.ConvTranspose2d(ngf, n_classes + 1, 4, 2, 1, bias=False),
                                    nn.Softmax2d())

    def forward(self, z):
        # Setting up appropriate prefix
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.block0(z)
        x = self.block1(x) + self.upsample(x)
        x = self.block2(x) + self.upsample(x)
        x = self.block3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ndf, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.block0 = nn.Sequential(nn.Conv2d(n_classes + 1, ndf, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block1 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block3 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block4 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False))

    def forward(self, input):
        x = self.block0(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return x

# Loss weight for gradient penalty
lambda_gp = 10

dataloader = torch.utils.data.DataLoader(
    pdataData(opt.dataroot, 32, 10),
    batch_size=opt.batch_size,
    num_workers=0,
    shuffle=True)

# Initialize generator and discriminator
generator = Generator(ngf=10, nz=opt.latent_dim, n_classes=10).to(device)
discriminator = Discriminator(ndf=6,  n_classes=10).to(device)

if 0:
    torch.save(generator.state_dict(), 'g.temp')
    torch.save(discriminator.state_dict(), 'd.temp')
    exit()

def sample_image(n_row, batches_done, z=None):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    if z == None:
        z = torch.randn((n_row ** 2, opt.latent_dim), device=device)

    gen_imgs = generator(z)
    gen_imgs = colors[gen_imgs.argmax(1)].permute(0, 3, 1, 2)
    gen_imgs = F.interpolate(gen_imgs, scale_factor=4)
    save_image(gen_imgs.data, "pdata/%d.png" % batches_done, nrow=n_row, normalize=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones(real_samples.shape[0], 1, requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class_aux_loss = torch.nn.CrossEntropyLoss()
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        # Configure input
        real_imgs = imgs.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], opt.latent_dim, 1, 1).to(device)

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                if not os.path.isfile("pdata/true.png"):
                    real_imgs = colors[real_imgs.argmax(1)].permute(0, 3, 1, 2)
                    real_imgs = F.interpolate(real_imgs, scale_factor=4)
                    save_image(real_imgs.data, "pdata/true.png", nrow=5, normalize=True)
                sample_image(n_row=5, batches_done=batches_done)
                torch.save(generator.state_dict(), "pdata/%d_gen.pth" % batches_done)
                torch.save(discriminator.state_dict(), "pdata/%d_dis.pth" % batches_done)

            batches_done += opt.n_critic