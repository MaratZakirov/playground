import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.datasets as dset

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--dataroot', default='', required=False, help='path to dataset')
parser.add_argument('--outdir', default='images', required=False, help='path to output')
opt = parser.parse_args()
print(opt)

os.makedirs(opt.outdir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self, ngf, nz, nc, n_classes):
        super(Generator, self).__init__()

        self.n_classes = n_classes
        self.upsample = lambda x : nn.Upsample(scale_factor=2)(x[:, :x.size(1) // 2])
        self.block0 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                                    nn.BatchNorm2d(ngf * 8),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block4 = nn.Sequential(nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf // 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block5 = nn.Sequential(nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
                                    nn.Tanh())

    def forward(self, z, z_class):
        # Setting up appropriate prefix
        z[:, :self.n_classes] = 0
        z[torch.arange(len(z_class)).to(device), z_class] = 1.0
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.block0(z)
        x = self.block1(x) + self.upsample(x)
        x = self.block2(x) + self.upsample(x)
        x = self.block3(x) + self.upsample(x)
        x = self.block4(x) + self.upsample(x)
        x = self.block5(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.block0 = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block1 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block3 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block4 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block5 = nn.Sequential(nn.Conv2d(ndf * 16, n_classes + 1, 4, 1, 0, bias=False))

    def forward(self, input):
        x = self.block0(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x_class = x[:, :self.n_classes]
        validity = x[:, self.n_classes:]
        return validity, x_class


# Loss weight for gradient penalty
lambda_gp = 10

if opt.dataroot == '':
    # Configure data loadergeos.makedirs("./data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    nc = 1
else:
    dataloader = torch.utils.data.DataLoader(
        dset.ImageFolder(root=opt.dataroot,
                         transform=transforms.Compose([
                             transforms.Resize(opt.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                         ),
        batch_size=opt.batch_size,
        num_workers=16,
        shuffle=True,
    )
    nc = 3

# Initialize generator and discriminator
generator = Generator(ngf=22, nz=opt.latent_dim, nc=nc, n_classes=10).to(device)
discriminator = Discriminator(ndf=11, nc=nc, n_classes=10).to(device)

if 0:
    torch.save(generator.state_dict(), 'g.temp')
    torch.save(discriminator.state_dict(), 'd.temp')
    exit()

def sample_image(n_row, batches_done, z=None):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z_class = torch.arange(10).repeat(n_row).to(device)

    if z == None:
        z = torch.randn((n_row ** 2, opt.latent_dim), device=device)

    gen_imgs = generator(z, z_class)
    save_image(gen_imgs.data, os.path.join(opt.outdir, "%d.png" % batches_done), nrow=n_row, normalize=True)

if 1:
    import ganutils
    with torch.no_grad():
        generator.load_state_dict(torch.load('images128/100000_gen.pth'))
        discriminator.load_state_dict(torch.load('images128/100000_dis.pth'))
        generator.eval()
        discriminator.eval()
        ganutils.produceNewBirkas(generator, discriminator, device, opt.latent_dim, 100)
    exit()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)

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
    for i, (imgs, target) in enumerate(dataloader):

        # Configure input
        real_imgs = imgs.to(device)
        real_target = target.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], opt.latent_dim, 1, 1).to(device)

        # Generate a batch of images
        fake_target = torch.randint(0, 10, size=(imgs.shape[0], )).to(device)
        fake_imgs = generator(z, fake_target)

        # Real images
        real_validity, real_class = discriminator(real_imgs)
        # Fake images
        fake_validity, fake_class = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Real Cross Entropy
        real_class_err = class_aux_loss(real_class, real_target)
        # Fake Cross Entropy
        fake_class_err = class_aux_loss(fake_class, fake_target)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty\
                 + real_class_err + fake_class_err

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z, fake_target)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity, fake_class = discriminator(fake_imgs)
            # Fake Cross Entropy
            fake_class_err = class_aux_loss(fake_class, fake_target)
            g_loss = -torch.mean(fake_validity) + fake_class_err

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)
                torch.save(generator.state_dict(), os.path.join(opt.outdir, "%d_gen.pth" % batches_done))
                torch.save(discriminator.state_dict(), os.path.join(opt.outdir, "%d_dis.pth" % batches_done))

            batches_done += opt.n_critic