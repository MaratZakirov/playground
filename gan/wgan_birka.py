import argparse
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.datasets as dset

import torch.nn as nn
import torch.autograd as autograd
import torch

from PIL import Image

os.makedirs("birka", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--dataroot', default='', required=False, help='path to dataset')
parser.add_argument('--flag0', default='', required=False, help='whenever rotation affects label')
opt = parser.parse_args()
print(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class itemDataset(torch.utils.data.Dataset):
    def __init__(self, root, flag0, size=512):
        self.size = size
        self.flag0 = flag0
        self.img_files = open(root).readlines()
        self.transform = transforms.Compose([
                             transforms.RandomCrop(size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __getitem__(self, item):
        img_file = self.img_files[item].rstrip()
        img = Image.open(img_file)

        w, h = img.width, img.height

        if min(w, h) < self.size or min(w, h) > self.size * 1.4:
            coef = torch.clamp(torch.tensor(min(w, h)), self.size, self.size * 1.4) * (1.0 /  min(w, h))
            img = img.resize((int(w * coef), int(h * coef)))

        rot = torch.randint(1, 5, (1, ))[0]
        img = img.rotate((rot - 1) * 90)
        return self.transform(img), torch.tensor(0) if self.flag0 in img_file else rot



    def __len__(self):
        return len(self.img_files)


class Generator(nn.Module):
    def __init__(self, ngf, nz, nc, n_classes):
        super(Generator, self).__init__()

        self.n_classes = n_classes
        self.upsample = lambda x : nn.Upsample(scale_factor=2)(x[:, :x.size(1) // 2])
        self.block0 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 64, 4, 1, 0, bias=False),
                                    nn.BatchNorm2d(ngf * 64),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block1 = nn.Sequential(nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 32),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block2 = nn.Sequential(nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 16),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block3 = nn.Sequential(nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 8),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block4 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block5 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block6 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block7 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
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
        x = self.block5(x) + self.upsample(x)
        x = self.block6(x) + self.upsample(x)
        x = self.block7(x)
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
        self.block5 = nn.Sequential(nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block6 = nn.Sequential(nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block7 = nn.Sequential(nn.Conv2d(ndf * 64, n_classes + 1, 4, 1, 0, bias=False))

    def forward(self, input):
        x = self.block0(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
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
        itemDataset(opt.dataroot, opt.flag0, 512),
        batch_size=opt.batch_size,
        num_workers=0,
        shuffle=True)
    nc = 3

# Initialize generator and discriminator
generator = Generator(ngf=8, nz=opt.latent_dim, nc=nc, n_classes=5).to(device)
discriminator = Discriminator(ndf=8, nc=nc, n_classes=5).to(device)

if 0:
    torch.save(generator.state_dict(), 'g.temp')
    torch.save(discriminator.state_dict(), 'd.temp')
    exit()

def sample_image(n_row, batches_done, z=None):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z_class = torch.arange(n_row).repeat(n_row).to(device)

    if z == None:
        z = torch.randn((n_row ** 2, opt.latent_dim), device=device)

    gen_imgs = generator(z, z_class)
    save_image(gen_imgs.data, "birka/%d.png" % batches_done, nrow=n_row, normalize=True)

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
                sample_image(n_row=5, batches_done=batches_done)
                torch.save(generator.state_dict(), "birka/%d_gen.pth" % batches_done)
                torch.save(discriminator.state_dict(), "birka/%d_dis.pth" % batches_done)

            batches_done += opt.n_critic