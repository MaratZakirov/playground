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
import numpy as np

torch.manual_seed(50)
colors = torch.rand(11, 3)
colors[0, :] = 0

from PIL import Image

os.makedirs("birka", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--dataroot', default='', required=False, help='path to dataset')
parser.add_argument('--layout', required=True, help='whenever rotation affects label')
parser.add_argument('--genpth', required=False, help='whenever rotation affects label')
parser.add_argument('--dispth', required=False, help='whenever rotation affects label')
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

def showImgTensor(t):
    transforms.ToPILImage()(colors[t.argmax(0)].permute(2, 0, 1)).resize((128, 128)).show()

class itemDataset(torch.utils.data.Dataset):
    def __init__(self, root, n_classes, size=512):
        self.n_classes = n_classes
        self.img_size = size
        self.size = size // 16
        self.img_files = open(root).readlines()
        self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        with open(root, "r") as file:
            self.img_files = [f.rstrip() for f in file.readlines()]
            # TODO get rid off all empty labels
            self.img_files = filterEmpty(self.img_files)

        self.label_files = [
            path.replace("images", "labels8").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

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
        tlayout = torch.zeros(self.n_classes + 1, self.size, self.size)
        tlayout[0, :, :] = 1.0
        tlayout[0, ij[:, 1], ij[:, 0]] = 0.0
        tlayout[cl + 1, ij[:, 1], ij[:, 0]] = 1.0

        # Loading image
        img = Image.open(self.label_files[item].
                         replace('labels8', 'images').
                         replace('.txt', '.jpg')).resize((self.img_size, self.img_size))
        img = img.rotate(90 * radian / (3.14 / 2))

        if 0:
            img.show()
            showImgTensor(tlayout)
            exit()

        return self.transform(img), tlayout

    def __len__(self):
        return len(self.img_files)

class GeneratorLayout(nn.Module):
    def __init__(self, ngf, nz, n_classes):
        super(GeneratorLayout, self).__init__()

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

class Generator(nn.Module):
    def __init__(self, ngf, nz, nc, n_layout):
        super(Generator, self).__init__()

        self.n_layout = n_layout
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

    def forward(self, z, z_layout):
        # Setting up appropriate prefix

        z[:, :self.n_layout] = z_layout
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

        # Main backbone
        self.block0 = self._basic_block(nc, ndf // 2, ndf)
        self.block1 = self._basic_block(ndf, ndf, ndf * 2)
        self.block2 = self._basic_block(ndf * 2, ndf * 2, ndf * 4)
        self.block3 = self._basic_block(ndf * 4, ndf * 4, ndf * 8)

        # Detection head
        self.det_head = nn.Sequential(nn.Conv2d(ndf * 8, self.n_classes + 1, 1),
                                      nn.LogSoftmax(dim=1))

        # Validation head
        self.val_head = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(ndf * 2, ndf, 4, 2, 1, bias=False),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(ndf, 1, 4, 1, 0, bias=False))

    def _basic_block(self, inch, hich, outch):
        return nn.Sequential(
            nn.Conv2d(inch, hich, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hich, outch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        x = self.block0(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x_det = self.det_head(x)
        x_val = self.val_head(x)
        x_val = x_val.view(x_val.size(0), -1)
        return x_val, x_det

# Loss weight for gradient penalty
lambda_gp = 10

dataloader = torch.utils.data.DataLoader(
    itemDataset(opt.dataroot, 10, 512),
    batch_size=opt.batch_size,
    num_workers=4,
    shuffle=True)
nc = 3

genLay = GeneratorLayout(ngf=32, nz=10, n_classes=10).eval().to(device)
genLay.load_state_dict(torch.load(opt.layout))

# Initialize generator and discriminator
generator = Generator(ngf=8, nz=opt.latent_dim, nc=nc, n_layout=10).to(device)
discriminator = Discriminator(ndf=20, nc=nc, n_classes=10).to(device)

#torch.save(discriminator.state_dict(), 'AAA_discr.whole')
#torch.save(discriminator.val_head.state_dict(), 'AAA_val.head')
#torch.save(generator.state_dict(), 'AAA_gene.whole')
#exit()

if opt.genpth != None and opt.dispth != None:
    generator.load_state_dict(torch.load(opt.genpth))
    discriminator.load_state_dict(torch.load(opt.dispth))

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z_layout = torch.randn((n_row ** 2, 10))
    z = torch.randn((n_row ** 2, opt.latent_dim), device=device)

    gen_imgs = generator(z, z_layout)
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

def class_aux_loss(input, target):
    assert input.shape == target.shape
    nB, nC, nH, nW = input.shape

    target = target.permute(0, 2, 3, 1).argmax(3).view(nB * nH * nW)
    input = input.permute(0, 2, 3, 1).reshape(nB * nH * nW, nC)

    return 1000 * torch.nn.NLLLoss()(input, target)

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, tlayout) in enumerate(dataloader):
        # Configure input
        real_imgs = imgs.to(device)
        tlayout = tlayout.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], opt.latent_dim).to(device)

        # generating target layout for fake images
        z_layout = torch.randn(imgs.shape[0], 10).to(device)
        tfakelayout = genLay(z_layout)

        # Generate a batch of images
        fake_imgs = generator(z, z_layout)

        # Real images
        real_validity, real_layout = discriminator(real_imgs)
        # Fake images
        fake_validity, fake_layout = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Real Cross Entropy
        real_lay_err = class_aux_loss(real_layout, tlayout)
        # Fake Cross Entropy
        fake_lay_err = class_aux_loss(fake_layout, tfakelayout)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty \
                 + real_lay_err + fake_lay_err

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z, z_layout)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity, fake_layout = discriminator(fake_imgs)
            # Fake Cross Entropy
            fake_lay_err = class_aux_loss(fake_layout, tfakelayout)
            g_loss = -torch.mean(fake_validity) + fake_lay_err

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [R_det: %f] [F_det: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(),
                   real_lay_err.item(), fake_lay_err.item())
            )

            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=5, batches_done=batches_done)
                torch.save(generator.state_dict(), "birka/%d_gen.pth" % batches_done)
                torch.save(discriminator.state_dict(), "birka/%d_dis.pth" % batches_done)

            batches_done += opt.n_critic