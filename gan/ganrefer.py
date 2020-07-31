from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=True, help='debugging mode')
parser.add_argument('--dataroot', default='/mnt/hugedisk/data/ganbirka128/', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./ganres', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf, exist_ok=True)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# folder dataset
dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=0)

device = torch.device("cuda" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, n_classes):
        super(Generator, self).__init__()

        self.n_classes = n_classes
        self.upsample = lambda x : nn.Upsample(scale_factor=2)(x[:, :x.size(1) // 2])
        self.block0 = nn.Sequential(nn.ConvTranspose2d(nz + n_classes, ngf * 8, 4, 1, 0, bias=False),
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

    def forward(self, z, class_n):
        emb = torch.zeros(z.size(0), self.n_classes).to(device)
        i = torch.arange(len(class_n)).to(device)
        emb[i, class_n] = 1.0
        z = torch.cat([z, emb[..., None, None]], dim=1)
        x = self.block0(z)
        x = self.upsample(x) + self.block1(x)
        x = self.upsample(x) + self.block2(x)
        x = self.upsample(x) + self.block3(x)
        x = self.upsample(x) + self.block4(x)
        x = self.block5(x)
        return x


netG = Generator(len(dataset.classes)).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.block0 = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block1 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block3 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf * 8),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.block4 = nn.Sequential(nn.Conv2d(ndf * 8, n_classes + 1, 8, 1, 0, bias=False))

    def forward(self, input):
        x = self.block0(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x_class = nn.Softmax()(x[:, :self.n_classes])
        x_fr = nn.Sigmoid()(x[:, self.n_classes:])
        return x_fr, x_class

netD = Discriminator(len(dataset.classes)).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

bce = nn.BCELoss()
cse = nn.CrossEntropyLoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
fixed_class = torch.arange(len(dataset.classes), device=device).repeat(len(dataset.classes))

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.dry_run:
    opt.niter = 1

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        real_class = data[1].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output, outclass = netD(real_cpu)
        errD_real = bce(output, label) + cse(outclass, real_class)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fakeclass = torch.randint(low=0, high=len(dataset.classes), size=(batch_size, ), device=device)
        fake = netG(noise, fakeclass)
        label.fill_(fake_label)
        output, outclass = netD(fake.detach())
        errD_fake = bce(output, label) + cse(outclass, fakeclass)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output, outclass = netD(fake)
        errG = bce(output, label) + cse(outclass, fakeclass)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i == 0:
            vutils.save_image(real_cpu,
                              '%s/real_samples.png' % opt.outf,
                              normalize=True, nrow=10)
            fake = netG(fixed_noise, fixed_class)
            vutils.save_image(fake.detach(),
                              '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True, nrow=10)

        if opt.dry_run:
            break

    # do checkpointing
    if epoch % 20 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))