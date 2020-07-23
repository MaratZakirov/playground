import torch
from torch.nn import functional as F
import argparse
from torch.utils.data import DataLoader
from datagen import *
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR

class dataVAE(torch.nn.Module):
    def __init__(self, isize=160, zsize=64):
        super(dataVAE, self).__init__()

        self.isize = isize
        self.zsize = zsize

        self.encoder = nn.Sequential(nn.Conv2d(3,        zsize//8, 7, stride=2),
                                     nn.BatchNorm2d(zsize//8), nn.ReLU(),
                                     nn.Conv2d(zsize//8, zsize//4, 7, stride=2),
                                     nn.BatchNorm2d(zsize//4), nn.ReLU(),
                                     nn.Conv2d(zsize//4, zsize//2, 5, stride=2),
                                     nn.BatchNorm2d(zsize//2), nn.ReLU(),
                                     nn.Conv2d(zsize//2, zsize,    5, stride=2),
                                     nn.BatchNorm2d(zsize), nn.ReLU(),
                                     nn.Conv2d(zsize,    zsize*2,  5, stride=2))

        self.decoder = nn.Sequential(nn.ConvTranspose2d(zsize,     zsize,     4, stride=1, padding=0),
                                     nn.BatchNorm2d(zsize), nn.ReLU(),
                                     nn.ConvTranspose2d(zsize,     zsize//2,  4, stride=2, padding=0),
                                     nn.BatchNorm2d(zsize//2), nn.ReLU(),
                                     nn.ConvTranspose2d(zsize//2,  zsize//4,  4, stride=2, padding=1),
                                     nn.BatchNorm2d(zsize//4), nn.ReLU(),
                                     nn.ConvTranspose2d(zsize//4,  zsize//8,  4, stride=2, padding=1),
                                     nn.BatchNorm2d(zsize//8), nn.ReLU(),
                                     nn.ConvTranspose2d(zsize//8,  zsize//16, 4, stride=2, padding=1),
                                     nn.BatchNorm2d(zsize//16), nn.ReLU(),
                                     nn.ConvTranspose2d(zsize//16, 3,         4, stride=2, padding=1))

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x[:, :x.size(1)//2], x[:, x.size(1)//2:]

    def decode(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_recon = self.decoder(z)
        return x_recon

    def sample(self, mu, sigma):
        return mu + sigma * torch.randn(sigma.shape).to(mu.device)

    def forward(self, x, class_num):
        assert x.size(2) == x.size(3) == self.isize, 'image size check failed'
        mu, sigma = self.encode(x)
        assert mu.size(1) == sigma.size(1) == self.zsize, 'Encoder size check failed'
        z = self.sample(mu, sigma)
        x_recon = self.decode(z)
        assert x_recon.shape == x.shape, 'Decoder size check failed'
        return x_recon, mu, sigma

def lossfunc(img_t, img_recon, sigma, mu):
    REC = F.mse_loss(img_recon, img_t)
    KLD = -0.5 * (1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)).mean()
    return REC + KLD

def train(model, train_loader):
    model.train()

    L = 0

    for batch_i, (x, class_num) in enumerate(train_loader):
        optimizer.zero_grad()

        x_recon, mu, sigma = model(x.to(device), class_num)
        loss = lossfunc(x.to(device), x_recon, mu, sigma)

        loss.backward()
        optimizer.step()

        L += loss.item()

        if batch_i % 10 == 0:
            print('\t', batch_i, 'loss:', loss.item())

    return L/len(train_loader)

def test(model, test_loader):
    model.eval()

    L = 0

    mu_s    = []
    sigma_s = []

    with torch.no_grad():
        for batch_i, (x, class_num) in enumerate(test_loader):
            x_recon, mu, sigma = model(x.to(device), class_num)
            loss = lossfunc(x.to(device), x_recon, mu, sigma)
            L += loss.item()
            if batch_i % 10 == 0:
                print('\t', batch_i, 'loss:', loss.item())
            mu_s.append(mu.cpu())
            sigma_s.append(sigma.cpu())

    mu_s = torch.cat(mu_s)
    sigma_s = torch.cat(sigma_s)

    print('||||||||||| STAT REPORT ||||||||||||||')
    print('E(z_mean)', mu_s.mean(dim=0))
    print('E(z_sigma)', sigma_s.mean(dim=0))
    print('Divergence of:')
    print('S(z_mean)', mu_s.std(dim=0))
    print('S(z_sigma)', sigma_s.std(dim=0))
    print('||||||||||| END STAT REPORT ||||||||||')

    return L/len(test_loader)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', type=str, default='/home/marat/dataset/photo_birka/part1/train.txt',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test', type=str, default='/home/marat/dataset/photo_birka/part1/test.txt.true',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pretrained', type=str, default='vae2.model',
                    help='loading pretrained model')
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if len(args.pretrained) > 0:
    model = torch.load(args.pretrained).to(device)
else:
    model = dataVAE().to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

train_loader = DataLoader(dataGen(args.train, device), batch_size=args.batch_size,
                          collate_fn=collate_vae, shuffle=True)
test_loader = DataLoader(dataGen(args.test, device), batch_size=args.batch_size,
                         collate_fn=collate_vae, shuffle=False)

for epoch in range(args.epochs):
    print('=== Epoch:', epoch, '===')
    #train_loss = train(model, train_loader)
    #print('>> Epoch', epoch, 'train loss:', train_loss)
    test_loss = test(model, test_loader)
    print('>> Epoch', epoch, 'test loss:', test_loss)
    exit()
    scheduler.step()

torch.save(model, 'vae.model')