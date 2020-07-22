import torch
from torch.nn import functional as F
import argparse
from torch.utils.data import DataLoader
from datagen import *
from torch import optim, nn

class dataVAE(torch.nn.Module):
    def __init__(self):
        super(dataVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(50 * 50, 500), nn.BatchNorm1d(500), nn.ReLU(),
                                     nn.Linear(500, 200), nn.BatchNorm1d(200), nn.ReLU(),
                                     nn.Linear(200, 100))

        self.decoder = nn.Sequential(nn.Linear(50, 200), nn.BatchNorm1d(200), nn.ReLU(),
                                     nn.Linear(200, 500), nn.BatchNorm1d(500), nn.ReLU(),
                                     nn.Linear(500, 50 * 50))
        """
        self.encoder = nn.Sequential(nn.Conv2d(3, 8 * 2 , 7, stride=3), nn.BatchNorm2d(8*2), nn.ReLU(),
                                     nn.Conv2d(8 * 2, 16 * 2, 5, stride=2), nn.BatchNorm2d(16*2), nn.ReLU(),
                                     nn.Conv2d(16 * 2, 32 * 2, 5, stride=2), nn.BatchNorm2d(32 * 2), nn.ReLU(),
                                     nn.Conv2d(32 * 2, 64 * 2, 4, stride=2), nn.BatchNorm2d(64 * 2), nn.ReLU(),
                                     nn.Conv2d(64 * 2, 128 * 2, 3, stride=2))

        self.decoder = nn.Sequential(nn.ConvTranspose2d(2, 4, 5, stride=2, padding=0), nn.BatchNorm2d(4), nn.ReLU(),
                                     nn.ConvTranspose2d(4, 8, 4, stride=2, padding=0), nn.BatchNorm2d(8), nn.ReLU(),
                                     nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0), nn.BatchNorm2d(16), nn.ReLU(),
                                     nn.ConvTranspose2d(16, 3, 3, stride=2, padding=0))
        """

    def encode(self, x):
        x = x.view(x.size(0), 50 * 50)
        x = self.encoder(x)
        #x = x.view(x.size(0), -1)
        return x[:, :x.size(1)//2], x[:, x.size(1)//2:]

    def decode(self, z):
        #z = z.view(z.size(0), 2, 8, 8)
        x_recon = self.decoder(z)
        x_recon = x_recon.view(x_recon.size(0), 50, 50)
        return x_recon

    def sample(self, mu, sigma):
        return mu + sigma * torch.randn(sigma.shape).to(mu.device)

    def forward(self, x, class_num):
        mu, sigma = self.encode(x)
        z = self.sample(mu, sigma)
        x_recon = self.decode(z)
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

        x = x.mean(dim=1)

        x_recon, mu, sigma = model(x.to(device), class_num)

        loss = lossfunc(x.to(device), x_recon, mu, sigma)

        loss.backward()

        optimizer.step()

        L += loss.item()

        if batch_i % 20 == 0:
            print('\t', batch_i, 'loss:', loss.item())

    return L/len(train_loader)

def test(model, test_loader):
    model.eval()
    pass

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', type=str, default='/home/marat/dataset/photo_birka/part1/train.txt',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test', type=str, default='/home/marat/dataset/photo_birka/part1/test.txt',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

model = dataVAE().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

train_loader = DataLoader(dataGen(args.train, device), batch_size=args.batch_size,
                          collate_fn=collate_vae, shuffle=True)
test_loader = DataLoader(dataGen(args.test, device), batch_size=args.batch_size,
                         collate_fn=collate_vae, shuffle=False)

for epoch in range(10):#args.epochs):
    print('=== Epoch:', epoch, '===')
    train_loss = train(model, train_loader)
    #test_loss = test(model, test_loader)
    print('Overall epoch loss:', train_loss)