import torch
import torch.nn as nn
from loss import VGGLoss
from torch.utils.data import DataLoader
from arch import Generator, Discriminator
from dataset import MyDataset
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import datetime
from datetime import date, time as ttt
from tqdm import tqdm


time = datetime.datetime.now()
date_today = str(date.today().strftime("%d-%m-%y"))
log_path = f"./logs/{date_today}/{time}_log.txt"
test_transform = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
])
disc_log_fig = "./figures/training_disclossfig.png"
gen_log_fig = "./figures/training_genlossfig.png"
genonly_log_fig = "./figures/training_genonly_lossfig.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    os.mkdir("logs")
except:
    pass

try:
    os.mkdir("logs/" + date_today)
except:
    pass


log_path = f"./logs/{date_today}/{time}_log.txt"

with open(log_path, "a") as f:
    f.write(f"Time: {time} \n")

dataset = MyDataset(root_dir='/Users/shyam/Downloads/archive/DIV2K_train_HR/DIV2K_train_HR')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Initialize models
gen = Generator(upscale_factor = 4, num_blocks=16).to(device)
disc = Discriminator().to(device)

# Define loss functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
vgg_loss = VGGLoss()

# Define the path to the saved weights file
weight_path = "SRResNet_weight.pth"

# Load the saved weights
saved_G_weight = torch.load(weight_path,map_location=torch.device('cpu'))

# Load the saved weights into the generator network
gen.load_state_dict(saved_G_weight)

import torch.optim as optim

# Define learning rates
lr_gen = 1e-4
lr_disc = 1e-4

# Define the step size and gamma for lr_scheduler
step_size = 1e5
gamma = 0.1

# Set up optimizer and learning rate scheduler for the generator
opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.9, 0.999))
gen_lr_scheduler = optim.lr_scheduler.StepLR(opt_gen, step_size=step_size, gamma=gamma)

# Set up optimizer and learning rate scheduler for the discriminator
opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.9, 0.999))
disc_lr_scheduler = optim.lr_scheduler.StepLR(opt_disc, step_size=step_size, gamma=gamma)

# Define number of epochs and starting epoch
num_epochs = 1000
start_epoch = 0

gen.train()
disc.train()
disc_loss = []
loss_gen = []

for epoch in range(start_epoch, num_epochs):
    epoch_gen_loss = 0
    epoch_disc_loss = 0
    
    # Iterate over the data loader
    loop=tqdm(dataloader,leave=True)
    for i,dt in enumerate(loop):
        low_res = dt[0].to(device)
        high_res = dt[1].to(device)
        
        # Generate fake high-resolution images
        fake = gen(low_res)
        
        # Train the discriminator
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())  # Detach to prevent gradients from flowing back to the generator
        disc_loss_real = bce_loss(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (disc_loss_fake + disc_loss_real) / 2
        
        # Backpropagation and optimization for discriminator
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
        disc_lr_scheduler.step()
        
        # Accumulate discriminator loss
        epoch_disc_loss += (disc_loss_real.cpu() + disc_loss_fake.cpu()) / 2
        # Train generator
        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce_loss(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)  # Assuming vgg_loss is the VGG loss function
        content_loss = mse_loss(fake, high_res)

        # Total generator loss
        gen_loss = loss_for_vgg + adversarial_loss + content_loss
        #print(gen_loss)

        # Backpropagation and optimization for generator
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        gen_lr_scheduler.step()

        # Accumulate generator loss for the epoch
        epoch_gen_loss += loss_for_vgg.cpu() + adversarial_loss.cpu() + content_loss.cpu()
        #print(epoch_gen_loss)

    # Discriminator plotting
    with open(log_path, 'a') as f:
        f.write(f"disc loss for {epoch} is {epoch_disc_loss/len(dataloader)}\n")
    disc_loss.append(epoch_disc_loss.cpu().detach().numpy() / len(dataloader))
    fig = plt.figure()
    plt.plot(disc_loss)
    plt.savefig(disc_log_fig)
    plt.close(fig)

    # Generator plotting
    with open(log_path, 'a') as f:
        f.write(f"gen loss for {epoch} is {epoch_gen_loss/len(dataloader)}\n")
    loss_gen.append(epoch_gen_loss.cpu().detach().numpy() / len(dataloader))
    fig = plt.figure()
    plt.plot(loss_gen)
    plt.savefig(gen_log_fig)
    plt.close(fig)
    
    if (epoch+1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': gen.state_dict(),
            'optimizer_state_dict': opt_gen.state_dict(),
            'scheduler': gen_lr_scheduler.state_dict()
        }, f"./models/train_gen.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': disc.state_dict(),
            'optimizer_state_dict': opt_disc.state_dict(),
            'scheduler': disc_lr_scheduler.state_dict()
        }, f"./models/train_disc.pth")

        with open(log_path, 'a') as f:
            f.write(f"saved model for {epoch} \n")

   

