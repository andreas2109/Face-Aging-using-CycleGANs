import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import common.utils as utils
from common.dataset import FaceAgingDataset
from common.utils import load_checkpoint, save_checkpoint, ReplayBuffer
from generator import Generator
from discriminator import Discriminator


####### Function to adjust learning rate ########
def adjust_learning_rate(optimizer, epoch, lr_list):
    decay_start_epoch = 100
    decay_end_epoch = 200
    initial_lr = utils.LEARNING_RATE

    if epoch < decay_start_epoch:
        lr = initial_lr
    elif epoch <= decay_end_epoch:
        decay_epochs = decay_end_epoch - decay_start_epoch + 1
        lr = initial_lr * (1 - (epoch - decay_start_epoch + 1) / decay_epochs)
    else:
        lr = initial_lr * (1 - (decay_end_epoch - decay_start_epoch + 1) / decay_epochs)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    lr_list.append(lr)
#########################################

fake_young_buffer = ReplayBuffer()
fake_old_buffer = ReplayBuffer()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def train_func(disc_Y, disc_O, gen_Y, gen_O, opt_disc_Y,opt_disc_O, opt_gen, g_scaler, d_Y_scaler, d_O_scaler, L1, mse, loader, fake_young_buffer, fake_old_buffer):
    Y_reals = 0
    Y_fakes = 0
    loss_Disc = 0
    loss_Gen = 0
    loss_cycle = 0
    loss_identity = 0
    loop = tqdm(loader, leave=True)
    for idx, (old, young) in enumerate(loop):
        old = old.to(utils.DEVICE)
        young = young.to(utils.DEVICE)
        loss_Disc = 0
        loss_identity = 0
        loss_cycle = 0
        loss_Gen = 0
        # Train Discriminator Y & O
        with torch.cuda.amp.autocast():
            opt_disc_Y.zero_grad()
            fake_young = gen_Y(old)
            fake_young_ = fake_young_buffer.push_and_pop(fake_young)
            D_Y_real = disc_Y(young)
            D_Y_fake= disc_Y(fake_young_.detach())
            Y_reals += D_Y_real.mean().item()
            Y_fakes += D_Y_fake.mean().item()
            D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
            D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
            D_Y_loss = D_Y_real_loss + D_Y_fake_loss

            fake_old = gen_O(young)
            fake_old_ = fake_old_buffer.push_and_pop(fake_old)
            D_O_real= disc_O(old)
            D_O_fake= disc_O(fake_old_.detach())
            D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))
            D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))
            D_O_loss = D_O_real_loss + D_O_fake_loss

            D_loss = (D_Y_loss + D_O_loss) / 2
            loss_Disc += D_loss.item()
        
        opt_disc_Y.zero_grad()
        opt_disc_O.zero_grad()
        d_Y_scaler.scale(D_Y_loss).backward()
        d_O_scaler.scale(D_O_loss).backward()
        d_Y_scaler.step(opt_disc_Y)
        d_O_scaler.step(opt_disc_O)
        d_Y_scaler.update()
        d_O_scaler.update()

        # Train Generator Y & O
        with torch.cuda.amp.autocast():
            D_Y_fake  = disc_Y(fake_young)
            D_O_fake  = disc_O(fake_old)
            
            # Generator Loss for Domain Y
            loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))
            loss_G_O = mse(D_O_fake, torch.ones_like(D_O_fake))
            
            # Cycle loss
            cycle_old = gen_O(fake_young)
            cycle_young = gen_Y(fake_old)
            cycle_old_loss = L1(old, cycle_old)
            cycle_young_loss = L1(young, cycle_young)
            loss_cycle += (cycle_old_loss + cycle_young_loss).item()

            # Identity loss (optional)
            identity_old = gen_O(old)       
            identity_young = gen_Y(young)    
            identity_old_loss = L1(old, identity_old)
            identity_young_loss = L1(young, identity_young)
            loss_identity +=(identity_old_loss.item() + identity_young_loss.item()) 
        
            G_loss = (
                loss_G_O
                + loss_G_Y
                + cycle_old_loss * utils.LAMBDA_CYCLE
                + cycle_young_loss * utils.LAMBDA_CYCLE
                + identity_young_loss * utils.LAMBDA_IDENTITY * utils.LAMBDA_CYCLE 
                + identity_old_loss * utils.LAMBDA_IDENTITY * utils.LAMBDA_CYCLE 
            )
            loss_Gen +=  loss_G_O + loss_G_Y

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        # Save generated images and plot attention maps
        if idx % 200 == 0:
            save_image(young * 0.5 + 0.5, f"saved_images/young{idx}.png")
            save_image(fake_young * 0.5 + 0.5, f"saved_images/fake_young_{idx}.png")
            save_image(old * 0.5 + 0.5, f"saved_images/old_{idx}.png")
            save_image(fake_old * 0.5 + 0.5, f"saved_images/fake_old_{idx}.png")

         # Save the reconstructed images
            save_image(cycle_young * 0.5 + 0.5, f"saved_images/recon_young{idx}.png")
            save_image(cycle_old * 0.5 + 0.5, f"saved_images/recon_old{idx}.png")
 
        loop.set_postfix(Y_real=Y_reals / (idx + 1), Y_fake=Y_fakes / (idx + 1), loss_gen = loss_Gen / (idx + 1), loss_disc = loss_Disc / (idx + 1), loss_cyc =loss_cycle / (idx + 1), loss_id =  loss_identity / (idx + 1))
    return loss_Disc, loss_identity, loss_cycle, loss_Gen

def main():
    if not os.path.exists('saved_images'):
        os.makedirs('saved_images')
    
    disc_Y = Discriminator(img_channels=3).to(utils.DEVICE)
    disc_O = Discriminator(img_channels=3).to(utils.DEVICE)
    gen_Y = Generator(img_channels=3, num_residuals=9).to(utils.DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(utils.DEVICE)
    
    disc_Y.apply(weights_init_normal)
    disc_O.apply(weights_init_normal)
    gen_Y.apply(weights_init_normal)
    gen_O.apply(weights_init_normal)

    opt_disc_Y = optim.Adam(
        disc_Y.parameters(),
        lr=utils.LEARNING_RATE,
        betas= (0.5, 0.999)
    )
    opt_disc_O = optim.Adam(
        disc_Y.parameters(),
        lr=utils.LEARNING_RATE,
        betas= (0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_Y.parameters()) + list(gen_O.parameters()),
        lr=utils.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    if utils.LOAD_MODEL:
        start_epoch = load_checkpoint(utils.CHECKPOINT_GEN_Y, gen_Y, opt_gen, utils.LEARNING_RATE)
        _ = load_checkpoint(utils.CHECKPOINT_GEN_O, gen_O, opt_gen, utils.LEARNING_RATE)
        _ = load_checkpoint(utils.CHECKPOINT_CRITIC_Y, disc_Y, opt_disc_Y, utils.LEARNING_RATE)
        _ = load_checkpoint(utils.CHECKPOINT_CRITIC_O, disc_O, opt_disc_O, utils.LEARNING_RATE)
    else:
        start_epoch = 0

    train_dataset = FaceAgingDataset(root_old=utils.TRAIN_DIR + "/trainB", root_young=utils.TRAIN_DIR + "/trainA", transform=utils.transforms)
    train_loader = DataLoader(train_dataset, batch_size=utils.BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=utils.NUM_WORKERS)
    val_dataset = FaceAgingDataset(root_old=utils.VAL_DIR + "/valB", root_young=utils.VAL_DIR + "/valA", transform=utils.transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=utils.NUM_WORKERS)

    g_scaler = torch.cuda.amp.GradScaler()
    d_Y_scaler = torch.cuda.amp.GradScaler()
    d_O_scaler = torch.cuda.amp.GradScaler()


    gen_lr_list = []
    discy_lr_list = []
    gen_loss_list= []
    disc_loss_list = []
    cycle_loss_list = []
    identity_loss_list = []

    for epoch in range(utils.NUM_EPOCHS):
        adjust_learning_rate(opt_gen, epoch, gen_lr_list)
        adjust_learning_rate(opt_disc_Y, epoch, discy_lr_list)
        adjust_learning_rate(opt_disc_O, epoch, discy_lr_list)
        print(f"Epoch [{epoch + 1}/{utils.NUM_EPOCHS}] - Generator LR: {gen_lr_list[-1]:.6f}, Discriminator LR: {discy_lr_list[-1]:.6f}")
        loss_Disc, loss_identity, loss_cycle, loss_Gen = train_func(disc_Y, disc_O, gen_Y, gen_O, opt_disc_Y,opt_disc_O, opt_gen, g_scaler, d_Y_scaler, d_O_scaler, L1, mse, train_loader, fake_young_buffer, fake_old_buffer)
        gen_loss_list.append(loss_Gen)
        disc_loss_list.append(loss_Disc)
        cycle_loss_list.append(loss_cycle)
        identity_loss_list.append(loss_identity)
        
        if utils.SAVE_MODEL:
            save_checkpoint(gen_Y, opt_gen, epoch,  filename=utils.CHECKPOINT_GEN_Y)
            save_checkpoint(gen_O, opt_gen, epoch,  filename=utils.CHECKPOINT_GEN_O)
            save_checkpoint(disc_Y, opt_disc_Y, epoch, filename=utils.CHECKPOINT_CRITIC_Y)
            save_checkpoint(disc_O, opt_disc_O, epoch, filename=utils.CHECKPOINT_CRITIC_O)
        with open('losses.txt', 'a') as f:
            f.write(f"Epoch [{epoch + 1}/{utils.NUM_EPOCHS}] - Loss_Gen: {loss_Gen:.4f}, Loss_Disc: {loss_Disc:.4f}, Loss_Cycle: {loss_cycle:.4f}, Loss_Identity: {loss_identity:.4f}\n")

if __name__ == "__main__":
    main()