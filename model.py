import random
import numpy
import torch
import os
from DLStudio import *



import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvtF
import imageio
import time
import matplotlib.pyplot as plt






import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = self.relu1(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = self.relu2(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = self.relu3(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = self.relu4(self.bn4(self.conv4(x)))
        # print(x.shape)       
        x = self.sigmoid(self.conv5(x))
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(100, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * 8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(64 * 4, 64 * 2, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(64 * 2, 64, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.tanh(self.conv5(x))
        return x

class GANTrainer:
    def __init__(self, train_dataloader, device='cuda', epochs=10, batch_size=128, lr=0.0002, results_dir='DCGAN_output'):
        self.results_dir = results_dir
        self.train_dataloader = train_dataloader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def weights_init(self, m):
        """
        Initialize the weights of the network modules.
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def load_model_weights(self, model, checkpoint_file):
        """
        Load model weights from a checkpoint file if it exists.
        """
        if checkpoint_file is not None:
            if os.path.isfile(checkpoint_file):
                print(f'Weight file found for {model.__class__.__name__}. Using the saved weights!')
                model.load_state_dict(torch.load(checkpoint_file))
            else:
                print(f'Weight file not found for {model.__class__.__name__}. Initializing the weights!')
                model.apply(self.weights_init)
        else:
                print(f'Weight file not provided for {model.__class__.__name__}. Initializing the weights!')


    
    def train(self, discriminator_weight=None, generator_weight=None):
        """
        Train the GAN model.
        """
        results_dir = self.results_dir
        os.makedirs(results_dir, exist_ok=True)
        criterion = nn.BCELoss()
        channels = 100
        discriminator = Discriminator().to(self.device)
        generator = Generator().to(self.device)

        self.load_model_weights(discriminator, discriminator_weight)
        self.load_model_weights(generator, generator_weight)

        #use this fixed noise to track the progress of dcgan later
        fixed_noise = torch.randn(self.batch_size, channels, 1, 1, device=self.device) 
        optimizer_dis = optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_gen = optim.Adam(generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        image_list=[]
        discriminator_losses=[]
        generator_losses=[]
        
        for epoch in range(self.epochs):
            gen_loss_per_epoch, dis_loss_per_epoch = [], []            
            for i, images in enumerate(self.train_dataloader):

                
                #first part
                real_images= images.to(self.device)
                batch_size= real_images.shape[0]
                # assert(batch_size==real_images.shape[0])
                label = torch.ones(batch_size)
                label= label.to(self.device)
                discriminator.zero_grad()
                pred_label= discriminator(real_images).squeeze([1,2,3]) #to get to the same shape
                loss_dis_real = criterion(pred_label, label)
                loss_dis_real.backward()
                
                #second part
                noise= torch.randn(batch_size, channels, 1, 1, device=self.device)
                fake_images= generator(noise)
                label.fill_(0) #replacing 1 with 0
                pred_label= discriminator(fake_images.detach()).squeeze([1,2,3]) #called detach as we don't want to update generator parameters
                loss_dis_fake= criterion(pred_label, label)
                loss_dis_fake.backward()
                loss_dis= loss_dis_real+loss_dis_fake
                optimizer_dis.step()  #updating the discriminator

                # Minimization Part for the update of generator
                generator.zero_grad()
                label.fill_(1) #replacing 0 with 1
                pred_label= discriminator(fake_images).squeeze([1,2,3])
                loss_gen= criterion(pred_label, label)
                loss_gen.backward()
                optimizer_gen.step()  #updating the generaor
                dis_loss_per_epoch.append(loss_dis)  #recording loss values
                gen_loss_per_epoch.append(loss_gen)
                generator_losses.append(loss_gen.item())
                discriminator_losses.append(loss_dis.item())
                
                #logging part based on run_gan_code DLStudio
                if i % 100 == 99:
                    dis_loss = torch.mean(torch.FloatTensor(dis_loss_per_epoch))
                    gen_loss = torch.mean(torch.FloatTensor(gen_loss_per_epoch))
                    print("epoch=%d/%d   iter=%4d   Dis Loss=%5.4f    Gen Loss=%5.4f" %
                          ((epoch+1), self.epochs, (i+1), dis_loss, gen_loss))
                    dis_loss_per_epoch, gen_loss_per_epoch = [], []

                if i % 500 == 0 or (epoch == self.epochs-1 and i == len(self.train_dataloader)-1): #track the progress every 500 iterations 
                                                                                                #and for the last iteration
                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()  # using detach to remove it from computational graph
                    image_list.append(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True))
            

                torch.save(discriminator.state_dict(), 'DCGANcheckpoints/netDcheckpoint.pt')
                torch.save(generator.state_dict(), 'DCGANcheckpoints/netGcheckpoint.pt')

        plt.figure(figsize=(10,5))    
        plt.title("Generator and Discriminator Loss During Training")    
        plt.plot(generator_losses, label="G")    
        plt.plot(discriminator_losses ,label="D") 
        plt.xlabel("iterations")   
        plt.ylabel("Loss")         
        plt.legend()          
        plt.savefig(results_dir + "/losses.png") 
        plt.show()          
        images_for_gif = [tvtF.to_pil_image(imgobj) for imgobj in images]
        #  Make an animated gif from the Generator output images stored in img_list:  
        imageio.mimsave(os.path.join(results_dir, "generation_animation.gif"), images_for_gif, fps=2)

