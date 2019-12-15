import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size      # 64
        self.img_channels = img_channels    # 3

        self.fc1 = nn.Linear(latent_size, 768)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=(0,1), output_padding=(0,1))
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv11 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, output_padding=0)
        self.deconv11_bn = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=(0,1), output_padding=(0,1))
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv22 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, output_padding=0)
        self.deconv22_bn = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.deconv3_bn = nn.BatchNorm2d(32)
        self.deconv33 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1, output_padding=0)
        self.deconv33_bn = nn.BatchNorm2d(32)

        self.ext_deconv1 = nn.ConvTranspose2d(32, 16, 5, stride=2)
        self.extdeconv1_bn = nn.BatchNorm2d(16)
        self.ext_deconv11 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1, output_padding=0)
        self.extdeconv11_bn = nn.BatchNorm2d(16)
        self.ext_deconv2 = nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.extdeconv2_bn = nn.BatchNorm2d(8)
        self.ext_deconv22 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1, output_padding=0)
        self.extdeconv22_bn = nn.BatchNorm2d(8)
        self.ext_deconv3 = nn.ConvTranspose2d(8, 4, 3, stride=2)
        self.extdeconv3_bn = nn.BatchNorm2d(4)
        self.ext_deconv33 = nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1, output_padding=0)
        self.extdeconv33_bn = nn.BatchNorm2d(4)
        self.ext_deconv4 = nn.ConvTranspose2d(4, img_channels, 4, stride=2)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))

        x = x.view(x.size(0), 256, 1, 3)
       
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv11_bn(self.deconv11(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv22_bn(self.deconv22(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv33_bn(self.deconv33(x)))
        
        x = F.relu(self.extdeconv1_bn(self.ext_deconv1(x)))
        x = F.relu(self.extdeconv11_bn(self.ext_deconv11(x)))
        x = F.relu(self.extdeconv2_bn(self.ext_deconv2(x)))
        x = F.relu(self.extdeconv22_bn(self.ext_deconv22(x)))
        x = F.relu(self.extdeconv3_bn(self.ext_deconv3(x)))
        x = F.relu(self.extdeconv33_bn(self.ext_deconv33(x)))

        reconstruction = F.sigmoid(self.ext_deconv4(x))
        
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.ext_conv1 = nn.Conv2d(img_channels, 4, 3, stride=2)
        self.extconv1_bn = nn.BatchNorm2d(4)
        self.conv11 = nn.Conv2d(4, 4, 3, stride = 1, padding = 1) #!!!!!!
        self.conv11_bn = nn.BatchNorm2d(4)
        self.ext_conv2 = nn.Conv2d(4, 8, 3, stride=2)
        self.extconv2_bn = nn.BatchNorm2d(8)
        self.conv22 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1) #!!!!!!
        self.conv22_bn = nn.BatchNorm2d(8)
        self.ext_conv3 = nn.Conv2d(8, 16, 3, stride=2)
        self.extconv3_bn = nn.BatchNorm2d(16)
        self.conv33 = nn.Conv2d(16, 16, 3, stride = 1, padding = 1) #!!!!!!
        self.conv33_bn = nn.BatchNorm2d(16)

        self.conv1 = nn.Conv2d(16, 32, 4, stride=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv44 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1) #!!!!!!
        self.conv44_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv55 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1) #!!!!!!
        self.conv55_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv66 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1) #!!!!!!
        self.conv66_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(768, latent_size)
        self.fc_logsigma = nn.Linear(768, latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        
        x = F.relu(self.extconv1_bn(self.ext_conv1(x)))
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.extconv2_bn(self.ext_conv2(x)))
        x = F.relu(self.conv22_bn(self.conv22(x)))
        x = F.relu(self.extconv3_bn(self.ext_conv3(x)))
        x = F.relu(self.conv33_bn(self.conv33(x)))
        
       
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv44_bn(self.conv44(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv55_bn(self.conv55(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv66_bn(self.conv66(x)))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class VAE(nn.Module):
    
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
      
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
