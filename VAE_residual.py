import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# dimension unchanging
def reparameterization(mean, var, device):
    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.sqrt(var) * epsilon


def gauss_gauss_kl(mean1, var1, mean2, var2):
    epsilon_val = 1e-8  # Small constant to avoid NaN
    _var2 = var2 + epsilon_val
    _kl = torch.log(_var2) - torch.log(var1) \
          + (var1 + (mean1 - mean2) ** 2) / _var2 - 1
    return 0.5 * torch.mean(torch.sum(_kl))


def gauss_unitgauss_kl(mean, var):
    return -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean ** 2 - var))


def rec_loss_norm4D(x, mean, var):
    return -torch.mean(
        torch.sum(-0.5 * ((x - mean) ** 2 / var + torch.log(var) + torch.log(torch.tensor(2 * torch.pi))),
                  dim=(1, 2, 3)))


def rec_loss_norm2D(x, mean, var):
    return -torch.mean(
        torch.sum(-0.5 * ((x - mean) ** 2 / var + torch.log(var) + torch.log(torch.tensor(2 * torch.pi))),
                  dim=1))


class resblock_enc(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size):
        super(resblock_enc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)

        self.model = nn.Sequential(
            nn.ReLU(),
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.Dropout(p=0.5),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

        self.bypass_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.)

        self.bypass = nn.Sequential(
            spectral_norm(self.bypass_conv),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

        self._initialize_weights()

    def forward(self, x):
        out_x = self.model(x) + self.bypass(x)
        return out_x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class first_resblock_enc(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size):
        super(first_resblock_enc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.Dropout(p=0.5),  # Dropout 추가
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

        self.bypass_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.)

        self.bypass = nn.Sequential(
            spectral_norm(self.bypass_conv),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

        self._initialize_weights()

    def forward(self, x):
        out_x = self.model(x) + self.bypass(x)
        return out_x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    def __init__(self, z_dim, ch, size, device):
        super(Encoder, self).__init__()
        self.device = device
        self.size = size

        self.enc_block_1 = first_resblock_enc(ch, ch * 2, pooling_size=(1, 2))  # 1, 128 ->  1, 64
        self.enc_block_2 = resblock_enc(ch * 2, ch * 4, pooling_size=(1, 2))  # 1, 64 -> 1, 32
        self.enc_block_3 = resblock_enc(ch * 4, ch * 8, pooling_size=(1, 2))  # 1, 32 -> 1, 16

        self.blocks = nn.Sequential(
            self.enc_block_1,
            self.enc_block_2,
            self.enc_block_3,
            nn.LeakyReLU(0.2, inplace=True)
        )

        conv_output_size = ch * 2 ** 3 * int(self.size / 2 ** 3)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu = nn.Sequential(
            nn.Linear(32, z_dim)
        )

        self.var = nn.Sequential(
            nn.Linear(32, z_dim),
            nn.Softplus()
        )

        self._initialize_weights()

    def forward(self, x):
        encoded = self.blocks(x)
        encoded = self.fc(encoded.view(-1, encoded.shape[1] * encoded.shape[2] * encoded.shape[3]))
        mu = self.mu(encoded)
        var = self.var(encoded)
        z = reparameterization(mu, var, self.device)
        return z, mu, var

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class resblock_enc_small(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(resblock_enc_small, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        # Ensure dimensions match for bypass addition
        self.adjust_dim = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.dropout(out)
        bypass_out = self.adjust_dim(x)  # Adjust dimension only if needed
        return self.activation(out + bypass_out)


class Encoder_w(nn.Module):
    def __init__(self, z_dim, n_label, device):
        super(Encoder_w, self).__init__()
        self.device = device

        # Residual blocks for small dimensions
        self.res_block1 = resblock_enc_small(n_label, z_dim * 2)
        self.res_block2 = resblock_enc_small(z_dim * 2, z_dim * 4)
        self.res_block3 = resblock_enc_small(z_dim * 4, z_dim * 2)
        self.res_block4 = resblock_enc_small(z_dim * 2, z_dim)

        # Output layers for mean and variance
        self.mu = nn.Sequential(
            nn.Linear(z_dim, z_dim)
        )
        self.var = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus()
        )

    def forward(self, w):
        out = self.res_block1(w)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        mu = self.mu(out)
        var = self.var(out)
        z = reparameterization(mu, var, self.device)
        return z, mu, var


class resblock_dec(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample):
        super(resblock_dec, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Upsample(scale_factor=up_sample, mode='bilinear', align_corners=True),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2,

        )

        self.bypass_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.)

        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=up_sample, mode='bilinear', align_corners=True),
            self.bypass_conv
        )

        self._initialize_weights()

    def forward(self, x):
        out_x = self.model(x) + self.bypass(x)
        return out_x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class Decoder(nn.Module):
    def __init__(self, z_dim, ch, size):
        super(Decoder, self).__init__()
        self.ch = ch
        self.size = size

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, ch * 2 ** 3 * int(size / 2 ** 3)),
            nn.BatchNorm1d(ch * 2 ** 3 * int(size / 2 ** 3)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dec_block_1 = resblock_dec(ch * 8, ch * 4, up_sample=(1, 2))  # 1,16 -> 1,32
        self.dec_block_2 = resblock_dec(ch * 4, ch * 2, up_sample=(1, 2))  # 1,32 -> 1, 64

        self.blocks = nn.Sequential(
            self.dec_block_1,
            self.dec_block_2
        )

        self.decoder_mu = nn.Sequential(
            resblock_dec(ch * 2, ch, up_sample=(1, 2)))  # 1,64 -> 1, 128
        self.decoder_var = nn.Sequential(
            resblock_dec(ch * 2, ch, up_sample=(1, 2)),  # 1,64 -> 1, 128
            nn.Softplus())

        self._initialize_weights()

    def forward(self, z):
        xx = self.fc(z)
        decoded = self.blocks(xx.view(-1, self.ch * 2 ** 3, 1, int(self.size / 2 ** 3)))
        mu = self.decoder_mu(decoded)
        var = self.decoder_var(decoded)
        return mu, var

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)



class VAE(nn.Module):
    def __init__(self, z_dim, ch, size, device):
        super(VAE, self).__init__()
        self.enc = Encoder(z_dim, ch, size, device)
        self.dec = Decoder(z_dim, ch, size)

    def forward(self, x):
        z1, mu1, var1 = self.enc(x)  # Encode
        x_mu1, x_var1 = self.dec(z1)  # Decode
        return x_mu1, x_var1

    def loss(self,x):
        z1, mu1, var1 = self.enc(x)
        x_mu1, x_var1 = self.dec(z1)

        KL1 = gauss_unitgauss_kl(mu1, var1)
        rec_loss_xx = rec_loss_norm4D(x, x_mu1, x_var1) * 4

        lower_bound = [-KL1, -rec_loss_xx]
        return -sum(lower_bound) ,KL1,  rec_loss_xx

    def waveshow(self, true, rec_x_x, i_check=[0, 1, 2, 3]):
        n_fig = len(i_check)
        fig, ax = plt.subplots(n_fig, 1, figsize=(8, 8),
                               sharex="col", sharey="col")
        fig.subplots_adjust(left=0.1, bottom=0.07, top=0.9, right=0.97, wspace=0.25, hspace=0.2)
        first = True
        for i in i_check:
            if true.shape[2] == 2:
                ax[i].plot(np.sqrt(true[i, 0, 0, :] ** 2 + true[i, 1, 0, :] ** 2), label='True', color='r')
                ax[i].plot(np.sqrt(rec_x_x[i, 0, 0, :] ** 2 + rec_x_x[i, 1, 0, :] ** 2), label='rec_x_x')
            elif true.shape[2] == 1:
                ax[i].plot(true[i, 0, 0, :], label='True', color='r')
                ax[i].plot(rec_x_x[i, 0, 0, :], label='rec_x_x')

            ax[i].set_ylabel('Amp.')
            if first:
                ax[i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=False, ncol=2)
            first = False
        # ax[0][0].set_ylim(bottom=0)
        # ax[0][1].set_ylim(bottom=0)
        # ax[i][0].set_xticks([0, 1])
        # ax[i][1].set_xticks([0, 1])
        ax[len(i_check)-1].set_xlabel(r'$k$')
        plt.show()