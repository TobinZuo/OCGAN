""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel
import math
##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    nc:图片channel
    ndf：图片卷积首层核数目
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
            main.add_module('final-{0}-tanh'.format(nz),
                            nn.Tanh())

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##


class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    nc:图片channel
    ngf：图片卷积首层核数目
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


##

# class Generator(nn.Module):
#     def __init__(self,opt):
#         super(Generator,self).__init__()
#         self.encoder=Encoder(opt.isize, opt.nc, opt.ndf)
#         self.decoder=Decoder(opt.isize, opt.nc, opt.ngf)
#     def forward(self, input):
#         v=self.encoder(input)
#         img=self.decodern(v)
#         return img

class Dv(nn.Module):
    """
    DISCRIMINATOR vision  NETWORK
    """

    def __init__(self, opt):
        super(Dv, self).__init__()
        model = Encoder(opt.isize, opt.nz,opt.nc, opt.nvf,opt.ngpu)
        layers = list(model.main.children())

        self.classifier = nn.Sequential(*layers[:])
        self.linear = nn.Sequential()
        self.linear.add_module('linear', nn.Linear(opt.nz, 1))
        self.linear.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.squeeze(1)

        return x

class Dl(nn.Module):
    """
    DISCRIMINATOR latent NETWORK
    """

    def __init__(self,nz):
        super(Dl, self).__init__()
        main = nn.Sequential()
        cn=nz
        while cn>16:
            main.add_module('fc-{0}-{1}'.format(cn,cn//2),nn.Linear(cn,cn//2))
            main.add_module('relu-{0}'.format(cn // 2),
                            nn.LeakyReLU(0.2, inplace=True))
            cn=cn//2
        main.add_module('fc-{0}-{1}'.format(cn, 1), nn.Linear(cn, 1))
        main.add_module('sigmoid',nn.Sigmoid())
        self.main=main

    def forward(self, x):
        x=x.view(x.size(0),-1)
        x = self.main(x)
        x = x.squeeze(1)

        return x

class Classfier(nn.Module):
    """
    Classfier NETWORK
    """

    def __init__(self, opt):
        super(Classfier, self).__init__()
        model = Encoder(opt.isize, opt.nz,opt.nc, opt.ndf,opt.ngpu)
        layers = list(model.main.children())

        self.classifier = nn.Sequential(*layers[:])
        self.linear=nn.Sequential()
        self.linear.add_module('linear', nn.Linear(opt.nz,1))
        self.linear.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x= self.linear(x)
        x = x.squeeze(1)

        return x

