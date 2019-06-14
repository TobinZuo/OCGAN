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

    def __init__(self, isize, nc, ndf):
        super(Encoder, self).__init__()
        # assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 5, 2, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(ndf),
                        nn.BatchNorm2d(ndf))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = math.floor((isize-5) / 2+1), ndf

        # # Extra layers
        # for t in range(n_extra_layers):
        #     main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
        #                     nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
        #     main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
        #                     nn.BatchNorm2d(cndf))
        #     main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
        #                     nn.LeakyReLU(0.2, inplace=True))

        while csize > 5:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 5, 2, 0, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = math.floor((csize-5) / 2+1)

        main.add_module('final-{0}-{1}-conv'.format(out_feat, out_feat*2),
                        nn.Conv2d(out_feat, out_feat*2, 5, 2, 0, bias=False))
        main.add_module('final-{0}-tanh'.format(out_feat*2),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

##


class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    nc:图片channel
    ngf：图片卷积首层核数目
    """
    def __init__(self, isize, nc, ngf):
        super(Decoder, self).__init__()
        # assert isize % 16 == 0, "isize has to be a multiple of 16"



        main = nn.Sequential()
        main.add_module('pyramid-{0}-{1}-convt'.format(256, 128),
                        nn.ConvTranspose2d(256, 128, 5, 2, 0, bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(128),
                        nn.BatchNorm2d(128))
        main.add_module('pyramid-{0}-relu'.format(128),
                        nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid-{0}-{1}-convt'.format(128, 64),
                        nn.ConvTranspose2d(128, 64, 5, 2, 0,1, bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(64),
                        nn.BatchNorm2d(64))
        main.add_module('pyramid-{0}-relu'.format(64),
                        nn.LeakyReLU(0.2, inplace=True))
        # # Extra layers
        # for t in range(n_extra_layers):
        #     main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
        #                     nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
        #     main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
        #                     nn.BatchNorm2d(cngf))
        #     main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
        #                     nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(64, nc),
                        nn.ConvTranspose2d(64, nc, 5, 2, 0,1, bias=False))
        main.add_module('final-{0}-batchnorm'.format(nc),
                        nn.BatchNorm2d(nc))
        main.add_module('final-{0}-relu'.format(nc),
                        nn.LeakyReLU(0.2, inplace=True))
        # main.add_module('final-{0}-tanh'.format(nc),
        #                 nn.Tanh())
        self.main = main

    def forward(self, input):
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
        model = Encoder(opt.isize, opt.nc, opt.nvf)
        layers = list(model.main.children())

        self.classifier = nn.Sequential(*layers[:])
        self.linear = nn.Sequential()
        self.linear.add_module('linear', nn.Linear(48, 1))
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
        model = Encoder(opt.isize, opt.nc, opt.ndf)
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

