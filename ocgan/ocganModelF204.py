"""ocgan
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from ocgan.ocganNetF204 import Encoder,Decoder, Dl,Dv,Classfier,weights_init
from lib.visualizer import Visualizer
from lib.evaluate import evaluate

##
class ocgan(object):
    """GANomaly Class
    """

    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'ocgan'

    def __init__(self, opt, dataloader=None):
        super(ocgan, self).__init__()
        ##
        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

        # -- Discriminator attributes.
        self.out_dv0=None
        self.out_dv1=None
        self.out_dv=None
        self.label_dv=None
        self.err_dv_bce=None
        self.out_dl0=None
        self.out_dl1=None
        self.out_dl=None
        self.label_dl=None
        self.err_dl_bce=None
        self.err_d=None

        # -- Generator attributes.
        self.out_gv0 = None
        self.out_gv1 = None
        self.out_gv = None
        self.label_gv = None
        self.err_gv_bce = None
        self.out_gl0 = None
        self.out_gl1 = None
        self.out_gl = None
        self.label_gl = None
        self.err_gl_bce = None
        self.err_g_mse =None
        self.err_g = None

        # -- Classfier attribute
        self.out_c0=None
        self.out_c1=None
        self.out_c=None
        self.label_c=None
        self.err_c_bce = None

        # -- mine attribute
        self.out_m = None
        self.err_m_bce = None

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0


        ##
        # Create and initialize networks.
        print('Create and initialize networks.')
        self.neten=Encoder(self.opt.isize,opt.nz,self.opt.nc,self.opt.ndf,opt.ngpu).to(self.device)
        self.netde=Decoder(self.opt.isize,opt.nz,self.opt.nc,self.opt.ngf,opt.ngpu).to(self.device)
        self.netdl=Dl(self.opt.nz).to(self.device)
        self.netdv=Dv(self.opt).to(self.device)
        self.netc=Classfier(self.opt).to(self.device)
        self.netdv.apply(weights_init)
        self.netc.apply(weights_init)
        self.neten.apply(weights_init)
        self.netde.apply(weights_init)
        print('end')
        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'neten.pth'))['epoch']
            self.neten.load_state_dict(torch.load(os.path.join(self.opt.resume, 'neten.pth'))['state_dict'])
            self.netde.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netde.pth'))['state_dict'])
            self.netdl.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netdl.pth'))['state_dict'])
            self.netdv.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netdv.pth'))['state_dict'])
            self.netc.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netc.pth'))['state_dict'])
            print("\tDone.\n")

        print(self.neten)
        print(self.netde)
        print(self.netdl)
        print(self.netdv)
        print(self.netc)
        ##
        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1l_criterion = nn.L1Loss()
        self.l2l_criterion = nn.MSELoss()

        ##
        # Initialize input tensors.
        print('Initialize input tensors.')
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.labelf = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0
        self.u=None
        self.n=None
        self.l1=None
        self.l2=torch.empty(size=(self.opt.batchsize, self.opt.nz,1,1), dtype=torch.float32)
        self.del1=None
        self.del2=None
        print('end')

        ##
        # Setup optimizer
        print('Setup optimizer')
        if self.opt.isTrain:
            self.neten.train()
            self.netde.train()
            self.netdv.train()
            self.netdl.train()
            self.netc.train()
            self.optimizer_en = optim.Adam(self.neten.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_de = optim.Adam(self.netde.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_dl = optim.Adam(self.netdl.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_dv = optim.Adam(self.netdv.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_c = optim.Adam(self.netc.parameters(), lr=self.opt.clr, betas=(self.opt.beta1, 0.999))
            self.optimizer_l2 = optim.Adam([{'params':self.l2}], lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('end')
    ##
    def set_input(self, input):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        self.input.data.resize_(input[0].size()).copy_(input[0])
        self.gt.data.resize_(input[1].size()).copy_(input[1])

        # Copy the first batch as the fixed input.
        if self.total_steps == self.opt.batchsize:
            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])
    ##
    def update_netd(self):
        """
        Update D network: Ladv = |f(real) - f(fake)|_2
        """
        ##
        # Feature Matching.
        self.netdv.zero_grad()
        self.netdl.zero_grad()
        # --
        self.out_dv0 = self.netdv(self.del2.detach())
        self.out_dv1 = self.netdv(self.input)
        self.out_dv = torch.cat([self.out_dv0, self.out_dv1], 0)

        self.labelf.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.label_dv = torch.cat([self.labelf, self.label], 0)

        self.err_dv_bce = self.bce_criterion(self.out_dv, self.label_dv)

        self.out_dl0 = self.netdl(self.l1.detach())
        self.out_dl1 = self.netdl(self.l2)
        self.out_dl = torch.cat([self.out_dl0, self.out_dl1], 0)

        self.labelf.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.label_dl = torch.cat([self.labelf, self.label], 0)

        self.err_dl_bce = self.bce_criterion(self.out_dl, self.label_dl)

        self.err_d=self.err_dl_bce+self.err_dv_bce
        self.err_d.backward(retain_graph=True)
        self.optimizer_dv.step()
        self.optimizer_dl.step()
    ##
    def update_netg(self):
        """
        # ============================================================ #
        # (2) Update G network: log(D(G(x)))  + ||G(x) - x||           #
        # ============================================================ #

        """
        self.neten.zero_grad()
        self.netde.zero_grad()
        # --
        # self.out_gv0 = self.netdv(self.input)
        # self.out_gv1 = self.netdv(self.del2)
        # self.out_gv = torch.cat([self.out_gv0, self.out_gv1], 0)
        self.out_gv1 = self.netdv(self.del2)

        # self.labelf.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        # self.label_gv = torch.cat([self.labelf, self.label], 0)

        # self.err_gv_bce = self.bce_criterion(self.out_gv, self.label_gv)
        self.err_gv_bce = self.bce_criterion(self.out_gv1, self.label)

        # self.out_gl0 = self.netdl(self.l2)
        # self.out_gl1 = self.netdl(self.l1)
        # self.out_gl = torch.cat([self.out_gl0, self.out_gl1], 0)
        self.out_gl1 = self.netdl(self.l1)

        # self.labelf.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        # self.label_gl = torch.cat([self.labelf, self.label], 0)

        # self.err_gl_bce = self.bce_criterion(self.out_gl, self.label_gl)
        self.err_gl_bce = self.bce_criterion(self.out_gl1, self.label)

        self.err_g_mse=self.l2l_criterion(self.input, self.del1)

        self.err_g = self.err_gl_bce + self.err_gv_bce+self.opt.w_rec*self.err_g_mse
        self.err_g.backward(retain_graph=True)
        self.optimizer_en.step()
        self.optimizer_de.step()
    ##
    def update_netc(self):
        self.netc.zero_grad()
        self.out_c0=self.netc(self.del2.detach())
        # self.out_c1 = self.netc(self.input)
        self.out_c1=self.netc(self.del1.detach())
        self.out_c=torch.cat([self.out_c0,self.out_c1],0)

        self.labelf.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.label_c=torch.cat([self.labelf,self.label],0)

        self.err_c_bce = self.bce_criterion(self.out_c, self.label_c)
        self.err_c_bce.backward()
        self.optimizer_c.step()

    def update_l2(self):
        for i in range(5):
            self.optimizer_l2.zero_grad()
            self.out_m=self.netc(self.del2)
            self.labelf.data.resize_(self.opt.batchsize).fill_(self.fake_label)
            self.err_m_bce = self.bce_criterion(self.out_m, self.labelf)
            self.err_m_bce.backward()
            self.optimizer_l2.step()
            self.del2=self.netde(self.l2)

    def optimize(self):
        """ Optimize netD and netG  networks.
        """
        self.u = np.random.uniform(-1, 1, (self.opt.batchsize, self.opt.nz, 1, 1))
        self.l2 = torch.from_numpy(self.u).float()
        self.l2=self.l2.to(self.device)
        self.n = torch.randn(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize, dtype=torch.float, device=self.device)
        # print(type(self.input))
        # print(type(self.n))
        self.l1 = self.neten(self.input + self.n)
        self.del1=self.netde(self.l1)
        self.del2=self.netde(self.l2)
        self.update_netc()
        self.update_netd()
        if self.opt.mine==True:
            self.update_l2()
        self.update_netg()


    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([('err_d', self.err_d.item()),
                              ('err_g', self.err_g.item()),
                              ('err_dv_bce', self.err_dv_bce.item()),
                              ('err_dl_bce', self.err_dl_bce.item()),
                              ('err_gv_bce', self.err_gv_bce.item()),
                              ('err_gl_l1l', self.err_gl_bce.item()),
                              ('err_g_mse', self.err_g_mse.item()),
                              ('err_c_bce', self.err_c_bce.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.del1.data
        fixed = self.netde(self.neten(self.fixed_input))[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.neten.state_dict()},
                   '%s/neten.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netde.state_dict()},
                   '%s/netde.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netdl.state_dict()},
                   '%s/netdl.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netdv.state_dict()},
                   '%s/netdv.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netc.state_dict()},
                   '%s/netc.pth' % (weight_dir))

    ##
    def train_epoch(self):
        """ Train the model for one epoch.
        """
        self.neten.train()
        self.netde.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)

            self.optimize()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name(), self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)
    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name())
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_epoch()
            res = self.test()
            # if res['AUC'] > best_auc:
            #     best_auc = res['AUC']
            #     self.save_weights(self.epoch)
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
            self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name())

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # print(00)
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                # print(11)
                path = "./output/{}/{}/train/weights/netc.pth".format(self.name().lower(), self.opt.dataset)
                # path1 = "./output/{}/{}/train/weights/netc.pth".format(self.name().lower(), self.opt.dataset)
                # path2 = "./output/{}/{}/train/weights/neten.pth".format(self.name().lower(), self.opt.dataset)
                # path3 = "./output/{}/{}/train/weights/netde.pth".format(self.name().lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']
                # pretrained_dict1 = torch.load(path1)['state_dict']
                # pretrained_dict2 = torch.load(path2)['state_dict']
                # pretrained_dict3 = torch.load(path3)['state_dict']

                try:
                    self.netc.load_state_dict(pretrained_dict)
                    # self.netc.load_state_dict(pretrained_dict1)
                    # self.neten.load_state_dict(pretrained_dict2)
                    # self.netde.load_state_dict(pretrained_dict3)

                except IOError:
                    raise IOError("netc weights not found")
                print('   Loaded weights.')
            # print(22)
            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,device=self.device)
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,device=self.device)
            # print("   Testing model %s." % self.name())
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            # print(self.dataloader['test'])
            # print(type(self.dataloader['test']))
            # print(33)
            for i, data in enumerate(self.dataloader['test'], 0):
                #
                # print(data)
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.l1=self.neten(self.input)
                self.del1=self.netde(self.l1)
                self.out_c = self.netc(self.del1)
                # self.out_c = self.netc(self.input)

                time_o = time.time()
                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + self.out_c.size(0)] = self.out_c
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+self.out_c.size(0)] = self.gt.reshape(self.out_c.size(0))
                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True,nrow=4)
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True,nrow=4)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # print(44)
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores,metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            # print(auc)

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance
