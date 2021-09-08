"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lib.models.networks import NetD, weights_init, define_G, define_D, get_scheduler, define_G_DFR, define_D_DFR
from lib.visualizer import Visualizer
from lib.loss import l2_loss, ssim_loss
from lib.evaluate import roc, ssim_score
from lib.models.basemodel import BaseModel
import shutil
from lib.DFR.feature import Extractor
from lib.DFR.feat_cae import FeatCAE
from sklearn.decomposition import PCA
from shutil import copyfile
import cv2
import yaml

class Skipganomaly(BaseModel):
    """GANomaly Class
    """
    @property
    def name(self): return 'skip-ganomaly'

    def __init__(self, opt, data=None):
        super(Skipganomaly, self).__init__(opt, data)
        ##

        # -- Misc attributes
        self.add_noise = True
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        self.check_cuda(True)

        # add CNN for DFR ===========================================================
        if self.opt.DFR:
            print('++++++++++++++DFR+++++++++++++++++++++')
            print('device ', self.device)
            cnn_layers = ('relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4')
            self.extractor = Extractor(backbone='vgg19',
                                       cnn_layers=cnn_layers,
                                       is_agg=True,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       featmap_size=(self.opt.isize, self.opt.isize),
                                       device=self.device,
                                       fine_tuned=self.opt.extractor_fine_tuned)
            self.extractor.to(self.device)
            self.set_nc()
        # add CNN for DFR ===========================================================

        ##
        # Create and initialize networks.
        if self.opt.netg == 'CAE':
            # PCA decide latent_dim
            # self.n_dim = None
            # self.netg = self.build_classifier()

            # default latent_dim 200
            self.netg = FeatCAE(in_channels=self.opt.nc, latent_dim=200).to(self.device)
        elif self.opt.netg == 'Unet_DFR':
            self.netg = define_G_DFR(self.opt, norm='batch', use_dropout=False, init_type='normal')
        else:
            self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal')

        if self.opt.netg == 'Unet_DFR':
            self.netd = define_D_DFR(self.opt, norm='batch', use_sigmoid=False, init_type='normal')
        else:
            self.netd = define_D(self.opt, norm='batch', use_sigmoid=False, init_type='normal')

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG_best.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG_best.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD_best.pth'))['state_dict'])
            print("\tDone.\n")

        if self.opt.verbose:
            print(self.netg)
            print(self.netd)

        ##
        # Loss Functions
        self.l_adv = nn.BCELoss()
        if self.opt.l_con == 'l1':
            self.l_con = nn.L1Loss()
        elif self.opt.l_con == 'l2':
            self.l_con = l2_loss
        elif self.opt.l_con == 'ssim':
            self.l_con = ssim_loss
        self.l_lat = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            if self.opt.DFR:
                self.extractor.train()
            self.netg.train()
            self.netd.train()
            self.optimizers = []
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def forward(self):
        self.forward_g()
        self.forward_d()

    def forward_g(self):
        """ Forward propagate through netG
        """
        #self.fake = self.netg(self.input + self.noise)
        # no noise
        if self.opt.DFR:
            self.input = self.extractor(self.input)
        self.fake = self.netg(self.input)

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake)

    def backward_g(self):
        """ Backpropagate netg
        """
        if not self.opt.WGAN:
            self.err_g_adv = self.opt.w_adv * self.l_adv(self.pred_fake, self.real_label)
        else:
            self.err_g_adv = -1*(self.opt.w_adv * self.pred_fake.mean())

        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)
        self.err_g_lat = self.opt.w_lat * self.l_lat(self.feat_fake, self.feat_real)

        if self.opt.no_discriminator:
            self.err_g = self.err_g_con
        else:
            self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        if not self.opt.WGAN:
            # Fake
            pred_fake, _ = self.netd(self.fake.detach())
            self.err_d_fake = self.l_adv(pred_fake, self.fake_label)

            # Real
            # pred_real, feat_real = self.netd(self.input)
            self.err_d_real = self.l_adv(self.pred_real, self.real_label)

            # Combine losses.
            self.err_d = self.err_d_real + self.err_d_fake + self.err_g_lat
            self.err_d.backward(retain_graph=True)
        else:
            # Fake
            self.err_d_fake, _ = self.netd(self.fake.detach())
            self.err_d_fake = self.err_d_fake.mean()

            # Real
            # pred_real, feat_real = self.netd(self.input)
            self.err_d_real = self.pred_real.mean()

            gradient_penalty = self.calculate_gradient_penalty(self.input.data, self.fake.data)

            self.err_d = self.err_d_fake - self.err_d_real + gradient_penalty + self.err_g_lat
            self.err_d.backward(retain_graph=True)


    def update_netg(self):
        """ Update Generator Network.
        """       
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    def update_netd(self):
        """ Update Discriminator Network.
        """       
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        #if self.err_d < 1e-5: self.reinit_d()
    ##
    def optimize_params(self):
        """ Optimize netD and netG  networks.
        """
        self.forward()
        self.update_netg()
        self.update_netd()

    ##
    def test_og(self, plot_hist=False, test_set='test'):
        """ Test GANomaly model.

        Args:
            data ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                self.load_weights(is_best=True)

            self.opt.phase = 'test'

            scores = {}

            # Create big error tensor for the test set.
            # self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            # self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            # self.features  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            print("   Testing %s" % self.name)
            print("   Test set %s" % test_set)
            if test_set == 'val':
                test_data = self.data.valid
            else:
                test_data = self.data.test

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(test_data.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(test_data.dataset),), dtype=torch.long, device=self.device)
            self.features  = torch.zeros(size=(len(test_data.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(test_data, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                # Forward - Pass
                self.set_input(data)
                self.fake = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9*rec + 0.1*lat

                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    #dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    dst = os.path.join(self.opt.outf, self.opt.name, test_set, 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    else:
                        shutil.rmtree(dst)
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    #vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    #vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc(self.gt_labels, self.an_scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            ##
            # PLOT HISTOGRAM
            if plot_hist:
                plt.ion()
                # Create data frame for scores and labels.
                scores['scores'] = self.an_scores
                scores['labels'] = self.gt_labels
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv("histogram.csv")

                # Filter normal and abnormal scores.
                abn_scr = hist.loc[hist.labels == 1]['scores']
                nrm_scr = hist.loc[hist.labels == 0]['scores']

                # Create figure and plot the distribution.
                # fig, ax = plt.subplots(figsize=(4,4));
                sns.distplot(nrm_scr, label=r'Normal Scores')
                sns.distplot(abn_scr, label=r'Abnormal Scores')

                plt.legend()
                plt.yticks([])
                plt.xlabel(r'Anomaly Scores')

            ##
            # PLOT PERFORMANCE
            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            self.visualizer.print_current_performance(performance, performance['AUC'])

            ##
            # RETURN
            return performance

    def test(self, plot_hist=True, test_set='test'):
        """ Test GANomaly model.

        Args:
            data ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                self.load_weights(is_best=True)

            self.opt.phase = 'test'

            scores = {}

            # Create big error tensor for the test set.
            # self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            # self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            # self.features  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            print("   Testing %s" % self.name)
            print("   Test set %s" % test_set)
            if test_set == 'train':
                test_data = self.data.train
            elif test_set == 'val':
                test_data = self.data.valid
            else:
                test_data = self.data.test

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(test_data.dataset),), dtype=torch.float32, device=self.device)
            self.rec_scores = torch.zeros(size=(len(test_data.dataset),), dtype=torch.float32, device=self.device)
            self.lat_scores = torch.zeros(size=(len(test_data.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(test_data.dataset),), dtype=torch.long, device=self.device)
            self.features  = torch.zeros(size=(len(test_data.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            dst = os.path.join(self.opt.outf, self.opt.name, test_set, 'images')
            if not os.path.isdir(dst):
                os.makedirs(dst)
            else:
                shutil.rmtree(dst)
                os.makedirs(dst)
            self.pixel_max = 0
            self.pixel_min = 1
            for i, data in enumerate(test_data, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                # Forward - Pass
                self.set_input(data)
                if self.opt.DFR:
                    self.input = self.extractor(self.input)
                    self.fake = self.netg(self.input)
                else:
                    self.fake = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                if self.opt.l_con == 'l1' or self.opt.l_con == 'l2':
                    rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                elif self.opt.l_con == 'ssim':
                    rec = ssim_score(self.input, self.fake)
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])

                if self.opt.l_con == 'l1':
                    rec = torch.mean(torch.abs(rec), dim=1)
                elif self.opt.l_con == 'l2':
                    rec = torch.mean(torch.pow(rec, 2), dim=1)
                elif self.opt.l_con == 'ssim':
                    pass

                lat = torch.mean(torch.pow(lat, 2), dim=1)

                if self.opt.no_discriminator:
                    error = rec
                else:
                    error = 0.9 * rec + 0.1 * lat

                # +++++++++++++++ heatmap #

                h = torch.mean((self.input - self.fake) ** 2, dim=1)
                temp_pixel_max = h.max().data
                temp_pixel_min = h.min().data
                if temp_pixel_max > self.pixel_max:
                    self.pixel_max = temp_pixel_max
                if temp_pixel_min < self.pixel_min:
                    self.pixel_min = temp_pixel_min

                # ++++++++++++++++++ heatmap #

                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.rec_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = rec.reshape(error.size(0))
                self.lat_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = lat.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    #dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    real, fake, _ = self.get_current_images()
                    #vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    #vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            min = torch.min(self.an_scores)
            max = torch.max(self.an_scores)
            if self.opt.l_con != 'ssim':
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                                 (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc(self.gt_labels, self.an_scores, saveto=os.path.join(self.opt.outf, self.opt.name, test_set))
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc), ('min', min), ('max', max), ('pixel min', self.pixel_min), ('pixel max', self.pixel_max)])

            ##
            # PLOT HISTOGRAM
            if plot_hist:
                save_path = os.path.join(self.opt.outf, self.opt.name, test_set)
                plt.ion()
                # Create data frame for scores and labels.
                scores['min'] = min.cpu().numpy()
                scores['max'] = max.cpu().numpy()
                scores['pixel_min'] = self.pixel_min.cpu().numpy()
                scores['pixel_max'] = self.pixel_max.cpu().numpy()
                scores['rec scores'] = self.rec_scores.cpu()
                scores['lat scores'] = self.lat_scores.cpu()
                scores['scores'] = self.an_scores.cpu()
                scores['labels'] = self.gt_labels.cpu()
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv(os.path.join(save_path, 'histogram.csv'))

                data = dict(min=float(min.cpu().numpy()), max=float(max.cpu().numpy()), pixel_min=float(self.pixel_min.cpu().numpy()), pixel_max=float(self.pixel_max.cpu().numpy()))
                with open(os.path.join(save_path, 'value.yaml'), 'w') as outfile:
                    yaml.dump(data, outfile, default_flow_style=False)

                # Filter normal and abnormal scores.
                #abn_scr = hist.loc[hist.labels == 1]['scores']
                #nrm_scr = hist.loc[hist.labels == 0]['scores']
                #abn_scr = abn_scr.values
                #nrm_scr = nrm_scr.values
                # abn_scr = abn_scr.cpu()
                # nrm_scr = nrm_scr.cpu()
                # Create figure and plot the distribution.
                #fig, ax = plt.subplots(figsize=(4,4))
                #nrm_scr = [1, 2, 3, 4, 5, 6]
                #abn_scr = [1, 2, 3, 4, 5, 6]
                #sns.distplot(nrm_scr, label='Normal Scores')
                #sns.distplot(abn_scr, label='Abnormal Scores')
                #plt.legend()
                #plt.yticks([])
                #plt.xlabel('Anomaly Scores')

                #plt.savefig(os.path.join(save_path, 'Anomaly_scores.png'))
                #plt.savefig('Anomaly_scores.png')

            ##
            # PLOT PERFORMANCE
            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            ##
            # RETURN
            return performance

    def eval(self, plot_hist=False, test_set='test', min=0.0, max=0.0, pixel_min=0, pixel_max=1, threshold=0.0):
        """ Test GANomaly model.

                Args:
                    data ([type]): Dataloader for the test set

                Raises:
                    IOError: Model weights not found.
                """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                self.load_weights(is_best=True)

            self.opt.phase = 'test'

            scores = {}

            # Create big error tensor for the test set.
            # self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            # self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            # self.features  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            print("   Validation %s" % self.name)
            print("   Val set %s" % test_set)
            if test_set == 'train':
                test_data = self.data.train
            elif test_set == 'val':
                test_data = self.data.valid
            else:
                test_data = self.data.test

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(test_data.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(test_data.dataset),), dtype=torch.long, device=self.device)
            self.features = torch.zeros(size=(len(test_data.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            dst = os.path.join(self.opt.outf, self.opt.name, test_set, 'images_all')
            if not os.path.isdir(dst):
                os.makedirs(dst)
            else:
                shutil.rmtree(dst)
                os.makedirs(dst)

            nor_dst = os.path.join(self.opt.outf, self.opt.name, test_set, 'images_nor_error')
            if not os.path.isdir(nor_dst):
                os.makedirs(nor_dst)
            else:
                shutil.rmtree(nor_dst)
                os.makedirs(nor_dst)

            abn_dst = os.path.join(self.opt.outf, self.opt.name, test_set, 'images_abn_error')
            if not os.path.isdir(abn_dst):
                os.makedirs(abn_dst)
            else:
                shutil.rmtree(abn_dst)
                os.makedirs(abn_dst)

            for i, data in enumerate(test_data, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                # Forward - Pass

                self.set_input(data)
                self.og_img = self.input
                if self.opt.DFR:
                    self.input = self.extractor(self.input)
                    self.fake = self.netg(self.input)
                else:
                    self.fake = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                if self.opt.l_con == 'l1' or self.opt.l_con == 'l2':
                    rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                elif self.opt.l_con == 'ssim':
                    rec = ssim_score(self.input, self.fake)
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])

                if self.opt.l_con == 'l1':
                    rec = torch.mean(torch.abs(rec), dim=1)
                elif self.opt.l_con == 'l2':
                    rec = torch.mean(torch.pow(rec, 2), dim=1)
                elif self.opt.l_con == 'ssim':
                    pass
                lat = torch.mean(torch.pow(lat, 2), dim=1)

                if self.opt.no_discriminator:
                    error = rec
                else:
                    error = 0.9 * rec + 0.1 * lat

                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    # dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    #real, fake, _ = self.get_current_images()
                    # vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    # vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

                    # heatmap++++++++++++++++++++++++++++
                    h = torch.mean((self.input - self.fake) ** 2, dim=1)
                    h = self.min_max_norm(h, pixel_min, pixel_max)
                    # heatmap++++++++++++++++++++++++++++
                    for j in range(0, error.size(0)):
                        vutils.save_image(self.og_img[j].data, '%s/%03d_real.png' % (dst, i * self.opt.batchsize+j), normalize=True)
                        if not self.opt.DFR:
                            vutils.save_image(self.fake[j].data, '%s/%03d_fake.png' % (dst, i * self.opt.batchsize+j), normalize=True)

                        heat = self.cvt2heatmap(h[j].cpu().numpy()*255)
                        self.save_heatmap('%s/%03d_heat.png' % (dst, i * self.opt.batchsize+j), heat)
                    # vutils.save_image(real, '%s/real_%03d.png' % (dst, i + 1), normalize=True)
                    # vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i + 1), normalize=True)
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            # self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
            #                  (torch.max(self.an_scores) - torch.min(self.an_scores))
            if self.opt.l_con != 'ssim':
                self.an_scores = (self.an_scores - torch.tensor(min, dtype=torch.float32, device=self.device)) / \
                                 (torch.tensor(max, dtype=torch.float32, device=self.device) - torch.tensor(min, dtype=torch.float32, device=self.device))

            #auc = roc(self.gt_labels, self.an_scores, saveto=os.path.join(self.opt.outf, self.opt.name, test_set))
            #performance = OrderedDict(
            #    [('Avg Run Time (ms/batch)', self.times), ('AUC', auc), ('min', torch.min(self.an_scores)),
            #     ('max', torch.max(self.an_scores))])

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            self.an_scores = self.an_scores.float()
            self.gt_labels = self.gt_labels.float()
            for i in range(0, len(self.an_scores)):
                #print('score ', self.an_scores[i])
                if (self.an_scores[i] >= threshold):
                    predict = 1.0
                else:
                    predict = 0.0
                #print('predict ', predict)
                #print('gt ', self.gt_labels[i])
                if (self.gt_labels[i] == 0) and (predict == self.gt_labels[i]):
                    # tp+=1
                    pass
                elif (self.gt_labels[i] == 1) and (predict == self.gt_labels[i]):
                    tp += 1
                elif (self.gt_labels[i] == 0) and (predict != self.gt_labels[i]):
                    copyfile('%s/%03d_real.png' % (dst, i), '%s/%03d_real.png' % (nor_dst, i))
                    copyfile('%s/%03d_heat.png' % (dst, i), '%s/%03d_heat.png' % (nor_dst, i))
                    fp += 1
                elif (self.gt_labels[i] == 1) and (predict != self.gt_labels[i]):
                    copyfile('%s/%03d_real.png' % (dst, i), '%s/%03d_real.png' % (abn_dst, i))
                    copyfile('%s/%03d_heat.png' % (dst, i), '%s/%03d_heat.png' % (abn_dst, i))
                    # fp+=1
                    pass

                # print('tp ', tp / len(self.gt_labels == 1))
                # print('tn ', tn / len(self.an_scores))
                # print('fp ', fp / len(self.gt_labels == 0))
                # print('fn', fn / len(self.an_scores))

            ##
            # PLOT HISTOGRAM
            if plot_hist:
                save_path = os.path.join(self.opt.outf, self.opt.name, test_set)
                plt.ion()
                # Create data frame for scores and labels.

                scores['min'] = torch.min(self.an_scores).cpu()
                scores['max'] = torch.max(self.an_scores).cpu()
                scores['score - min'] = ((self.an_scores - torch.min(self.an_scores))).cpu()
                scores['max - min'] = (torch.max(self.an_scores) - torch.min(self.an_scores)).cpu()
                scores['og'] = self.an_scores.cpu()
                scores['scores'] = self.an_scores.cpu()
                scores['labels'] = self.gt_labels.cpu()
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv(os.path.join(save_path, 'eval_histogram.csv'))

    def build_classifier(self):
        # self.load_dim(self.model_path)
        if self.n_dim is None:
            print("Estimating one class classifier AE parameter...")
            feats = torch.Tensor()
            for i, normal_img in enumerate(self.data.valid):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img[0].to(self.device)
                feat = self.extractor.feat_vec(normal_img)
                feats = torch.cat([feats, feat.cpu()], dim=0)
            # to numpy
            feats = feats.detach().numpy()
            print('feats shape ', feats.shape)
            # estimate parameters for mlp
            pca = PCA(n_components=0.90)  # 0.9 here try 0.8
            pca.fit(feats)
            n_dim, in_feat = pca.components_.shape
            print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))
            self.n_dim = n_dim
        else:
            for i, normal_img in enumerate(self.data.valid):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
            in_feat = feat.shape[1]

        #print("BN?:", self.cfg.is_bn)
        autoencoder = FeatCAE(in_channels=in_feat, latent_dim=self.n_dim, is_bn=True).to(self.device)

        return autoencoder

    def set_nc(self):
        for i, normal_img in enumerate(self.data.valid):
            i += 1
            if i > 1:
                break
            normal_img = normal_img[0].to(self.device)
            feat = self.extractor.feat_vec(normal_img)
        nc = feat.shape[1]
        self.opt.nc = nc

    def min_max_norm(self, image, pixel_min, pixel_max):
        return (image-pixel_min)/(pixel_max - pixel_min)

    def cvt2heatmap(self, gray):
        heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
        return heatmap

    def save_heatmap(sefl, des, heatmap):
        cv2.imwrite(des, heatmap)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.opt.batchsize, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.opt.batchsize, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated, _ = self.netd(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                      prob_interpolated.size()),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return grad_penalty
