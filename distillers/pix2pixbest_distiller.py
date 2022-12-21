import ntpath
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import gather, parallel_apply, replicate
from tqdm import tqdm

from metric import get_fid, get_cityscapes_mIoU
from utils import util
from utils.vgg_feature import VGGFeature
from .base_pix2pixbest_distiller import BasePix2PixBestDistiller
from models.modules import pytorch_ssim

from relation_kd import RKDLoss


class Pix2PixBestDistiller(BasePix2PixBestDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(Pix2PixBestDistiller, Pix2PixBestDistiller).modify_commandline_options(parser, is_train)
        parser.add_argument('--AGD_weights', type=str, default='1e1, 1e4, 1e1, 1e-5', help='weights for losses in AGD mode')
        parser.add_argument('--n_dis', type=int, default=1, help='iter time for student before update teacher')
        parser.set_defaults(norm='instance', dataset_mode='aligned')

        return parser

    def __init__(self, opt):
        assert opt.isTrain
        super(Pix2PixBestDistiller, self).__init__(opt)
        self.best_fid_teacher, self.best_fid_student = 1e9,  1e9
        self.best_mIoU_teacher, self.best_mIoU_student = -1e9, -1e9
        self.fids_teacher, self.fids_student, self.mIoUs_teacher, self.mIoUs_student = [], [], [], []
        self.npz = np.load(opt.real_stat_path)
        # weights for AGD mood
        loss_weight = [float(char) for char in opt.AGD_weights.split(',')]
        self.lambda_SSIM = loss_weight[0]
        self.lambda_style = loss_weight[1]
        self.lambda_feature = loss_weight[2]
        self.lambda_tv = loss_weight[3]
        self.vgg = VGGFeature().to(self.device)
        self.rkd = RKDLoss()

    def forward(self):
        self.Tfake_B = self.netG_teacher(self.real_A)
        self.Sfake_B = self.netG_student(self.real_A)

    def backward_G_teacher(self):
        fake_AB = torch.cat((self.real_A, self.Tfake_B), 1)
        pred_fake = self.netD_teacher(fake_AB)
        self.loss_G_gan = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        # Second, G(A) = B
        self.loss_G_recon = self.criterionRecon(self.Tfake_B, self.real_B) * self.opt.lambda_recon
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_gan + self.loss_G_recon

        self.loss_G.backward()


    def backward_G_student(self):
        self.loss_G_student = 0
        teacher_image = self.Tfake_B.detach()

        Tfeatures = self.vgg(teacher_image)
        Sfeatures = self.vgg(self.Sfake_B)
        Tgram = [self.gram(fmap) for fmap in Tfeatures]
        Sgram = [self.gram(fmap) for fmap in Sfeatures]
        self.loss_G_style = 0
        for i in range(len(Tgram)):
            self.loss_G_style += self.lambda_style * F.l1_loss(Sgram[i], Tgram[i])
        Srecon, Trecon = Sfeatures[1], Tfeatures[1]
        self.loss_G_feature = self.lambda_feature * F.l1_loss(Srecon, Trecon)
        self.loss_G_student += 2 * self.loss_G_style + 2 * self.loss_G_feature

        batch_size = self.Sfake_B.shape[0]
        self.loss_G_row = 0
        for i in range(batch_size):
            self.loss_G_row += self.rkd(self.Sfake_B[i].transpose(0, 1), teacher_image[i].transpose(0, 1))
        self.loss_G_student += 2.5 * self.loss_G_row
        self.loss_G_column = 0
        for j in range(batch_size):
            self.loss_G_column += self.rkd(self.Sfake_B[j].transpose(0, 2), teacher_image[j].transpose(0, 2))
        self.loss_G_student += 2.5 * self.loss_G_column
        self.loss_G_patch = 0
        s = F.unfold(self.Sfake_B, kernel_size=32, stride=32).permute(0, 2, 1).reshape(-1, 3, 32, 32)
        t = F.unfold(teacher_image, kernel_size=32, stride=32).permute(0, 2, 1).reshape(-1, 3, 32, 32)
        for k in range(batch_size):
            self.loss_G_patch += self.rkd(s[k], t[k])
        self.loss_G_student += 2.5 * self.loss_G_patch

        self.loss_G_student.backward()

    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def optimize_parameters(self, steps):
        self.optimizer_D_teacher.zero_grad()
        self.optimizer_G_teacher.zero_grad()
        self.optimizer_G_student.zero_grad()
        self.forward()
        if steps % self.opt.n_dis == 0:
            util.set_requires_grad(self.netD_teacher, True)
            self.backward_D_teacher()
            util.set_requires_grad(self.netD_teacher, False)
            self.backward_G_teacher()
            self.optimizer_D_teacher.step()
            self.optimizer_G_teacher.step()
        self.backward_G_student()
        self.optimizer_G_student.step()

    def load_networks(self, verbose=True):
        super(Pix2PixBestDistiller, self).load_networks()

    def evaluate_model(self, step):
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_student.eval()
        self.netG_teacher.eval()
        S_fakes, T_fakes, names = [], [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
            self.set_input(data_i)
            self.test()
            S_fakes.append(self.Sfake_B.cpu())
            T_fakes.append(self.Tfake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                # if cnt < 10 * len(self.Tfake_B):
                #     Tfake_im = util.tensor2im(self.Tfake_B[j])
                #     input_im = util.tensor2im(self.real_A[j])
                #     Sfake_im = util.tensor2im(self.Sfake_B[j])
                #     util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png') % name, create_dir=True)
                #     util.save_image(Sfake_im, os.path.join(save_dir, 'Sfake', '%s.png' % name), create_dir=True)
                #     util.save_image(Tfake_im, os.path.join(save_dir, f'Tfake_', '%s.png' %name), create_dir=True)
                #     if self.opt.dataset_mode == 'aligned' and k == 0:
                #         real_im = util.tensor2im(self.real_B[j])
                #         util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                cnt += 1
        fid_teacher = get_fid(T_fakes, self.inception_model, self.npz, device=self.device,
                      batch_size=self.opt.eval_batch_size, tqdm_position=2)
        fid_student = get_fid(S_fakes, self.inception_model, self.npz, device=self.device,
                      batch_size=self.opt.eval_batch_size, tqdm_position=2)
        if fid_student < self.best_fid_student:
            self.is_best = True
            self.best_fid_student = fid_student

        ret = {}
        ret[f'metric/fid_teacher'] = fid_teacher
        if fid_teacher < self.best_fid_teacher:
            self.best_fid_teacher = fid_teacher
        ret[f'metric/fid-best_teacher'] = self.best_fid_teacher
        ret['metric/fid_student'] = fid_student
        ret['metric/fid-best_student'] = self.best_fid_student

        self.netG_teacher.train()
        self.netG_student.train()
        return ret
