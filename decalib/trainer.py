# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import csv
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from .utils.renderer import SRenderY
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .datasets import datasets
from .utils.config import cfg
torch.backends.cudnn.benchmark = True
from .utils import lossfunc
from .datasets import build_datasets

class Trainer(object):
    def __init__(self, model, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.K = self.cfg.dataset.K
        # training stage: coarse and detail
        self.train_detail = self.cfg.train.train_detail

        # deca model
        self.deca = model.to(self.device)
        self.configure_optimizers()
        self.load_checkpoint()

        # initialize loss  
        # # initialize loss   
        if self.train_detail:     
            self.mrf_loss = lossfunc.IDMRFLoss()
            self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')
        else:
            self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      

        self.segfaceMaskGenerator = None
        if self._use_segface_masks():
            self.segfaceMaskGenerator = self._build_segface_mask_generator()
        
        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        self.result_dir = os.path.join(self.cfg.output_dir, self.cfg.train.result_dir_name)
        os.makedirs(self.result_dir, exist_ok=True)
        self.loss_csv_path = os.path.join(self.result_dir, 'loss_history.csv')
        self.loss_history = self.load_loss_history()
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

    def _use_segface_masks(self):
        return bool(self.cfg.loss.useSeg and getattr(self.cfg.loss, 'useSegface', False))

    def _build_segface_mask_generator(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from tools.segface_mask import SegFaceMaskGenerator

        model_path = getattr(self.cfg.loss, 'segfaceModelPath', '')
        if not model_path:
            raise ValueError('cfg.loss.segfaceModelPath must be set when cfg.loss.useSegface is True')
        logger.info(f'using online SegFace masks from {model_path}')
        return SegFaceMaskGenerator(model_path, self.device)

    def load_loss_history(self):
        if not os.path.exists(self.loss_csv_path):
            return []

        history = []
        with open(self.loss_csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {}
                for key, value in row.items():
                    if value == '':
                        continue
                    clean_row[key] = int(float(value)) if key == 'step' else float(value)
                if clean_row.get('step', -1) <= self.global_step:
                    history.append(clean_row)
        return history

    def save_loss_plots(self):
        if not self.loss_history:
            return

        keys = [key for key in self.loss_history[-1].keys() if key != 'step']
        all_keys = sorted({key for row in self.loss_history for key in row.keys() if key != 'step'})

        with open(self.loss_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step'] + all_keys)
            writer.writeheader()
            writer.writerows(self.loss_history)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f'using OpenCV fallback for loss plots: {exc}')
            self.save_loss_plots_cv2(keys)
            return

        steps = [row['step'] for row in self.loss_history]
        plt.figure(figsize=(12, 7))
        for key in keys:
            values = [row.get(key, np.nan) for row in self.loss_history]
            plt.plot(steps, values, label=key)
        plt.xlabel('global step')
        plt.ylabel('loss')
        plt.title('Training losses')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'losses_over_time.png'), dpi=150)
        plt.close()

        loss_plot_dir = os.path.join(self.result_dir, 'losses')
        os.makedirs(loss_plot_dir, exist_ok=True)
        for key in keys:
            values = [row.get(key, np.nan) for row in self.loss_history]
            plt.figure(figsize=(10, 5))
            plt.plot(steps, values)
            plt.xlabel('global step')
            plt.ylabel(key)
            plt.title(f'{key} over time')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            safe_key = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in key)
            plt.savefig(os.path.join(loss_plot_dir, f'{safe_key}.png'), dpi=150)
            plt.close()

    def save_loss_plots_cv2(self, keys):
        steps = np.asarray([row['step'] for row in self.loss_history], dtype=np.float32)
        if steps.size == 0:
            return

        def safe_name(name):
            return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)

        def draw_plot(plot_keys, ylabel, path):
            width, height = 1100, 650
            left, right, top, bottom = 80, 30, 50, 75
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
            x0, x1 = left, width - right
            y0, y1 = height - bottom, top
            cv2.rectangle(canvas, (x0, y1), (x1, y0), (210, 210, 210), 1)

            x_min = float(np.min(steps))
            x_max = float(np.max(steps))
            if x_max == x_min:
                x_max = x_min + 1.0

            all_values = []
            for key in plot_keys:
                values = np.asarray([row.get(key, np.nan) for row in self.loss_history], dtype=np.float32)
                values = values[np.isfinite(values)]
                if values.size:
                    all_values.append(values)
            if not all_values:
                return

            all_values = np.concatenate(all_values)
            y_min = float(np.min(all_values))
            y_max = float(np.max(all_values))
            if y_max == y_min:
                y_max = y_min + 1.0
            y_pad = (y_max - y_min) * 0.08
            y_min -= y_pad
            y_max += y_pad

            colors = [
                (31, 119, 180), (255, 127, 14), (44, 160, 44),
                (214, 39, 40), (148, 103, 189), (140, 86, 75),
                (227, 119, 194), (127, 127, 127), (188, 189, 34),
                (23, 190, 207),
            ]

            for index, key in enumerate(plot_keys):
                values = np.asarray([row.get(key, np.nan) for row in self.loss_history], dtype=np.float32)
                valid = np.isfinite(values)
                if np.count_nonzero(valid) < 1:
                    continue
                xs = x0 + (steps[valid] - x_min) / (x_max - x_min) * (x1 - x0)
                ys = y0 - (values[valid] - y_min) / (y_max - y_min) * (y0 - y1)
                points = np.stack([xs, ys], axis=1).astype(np.int32)
                color = colors[index % len(colors)]
                if len(points) == 1:
                    cv2.circle(canvas, tuple(points[0]), 3, color, -1)
                else:
                    cv2.polylines(canvas, [points], False, color, 2)
                cv2.putText(canvas, key, (x0 + 10, y1 + 25 + index * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            cv2.putText(canvas, 'global step', (width // 2 - 60, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(canvas, ylabel, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'{x_min:.0f}', (x0 - 10, y0 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'{x_max:.0f}', (x1 - 55, y0 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'{y_min:.4f}', (5, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'{y_max:.4f}', (5, y1 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.imwrite(path, canvas)

        draw_plot(keys, 'loss', os.path.join(self.result_dir, 'losses_over_time.png'))

        loss_plot_dir = os.path.join(self.result_dir, 'losses')
        os.makedirs(loss_plot_dir, exist_ok=True)
        for key in keys:
            draw_plot([key], key, os.path.join(loss_plot_dir, f'{safe_name(key)}.png'))
    
    def configure_optimizers(self):
        if self.train_detail:
            self.opt = torch.optim.Adam(
                                list(self.deca.E_detail.parameters()) + \
                                list(self.deca.D_detail.parameters())  ,
                                lr=self.cfg.train.lr,
                                amsgrad=False)
        else:
            self.opt = torch.optim.Adam(
                                    self.deca.E_flame.parameters(),
                                    lr=self.cfg.train.lr,
                                    amsgrad=False)
    def load_checkpoint(self):
        model_dict = self.deca.model_dict()
        # resume training, including model weight, opt, steps
        # import ipdb; ipdb.set_trace()
        if self.cfg.train.resume and os.path.isfile(os.path.join(self.cfg.output_dir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        # load model weights only
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_flame'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0

    def training_step(self, batch, batch_nb, training_type='coarse'):
        self.deca.train()
        if self.train_detail:
            self.deca.E_flame.eval()
        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
        lmk = batch['landmark'].to(self.device); lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        masks = batch['mask'].to(self.device); masks = masks.view(-1, images.shape[-2], images.shape[-1]) 
        if self.segfaceMaskGenerator is not None:
            images_bhwc = images.permute(0, 2, 3, 1).contiguous()
            masks = self.segfaceMaskGenerator.predict(images_bhwc)[..., 0]

        #-- encoder
        codedict = self.deca.encode(images, use_detail=self.train_detail)
        
        ### shape constraints for coarse model
        ### detail consistency for detail model
        # import ipdb; ipdb.set_trace()
        if self.cfg.loss.shape_consistency or self.cfg.loss.detail_consistency:
            '''
            make sure s0, s1 is something to make shape close
            the difference from ||so - s1|| is 
            the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
            '''
            new_order = np.array([np.random.permutation(self.K) + i*self.K for i in range(self.batch_size)])
            new_order = new_order.flatten()
            shapecode = codedict['shape']
            if self.train_detail:
                detailcode = codedict['detail']
                detailcode_new = detailcode[new_order]
                codedict['detail'] = torch.cat([detailcode, detailcode_new], dim=0)
                codedict['shape'] = torch.cat([shapecode, shapecode], dim=0)
            else:
                shapecode_new = shapecode[new_order]
                codedict['shape'] = torch.cat([shapecode, shapecode_new], dim=0)
            for key in ['tex', 'exp', 'pose', 'cam', 'light', 'images']:
                code = codedict[key]
                codedict[key] = torch.cat([code, code], dim=0)
            ## append gt
            images = torch.cat([images, images], dim=0)# images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
            lmk = torch.cat([lmk, lmk], dim=0) #lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
            masks = torch.cat([masks, masks], dim=0)

        batch_size = images.shape[0]

        ###--------------- training coarse model
        if not self.train_detail:
            #-- decoder
            rendering = True if self.cfg.loss.photo>0 else False
            opdict = self.deca.decode(codedict, rendering = rendering, vis_lmk=False, return_vis=False, use_detail=False)
            opdict['images'] = images
            opdict['lmk'] = lmk

            if self.cfg.loss.photo > 0.:
                #------ rendering
                # mask
                mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), opdict['grid'].detach(), align_corners=False) 
                # images
                predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
                opdict['predicted_images'] = predicted_images

            #### ----------------------- Losses
            losses = {}
            
            ############################# base shape
            predicted_landmarks = opdict['landmarks2d']
            if self.cfg.loss.useWlmk:
                losses['landmark'] = lossfunc.weighted_landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
            else:    
                losses['landmark'] = lossfunc.landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
            if self.cfg.loss.eyed > 0.:
                losses['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks, lmk)*self.cfg.loss.eyed
            if self.cfg.loss.lipd > 0.:
                losses['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks, lmk)*self.cfg.loss.lipd
            
            if self.cfg.loss.photo > 0.:
                if self.cfg.loss.useSeg:
                    masks = masks[:,None,:,:]
                else:
                    masks = mask_face_eye*opdict['alpha_images']
                losses['photometric_texture'] = (masks*(predicted_images - images).abs()).mean()*self.cfg.loss.photo
                opdict['photometric_masks'] = masks.detach()

            if self.cfg.loss.id > 0.:
                shading_images = self.deca.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
                albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
                overlay = albedo_images*shading_images*mask_face_eye + images*(1-mask_face_eye)
                losses['identity'] = self.id_loss(overlay, images) * self.cfg.loss.id
            
            losses['shape_reg'] = (torch.sum(codedict['shape']**2)/2)*self.cfg.loss.reg_shape
            losses['expression_reg'] = (torch.sum(codedict['exp']**2)/2)*self.cfg.loss.reg_exp
            losses['tex_reg'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.loss.reg_tex
            losses['light_reg'] = ((torch.mean(codedict['light'], dim=2)[:,:,None] - codedict['light'])**2).mean()*self.cfg.loss.reg_light
            if self.cfg.model.jaw_type == 'euler':
                # import ipdb; ipdb.set_trace()
                # reg on jaw pose
                losses['reg_jawpose_roll'] = (torch.sum(codedict['euler_jaw_pose'][:,-1]**2)/2)*100.
                losses['reg_jawpose_close'] = (torch.sum(F.relu(-codedict['euler_jaw_pose'][:,0])**2)/2)*10.
        
        ###--------------- training detail model
        else:
            #-- decoder
            shapecode = codedict['shape']
            expcode = codedict['exp']
            posecode = codedict['pose']
            texcode = codedict['tex']
            lightcode = codedict['light']
            detailcode = codedict['detail']
            cam = codedict['cam']

            # FLAME - world space
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode)
            landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:] #; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
            # world to camera
            trans_verts = util.batch_orth_proj(verts, cam)
            predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]
            # camera to image space
            trans_verts[:,:,1:] = -trans_verts[:,:,1:]
            predicted_landmarks[:,:,1:] = - predicted_landmarks[:,:,1:]
            
            albedo = self.deca.flametex(texcode)

            #------ rendering
            ops = self.deca.render(verts, trans_verts, albedo, lightcode) 
            # mask
            mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), ops['grid'].detach(), align_corners=False)
            # images
            predicted_images = ops['images']*mask_face_eye*ops['alpha_images']

            masks = masks[:,None,:,:]
            photometric_masks = masks.detach()

            uv_z = self.deca.D_detail(torch.cat([posecode[:,3:], expcode, detailcode], dim=1))
            # render detail
            uv_detail_normals = self.deca.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
            uv_texture = albedo.detach()*uv_shading
            predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

            #--- extract texture
            uv_pverts = self.deca.render.world2uv(trans_verts).detach()
            uv_gt = F.grid_sample(torch.cat([images, masks], dim=1), uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear', align_corners=False)
            uv_texture_gt = uv_gt[:,:3,:,:].detach(); uv_mask_gt = uv_gt[:,3:,:,:].detach()
            # self-occlusion
            normals = util.vertex_normals(trans_verts, self.deca.render.faces.expand(batch_size, -1, -1))
            uv_pnorm = self.deca.render.world2uv(normals)
            uv_mask = (uv_pnorm[:,[-1],:,:] < -0.05).float().detach()
            ## combine masks
            uv_vis_mask = uv_mask_gt*uv_mask*self.deca.uv_face_eye_mask
            
            #### ----------------------- Losses
            losses = {}
            ############################### details
            # if self.cfg.loss.old_mrf: 
            #     if self.cfg.loss.old_mrf_face_mask:
            #         masks = masks*mask_face_eye*ops['alpha_images']
            #     losses['photo_detail'] = (masks*(predicted_detailed_image - images).abs()).mean()*100
            #     losses['photo_detail_mrf'] = self.mrf_loss(masks*predicted_detailed_image, masks*images)*0.1
            # else:
            pi = 0
            new_size = 256
            uv_texture_patch = F.interpolate(uv_texture[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            uv_texture_gt_patch = F.interpolate(uv_texture_gt[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            uv_vis_mask_patch = F.interpolate(uv_vis_mask[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            
            losses['photo_detail'] = (uv_texture_patch*uv_vis_mask_patch - uv_texture_gt_patch*uv_vis_mask_patch).abs().mean()*self.cfg.loss.photo_D
            losses['photo_detail_mrf'] = self.mrf_loss(uv_texture_patch*uv_vis_mask_patch, uv_texture_gt_patch*uv_vis_mask_patch)*self.cfg.loss.photo_D*self.cfg.loss.mrf

            losses['z_reg'] = torch.mean(uv_z.abs())*self.cfg.loss.reg_z
            losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading)*self.cfg.loss.reg_diff
            if self.cfg.loss.reg_sym > 0.:
                nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
                losses['z_sym'] = (nonvis_mask*(uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum()*self.cfg.loss.reg_sym
            opdict = {
                'verts': verts,
                'trans_verts': trans_verts,
                'landmarks2d': landmarks2d,
                'predicted_images': predicted_images,
                'predicted_detail_images': predicted_detail_images,
                'images': images,
                'lmk': lmk,
                'photometric_masks': photometric_masks
            }
            
        #########################################################
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict
        
    def validation_step(self):
        self.deca.eval()
        try:
            batch = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_dataloader)
            batch = next(self.val_iter)
        images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
        with torch.no_grad():
            codedict = self.deca.encode(images)
            opdict, visdict = self.deca.decode(codedict)
        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:08}.jpg')
        grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
        self.writer.add_image('val_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)
        self.deca.train()

    def evaluate(self):
        ''' NOW validation 
        '''
        os.makedirs(os.path.join(self.cfg.output_dir, 'NOW_validation'), exist_ok=True)
        savefolder = os.path.join(self.cfg.output_dir, 'NOW_validation', f'step_{self.global_step:08}') 
        os.makedirs(savefolder, exist_ok=True)
        self.deca.eval()
        # run now validation images
        from .datasets.now import NoWDataset
        dataset = NoWDataset(scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max)/2)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)
        faces = self.deca.flame.faces_tensor.cpu().numpy()
        for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
            images = batch['image'].to(self.device)
            imagename = batch['imagename']
            with torch.no_grad():
                codedict = self.deca.encode(images)
                _, visdict = self.deca.decode(codedict)
                codedict['exp'][:] = 0.
                codedict['pose'][:] = 0.
                opdict, _ = self.deca.decode(codedict)
            #-- save results for evaluation
            verts = opdict['verts'].cpu().numpy()
            landmark_51 = opdict['landmarks3d_world'][:, 17:]
            landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
            landmark_7 = landmark_7.cpu().numpy()
            for k in range(images.shape[0]):
                os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
                # save mesh
                util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
                # save 7 landmarks for alignment
                np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
                for vis_name in visdict.keys(): #['inputs', 'landmarks2d', 'shape_images']:
                    if vis_name not in visdict.keys():
                        continue
                    # import ipdb; ipdb.set_trace()
                    image = util.tensor2image(visdict[vis_name][k])
                    name = imagename[k].split('/')[-1]
                    # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
                    cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'), image)
            # visualize results to check
            util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))

        ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
        self.deca.train()

    def prepare_data(self):
        self.train_dataset = build_datasets.build_train(self.cfg.dataset)
        self.val_dataset = build_datasets.build_val(self.cfg.dataset)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = None
        self.val_iter = None
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=False)
            self.val_iter = iter(self.val_dataloader)

    def fit(self):
        self.prepare_data()

        full_iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        max_iterations_per_epoch = int(self.cfg.dataset.max_iterations_per_epoch)
        if max_iterations_per_epoch < 0:
            iters_every_epoch = full_iters_every_epoch
        else:
            iters_every_epoch = min(max_iterations_per_epoch, full_iters_every_epoch)
        if iters_every_epoch < 1:
            raise ValueError(
                f'iters_every_epoch={iters_every_epoch}. Check dataset size, '
                f'batch_size={self.batch_size}, and max_iterations_per_epoch={max_iterations_per_epoch}.'
            )
        start_epoch = self.global_step//iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # A new iterator reshuffles the DataLoader each epoch; limiting the
            # number of steps therefore samples a random subset of data lines.
            self.train_iter = iter(self.train_dataloader)
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, opdict = self.training_step(batch, step)
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_values = {'step': self.global_step}
                    for k, v in losses.items():
                        if torch.is_tensor(v):
                            loss_values[k] = float(v.detach().cpu())
                        else:
                            loss_values[k] = float(v)
                    self.loss_history.append(loss_values)
                    self.save_loss_plots()

                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)                    
                    logger.info(loss_info)

                if self.global_step % self.cfg.train.vis_steps == 0:
                    visind = list(range(8))
                    shape_images = self.deca.render.render_shape(opdict['verts'][visind], opdict['trans_verts'][visind])
                    visdict = {
                        'inputs': opdict['images'][visind], 
                        'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                        'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                        'shape_images': shape_images,
                    }
                    if 'predicted_images' in opdict.keys():
                        visdict['predicted_images'] = opdict['predicted_images'][visind]
                    if 'predicted_detail_images' in opdict.keys():
                        visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]
                    if 'photometric_masks' in opdict.keys():
                        mask_alpha = 0.25
                        mask_vis = opdict['photometric_masks'][visind].clamp(0, 1)
                        reference_images = opdict['images'][visind]
                        red_overlay = torch.zeros_like(reference_images)
                        red_overlay[:, 0:1, :, :] = 1.
                        visdict['photometric_masks'] = (
                            reference_images*(1. - mask_alpha*mask_vis) + red_overlay*(mask_alpha*mask_vis)
                        ).clamp(0, 1)

                    savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                    grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
                    # import ipdb; ipdb.set_trace()                    
                    self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.deca.model_dict()
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))   
                    # 
                    if self.global_step % self.cfg.train.checkpoint_steps*10 == 0:
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))   

                if self.val_dataloader is not None and self.global_step % self.cfg.train.val_steps == 0:
                    self.validation_step()
                
                if self.cfg.train.eval_steps > 0 and self.global_step > 0 and self.global_step % self.cfg.train.eval_steps == 0:
                    self.evaluate()

                all_loss = losses['all_loss']
                self.opt.zero_grad(); all_loss.backward(); self.opt.step()
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break
