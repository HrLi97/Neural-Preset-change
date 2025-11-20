import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import numpy as np
import os
import cv2
from omegaconf import OmegaConf
from utils.setup import get_optimizer, get_scheduler
from datasets.rgb2lab import srgb_tensor_to_normalized_lab

class Solver(pl.LightningModule):
    """Neural Styler Solver using PyTorch Lightning.
    
    This solver implements the training and validation logic for the neural style transfer model.
    It handles both local visualization and wandb logging.
    """
    
    def __init__(self, net, criterion, cfg):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # Initialize components
        self.net = net
        self.criterion = criterion
        self.cfg = cfg
        self.current_stage = cfg.train.stage
        assert self.current_stage in ["pretrain", "full_train"], "train.stage must be 'pretrain' or 'full_train'"
        
        # Initialize loss functions
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

        if self.current_stage == "full_train" and cfg.train.pretrain_ckpt is not None:
            self.load_pretrain_weights(cfg.train.pretrain_ckpt)

    def load_pretrain_weights(self, ckpt_path):
        print(f"Loading pretrain weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get('state_dict', ckpt) 
        # 局部映射还需要加入新的参数
        # 筛选需要加载的参数（编码器+transform_p/transform_q，忽略其他可能新增的参数）
        pretrain_params = {}
        for k, v in state_dict.items():
            if "style_encoder." in k or "transform_p" in k or "transform_q" in k:
                pretrain_params[k] = v
        
        load_result = self.net.load_state_dict(pretrain_params, strict=False)
        print(f"Loaded pretrain params: {list(pretrain_params.keys())}")

    def _process_batch_pretrain(self, batch):
        img_i, img_j = batch["img_i"], batch["img_j"]
        
        # 前向传播（仅获取解耦特征z）
        z_i = self.net(img_i, img_j, phase="pretrain")  
        z_j = self.net(img_j, img_i, phase="pretrain") 
        
        return img_i, img_j, z_i, z_j

    def _process_batch_full_train(self, batch):
        img_i, img_j = batch["img_i"], batch["img_j"]

        stacked_content = torch.cat([img_i, img_j], dim=0)
        stacked_style = torch.cat([img_j, img_i], dim=0)

        stacked_z, stacked_result = self.net(stacked_content, stacked_style, phase="full_train")
        z_i, z_j = torch.split(stacked_z, [img_i.shape[0], img_j.shape[0]], dim=0)
        y_j, y_i = torch.split(stacked_result, [img_i.shape[0], img_j.shape[0]], dim=0)

        return img_i, img_j, z_i, z_j, y_i, y_j

    def _compute_losses_pretrain(self, z_i, z_j):
        consistency_loss = self.l2(z_i, z_j)
        total_loss = self.cfg.criterion.lambda_consistency * consistency_loss
        return {
            'consistency_loss': consistency_loss,
            'total_loss': total_loss
        }
        
    def _compute_losses_full_train(self, z_i, z_j, y_i, y_j, img_i, img_j):
        consistency_loss = self.l2(z_i, z_j)
        reconstruction_loss = self.l1(y_i, img_i) + self.l1(y_j, img_j)
        total_loss = reconstruction_loss + self.cfg.criterion.lambda_consistency * consistency_loss
        
        return {
            'consistency_loss': consistency_loss,
            'reconstruction_loss': reconstruction_loss,
            'total_loss': total_loss
        }
        

    # def _compute_losses_full_train(self, z_i, z_j, y_i, y_j, img_i, img_j):
    #     w_ab = getattr(self.cfg.criterion, 'lab_ab_weight', 2.0)
    #     lam_cons = getattr(self.cfg.criterion, 'lambda_consistency', 10.0)
    #     lam_moment = getattr(self.cfg.criterion, 'lambda_moment', 1.0)
    #     lam_cov = getattr(self.cfg.criterion, 'lambda_cov', 0.0)

    #     y_i_lab = srgb_tensor_to_normalized_lab(torch.clamp(y_i, 0, 1))
    #     y_j_lab = srgb_tensor_to_normalized_lab(torch.clamp(y_j, 0, 1))
    #     img_i_lab = srgb_tensor_to_normalized_lab(img_i)
    #     img_j_lab = srgb_tensor_to_normalized_lab(img_j)

    #     loss_L_i = self.l1(y_i_lab[:, 0], img_i_lab[:, 0])
    #     loss_L_j = self.l1(y_j_lab[:, 0], img_j_lab[:, 0])
    #     loss_ab_i = self.l1(y_i_lab[:, 1:], img_i_lab[:, 1:])
    #     loss_ab_j = self.l1(y_j_lab[:, 1:], img_j_lab[:, 1:])
    #     reconstruction_loss = (loss_L_i + loss_L_j) + w_ab * (loss_ab_i + loss_ab_j)

    #     def moments(x):
    #         m = x.mean(dim=[2, 3])
    #         s = x.std(dim=[2, 3])
    #         return m, s

    #     mu_i, std_i = moments(y_i_lab[:, 1:])
    #     mu_j, std_j = moments(y_j_lab[:, 1:])
    #     mu_i_t, std_i_t = moments(img_i_lab[:, 1:])
    #     mu_j_t, std_j_t = moments(img_j_lab[:, 1:])
    #     moment_loss = self.l1(mu_i, mu_i_t) + self.l1(std_i, std_i_t) + self.l1(mu_j, mu_j_t) + self.l1(std_j, std_j_t)

    #     def cov2(x):
    #         b, c, h, w = x.shape
    #         x = x.view(b, c, h * w)
    #         x = x - x.mean(dim=2, keepdim=True)
    #         cov = torch.matmul(x, x.transpose(1, 2)) / (h * w - 1)
    #         return cov

    #     cov_i_pred = cov2(y_i_lab[:, 1:])
    #     cov_j_pred = cov2(y_j_lab[:, 1:])
    #     cov_i_targ = cov2(img_i_lab[:, 1:])
    #     cov_j_targ = cov2(img_j_lab[:, 1:])
    #     cov_loss = self.l1(cov_i_pred, cov_i_targ) + self.l1(cov_j_pred, cov_j_targ)

    #     consistency_loss = self.l2(z_i[:, 0], z_j[:, 0])
    #     total_loss = reconstruction_loss + lam_cons * consistency_loss + lam_moment * moment_loss + lam_cov * cov_loss
    #     '''
    #     tensor(0.3161, device='cuda:0', grad_fn=<AddBackward0>) moment_lossmoment_loss
    #     tensor(0.0158, device='cuda:0', grad_fn=<AddBackward0>) lam_momentlam_moment
    #     tensor(1.1887, device='cuda:0', grad_fn=<AddBackward0>) reconstruction_lossreconstruction_lossreconstruction_loss
    #     tensor(2.3260, device='cuda:0', grad_fn=<MseLossBackward0>) consistency_lossconsistency_lossconsistency_loss
    #     '''

    #     return {
    #         'consistency_loss': consistency_loss,
    #         'reconstruction_loss': reconstruction_loss,
    #         'moment_loss': moment_loss,
    #         'total_loss': total_loss,
    #         'cov_loss': cov_loss
    #     }
    
    # 加入了lab中的损失
    def _compute_losses_moment(self, z_i, z_j, y_i, y_j, img_i, img_j):
        consistency_loss = self.l2(z_i, z_j)
        recon_loss = self.l1(y_i, img_i) + self.l1(y_j, img_j)

        # 3. Color Moment Loss (核心：颜色统计量损失)
        y_i_lab = rgb_to_lab(torch.clamp(y_i, 0, 1))
        img_j_lab = rgb_to_lab(img_j)
        
        y_j_lab = rgb_to_lab(torch.clamp(y_j, 0, 1))
        img_i_lab = rgb_to_lab(img_i)

        def get_moments(feat):
            mean = torch.mean(feat, dim=[2, 3])
            std = torch.std(feat, dim=[2, 3])
            return mean, std

        mu_pred_i, std_pred_i = get_moments(y_i_lab)
        mu_targ_j, std_targ_j = get_moments(img_j_lab)
        
        mu_pred_j, std_pred_j = get_moments(y_j_lab)
        mu_targ_i, std_targ_i = get_moments(img_i_lab)

        moment_loss = (self.l1(mu_pred_i, mu_targ_j) + self.l1(std_pred_i, std_targ_j)) + \
                      (self.l1(mu_pred_j, mu_targ_i) + self.l1(std_pred_j, std_targ_i))

        print(moment_loss,"moment_lossmoment_loss")
        total_loss = (self.cfg.criterion.lambda_consistency * consistency_loss + 
                      recon_loss + 
                      moment_loss * 5.0) 

        return {
            'consistency_loss': consistency_loss,
            'reconstruction_loss': recon_loss,
            'moment_loss': moment_loss,
            'total_loss': total_loss
        }

    # TODO - 换成lab之后的损失需要稍微改变
    # 将l1损失在lab上进行区分 更在乎颜色; 加上了统计量的损失
    def _compute_losses_moment(self, z_i, z_j, y_i, y_j, img_i, img_j):
        """Compute all losses for the model outputs."""
        # 重建损失 - 输出和输入 
        # 连续性损失 - 内容图的一致性
        # 这里z_i和z_j是内容图，y_i和y_j是风格迁移后的结果图
        # 更多损失 - 对比损失，
        consistency_loss = self.l2(z_i, z_j)
        # consistency_loss = self.l2(z_i[:, 0], z_j[:, 0])
    
        # 给颜色更多的权重
        def weighted_lab_l1(pred, target, w_L=1.0, w_ab=2.0):
            loss_L = self.l1(pred[:, 0], target[:, 0])       # L 通道
            loss_a = self.l1(pred[:, 1], target[:, 1])       # a 通道
            loss_b = self.l1(pred[:, 2], target[:, 2])       # b 通道
            return w_L * loss_L + w_ab * (loss_a + loss_b)
        
        recon_i = weighted_lab_l1(y_i, img_i, w_L=1, w_ab=2)
        recon_j = weighted_lab_l1(y_j, img_j, w_L=1, w_ab=2)
        reconstruction_loss = recon_i + recon_j
        
        def moment_loss(pred_lab, target_lab, channels=[0, 1, 2]): 
            loss = 0.0
            for c in channels:
                pred_mean = pred_lab[:, c].mean(dim=[1, 2])
                pred_std  = pred_lab[:, c].std(dim=[1, 2])
                targ_mean = target_lab[:, c].mean(dim=[1, 2])
                targ_std  = target_lab[:, c].std(dim=[1, 2])
                loss += self.l1(pred_mean, targ_mean) + self.l1(pred_std, targ_std)
            return loss
        
        # y_i 是 img_i 用 img_j 风格迁移的结果 → 应匹配 img_j  不给l施加约束
        style_moment_loss_i = moment_loss(y_i, img_j, channels=[ 1, 2])
        style_moment_loss_j = moment_loss(y_j, img_i, channels=[1, 2])
        moment_loss_total = style_moment_loss_i + style_moment_loss_j
        
        total_loss = reconstruction_loss + self.cfg.criterion.lambda_consistency * consistency_loss + moment_loss_total
        
        return {
            'consistency_loss': consistency_loss,
            'reconstruction_loss': reconstruction_loss,
            'moment_loss': moment_loss_total,
            'total_loss': total_loss
        }
    
    # 将l1损失在lab上进行区分 更在乎颜色
    def _compute_losses_lab(self, z_i, z_j, y_i, y_j, img_i, img_j):
        """Compute all losses for the model outputs."""
        # 重建损失 - 输出和输入 
        # 连续性损失 - 内容图的一致性
        # 这里z_i和z_j是内容图，y_i和y_j是风格迁移后的结果图
        # 更多损失 - 对比损失，
        # consistency_loss = self.l2(z_i, z_j)
        consistency_loss = self.l2(z_i[:, 0], z_j[:, 0])
        
        # 给颜色更多的权重
        def weighted_lab_l1(pred, target, w_L=1.0, w_ab=2.0):
            loss_L = self.l1(pred[:, 0], target[:, 0])       # L 通道
            loss_a = self.l1(pred[:, 1], target[:, 1])       # a 通道
            loss_b = self.l1(pred[:, 2], target[:, 2])       # b 通道
            return w_L * loss_L + w_ab * (loss_a + loss_b)
        
        recon_i = weighted_lab_l1(y_i, img_i, w_L=1, w_ab=2)
        recon_j = weighted_lab_l1(y_j, img_j, w_L=1, w_ab=2)
        reconstruction_loss = recon_i + recon_j

        total_loss = reconstruction_loss + self.cfg.criterion.lambda_consistency * consistency_loss
        
        return {
            'consistency_loss': consistency_loss,
            'reconstruction_loss': reconstruction_loss,
            'total_loss': total_loss
        }

    def _compute_losses(self, z_i, z_j, y_i, y_j, img_i, img_j):
        """Compute all losses for the model outputs."""
        # 重建损失 - 输出和输入 
        # 连续性损失 - 内容图的一致性
        # 这里z_i和z_j是内容图，y_i和y_j是风格迁移后的结果图
        # 更多损失 - 对比损失，
        consistency_loss = self.l2(z_i, z_j)
        reconstruction_loss = self.l1(y_i, img_i) + self.l1(y_j, img_j)
        total_loss = reconstruction_loss + self.cfg.criterion.lambda_consistency * consistency_loss
        
        return {
            'consistency_loss': consistency_loss,
            'reconstruction_loss': reconstruction_loss,
            'total_loss': total_loss
        }
    
    def training_step(self, batch, batch_idx):
        if self.current_stage == "pretrain":
            img_i, img_j, z_i, z_j = self._process_batch_pretrain(batch)
            losses = self._compute_losses_pretrain(z_i, z_j)
        else:
            img_i, img_j, z_i, z_j, y_i, y_j = self._process_batch_full_train(batch)
            losses = self._compute_losses_full_train(z_i, z_j, y_i, y_j, img_i, img_j)
        self._log_metrics(losses, 'train')
        return losses['total_loss']
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.current_stage == "pretrain":
            params = list(self.net.style_encoder.parameters()) + [self.net.transform_p, self.net.transform_q]
        else:
            params = self.net.parameters()
            
        optimizer = get_optimizer(
            opt_mode=self.cfg.train.optimizer.mode,
            net_params=params,
            **(self.cfg.train.optimizer[self.cfg.train.optimizer.mode])
        )
        
        scheduler = get_scheduler(
            sched_mode=self.cfg.train.scheduler.mode,
            optimizer=optimizer,
            **(self.cfg.train.scheduler[self.cfg.train.scheduler.mode])
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': self.cfg.train.check_val_every_n_epoch,
                'monitor': self.cfg.train.scheduler.monitor,
            }
        }

    def validation_step(self, batch, batch_idx):
        if self.current_stage == "pretrain":
            img_i, img_j, z_i, z_j = self._process_batch_pretrain(batch)
            losses = self._compute_losses_pretrain(z_i, z_j)
        else:
            img_i, img_j, z_i, z_j, y_i, y_j = self._process_batch_full_train(batch)
            losses = self._compute_losses_full_train(z_i, z_j, y_i, y_j, img_i, img_j)
        
        self._log_metrics(losses, 'val')
        print(batch,"batchbatchbatchbatch")
        
        phase = 'test' if self.trainer.testing else 'val'
        if phase == 'test':
            self.visualize_results(
                phase=phase,
                img_i=img_i[0], img_j=img_j[0],
                z_i=z_i[0], z_j=z_j[0],
                y_i=y_j if self.current_stage == "pretrain" else y_i[0],
                y_j=y_i if self.current_stage == "pretrain" else y_j[0],
                batch_idx=batch_idx
            )
        elif batch_idx == 0:
            self.visualize_results(
                phase=phase,
                img_i=img_i[0], img_j=img_j[0],
                z_i=z_i[0], z_j=z_j[0],
                y_i=y_j if self.current_stage == "pretrain" else y_i[0],
                y_j=y_i if self.current_stage == "pretrain" else y_j[0],
                batch_idx=batch_idx
            )
        
        return losses['total_loss']

    def visualize_results(self, phase, img_i, img_j, z_i, z_j, y_i, y_j, batch_idx=0):
        grid_img = self.make_grid_image(img_i, img_j, z_i, z_j, y_i, y_j)
        
        save_root = os.path.join(self.cfg.path.result_path, 'epoch_{:04d}'.format(self.current_epoch), phase)
        os.makedirs(save_root, exist_ok=True)  
        
        filename = f"epoch_{self.current_epoch:04d}_batch_{batch_idx:03d}.png"
        save_path = os.path.join(save_root, filename)
        
        cv2.imwrite(save_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        print(f"[Visualization Saved] Stage: {self.current_stage} | Phase: {phase} | Path: {save_path}")

    def make_grid_image(self, img_i, img_j, z_i, z_j, y_i, y_j):
        if self.current_stage == "pretrain":
            images = [img_i, z_i, img_j, z_j]
        else:
            images = [img_i, z_i, y_j, img_j, z_j, y_i]
        
        images_np = []
        for img in images:
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)  # 反归一化到0-255
            images_np.append(img_np)
        
        if self.current_stage == "pretrain":
            row1 = np.concatenate([images_np[0], images_np[1]], axis=1)
            row2 = np.concatenate([images_np[2], images_np[3]], axis=1)
        else:
            row1 = np.concatenate([images_np[0], images_np[1], images_np[2]], axis=1)
            row2 = np.concatenate([images_np[3], images_np[4], images_np[5]], axis=1)
        
        grid_img = np.concatenate([row1, row2], axis=0)
        return grid_img

    def _log_metrics(self, losses, phase):
        log_prefix = f'{phase}/{self.current_stage}/losses' if phase != 'test' else 'test/losses'
        log_dict = {
            f'{log_prefix}/consistency_loss': losses['consistency_loss'],
            f'{log_prefix}/total_loss': losses['total_loss']
        }
        
        if self.current_stage == "full_train":
            log_dict[f'{log_prefix}/reconstruction_loss'] = losses['reconstruction_loss']
            if 'moment_loss' in losses:
                log_dict[f'{log_prefix}/moment_loss'] = losses['moment_loss']
            if 'cov_loss' in losses:
                log_dict[f'{log_prefix}/cov_loss'] = losses['cov_loss']
        
        self.log_dict(
            log_dict,
            on_step=(phase == 'train'),
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True
        )
