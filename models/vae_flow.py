import torch
from torch.nn import Module
from .common import *
from .encoders import *
from .diffusion import *
from .flow import *


class FlowVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)  # 初始化PointNet编码器
        self.flow = build_latent_flow(args)  # 使用给定参数构建潜在流（latent flow）
        self.diffusion = DiffusionPoint(
            net=PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )  # 初始化扩散点（DiffusionPoint），包括PointwiseNet和VarianceSchedule的配置

    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)# 使用PointNet编码输入点云x，得到均值z_mu和方差z_sigma
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)# 通过重参数化技巧采样潜在变量z
        
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )# 计算Q(z|X)的熵
        # P(z)，先验概率，由流参数化：z -> w。
        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(x, z) # 计算负对数ELBO（边际似然的下界）


        # Loss
        loss_entropy = -entropy.mean()# 熵的均值作为损失项
        loss_prior = -log_pz.mean()# 先验概率的对数的均值作为损失项
        loss_recons = neg_elbo# 边际似然的下界作为重建损失项
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo# 总体损失

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss

    def get_latent(self, x):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)  # 使用PointNet编码输入点云x，得到均值z_mu和方差z_sigma
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)# 通过重参数化技巧采样潜在变量z

        return z



    def sample1(self, x,w,flexibility,truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
            # Reverse: z <- w.
        t = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample1(x,context=t,flexibility=flexibility)
        return samples

    def sample(self, w, num_points, flexibility, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples
