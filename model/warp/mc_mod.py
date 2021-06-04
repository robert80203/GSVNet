import torch
import torch.nn as nn
from model.warp.grid_sampler import GridSampler

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.0).float()

    def forward(self, x):
        if self.sigma != 0:
            self.noise = self.noise.to(x.device)
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class MC_Module(nn.Module):

    def __init__(self, h=None, w=None, train_noise_sigma=0, eval_noise_sigma=0):
        super().__init__()
        self.grid_sampler = GridSampler(h=h, w=w)
        self.train_gaussian_noise = GaussianNoise(train_noise_sigma)
        self.eval_gaussian_noise = GaussianNoise(eval_noise_sigma)


def forward(self, ref_pred, ref_mv, ref_weight=None, mask=None):
        ref_mv = ref_mv / 4
        if self.training:
            ref_mv = self.train_gaussian_noise(ref_mv)
        else:
            ref_mv = self.eval_gaussian_noise(ref_mv)
        warped_pred = self.grid_sampler(ref_pred, ref_mv, mask)
        if ref_weight is not None:
            warped_pred = warped_pred * ref_weight
        result = warped_pred.sum(dim=0)
        return result

class MC_Module_Batch(nn.Module):

    def __init__(self, h=None, w=None, train_noise_sigma=0, eval_noise_sigma=0):
        super().__init__()
        self.grid_sampler = GridSampler(h=h, w=w)
        self.train_gaussian_noise = GaussianNoise(train_noise_sigma)
        self.eval_gaussian_noise = GaussianNoise(eval_noise_sigma)

    def forward(self, ref_pred, ref_mv, ref_weight=None, mask=None):
        #ref_mv *= -1
        batch_size, ref_num = ref_pred.size()[:2]
        ref_pred = ref_pred.view(batch_size * ref_num, *ref_pred.size()[-3:])
        ref_mv = ref_mv.view(batch_size * ref_num, *ref_mv.size()[-3:])
        ref_mv = ref_mv / 4
        if self.training:
            ref_mv = self.train_gaussian_noise(ref_mv)
        else:
            ref_mv = self.eval_gaussian_noise(ref_mv)


        warped_pred = self.grid_sampler(ref_pred, ref_mv , mask)
        if ref_weight is not None:
            ref_weight = ref_weight.view(batch_size * ref_num, *ref_weight.size()[-3:])
            warped_pred = warped_pred * ref_weight
        warped_pred = warped_pred.view(batch_size, ref_num, *warped_pred.size()[-3:])
        result = warped_pred.sum(dim=1)
        return result

if __name__ == "__main__":
    mc_warp = MC_Module()
    import torch

    mc_warp = MC_Module_Batch()
    ref_pred = torch.randn(3, 2, 19, 512, 1024)
    ref_mv = torch.randn(3, 2, 2, 512, 1024)
    ref_weight = torch.randn(3, 2, 1, 512, 1024)
    warped_pred = mc_warp(ref_pred, ref_mv, None)
    print(warped_pred.size())
