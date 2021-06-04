import torch
import torch.nn as nn
import torch.nn.functional as F


class GridSampler(nn.Module):

    def gen_basegrid(self, h, w):
        grid_col = torch.linspace(-1, 1, h).unsqueeze(1).expand(h, w)  # each col is the same
        grid_row = torch.linspace(-1, 1, w).unsqueeze(0).expand(h, w)  # each row is the same
        return torch.cat([grid_row.unsqueeze(-1), grid_col.unsqueeze(-1)], dim=2)

    def gen_grid(self, flow):

        # batchx2xhxw
        _, _, cur_h, cur_w = flow.size()

        if self.h != cur_h or self.w != cur_w:
            self.h = cur_h
            self.w = cur_w
            self.base_grid = self.gen_basegrid(cur_h, cur_w )

        if self.base_grid.device != flow.device:
            self.base_grid = self.base_grid.to(flow.device)

        # adjflow is  (N, Hout, Wout, 2),
        adj_flow = flow.transpose(1,2).transpose(2,3)
        adj_flow[:,:,:,0] /= (cur_w-1) # divide by size
        adj_flow[:,:,:,1] /= (cur_h-1)
        adj_flow *= 2
        # for every member in a batch, add in its positional information
        adj_flow[:] = adj_flow[:] + self.base_grid
        return adj_flow

    def __init__(self, h=None, w=None):
        super().__init__()
        # Adjusted grid flow to be [-1,1]
        self.h, self.w = h, w
        if self.h is not None and self.w is not None:
            self.base_grid = self.gen_basegrid(h, w)

    def forward(self, key_feat, flow, mask=None):
        grid = self.gen_grid(flow)
        if mask is not None:
            mask = mask.unsqueeze(dim=-1).expand_as(grid)
            grid[mask] = -99 # don't sample anything at the mask
        return F.grid_sample(key_feat, grid, mode='bilinear', padding_mode="zeros")
