"""Masked-mean adaptive pooling of a cached 16x16 masked patch grid to a smaller grid (e.g. 6x6), training and
evaluation must use this same function."""
import torch, torch.nn.functional as F

def pool_patches(tok: torch.Tensor, grid_in: int = 16, grid_out: int = 6) -> torch.Tensor:
    """tok (..., grid_in*grid_in, D) -> (..., grid_out*grid_out, D); all-zero (masked) cells do not enter the mean; a target cell with no kept cell -> 0."""
    lead = tok.shape[:-2]; D = tok.shape[-1]
    x = tok.reshape(-1, grid_in, grid_in, D).permute(0, 3, 1, 2).float()          # (B,D,g,g)
    keep = (x.abs().sum(1, keepdim=True) > 0).float()                             # (B,1,g,g)
    s = F.adaptive_avg_pool2d(x * keep, grid_out) ; c = F.adaptive_avg_pool2d(keep, grid_out)
    y = torch.where(c > 0, s / c.clamp(min=1e-6), torch.zeros_like(s))
    return y.permute(0, 2, 3, 1).reshape(*lead, grid_out * grid_out, D).to(tok.dtype)
