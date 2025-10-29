import torch
import numba
import numpy as np
from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm1d, ResAwareBatchNorm2d
from pcdet.models.backbones_3d.spconv_backbone_2d import PillarRes18BackBone8x_pillar_calc
from pcdet.models.backbones_3d.spconv_backbone import VoxelResBackBone8x_voxel_calc_3d
from typing import Dict, List, Tuple, Optional, Final

def set_bn_resolution(resawarebns, res_idx):
    for rabn in resawarebns:
        rabn.setResIndex(res_idx)

def interpolate_batch_norms(resawarebns, max_grid_l):
    for rabn in resawarebns:
        rabn.interpolate(max_grid_l)

def get_all_resawarebn(model):
    resaware1dbns, resaware2dbns = [], []
    for module in model.modules():
        if isinstance(module, ResAwareBatchNorm1d):
            resaware1dbns.append(module)
        elif isinstance(module, ResAwareBatchNorm2d):
            resaware2dbns.append(module)
    return resaware1dbns, resaware2dbns

@numba.jit(nopython=True)
def get_xminmax_from_pc0(pc0):
    xminmax = np.empty((pc0.shape[0], 2), dtype=np.int32)
    for i in range(pc0.shape[0]):
        min_idx, max_idx = -1, -1
        for j in range(pc0.shape[1]):
            if pc0[i,j] != 0:
                if min_idx == -1:
                    min_idx = j
                max_idx = j
        xminmax[i, 0] = min_idx
        xminmax[i, 1] = max_idx
    return xminmax

@torch.jit.script
def get_slice_range(down_scale_factor : int, x_min: int, x_max: int, maxsz: int) \
        -> Tuple[int, int]:
    dsf = down_scale_factor
    x_min, x_max = x_min // dsf, x_max // dsf + 1
    denom = 4 # denom is dependent on strides within the dense covs
    minsz = 16

    rng = (x_max - x_min)
    if rng < minsz:
        diff = minsz - rng
        if x_max + diff <= maxsz:
            x_max += diff
        elif x_min - diff >= 0:
            x_min -= diff
        #else: # very unlikely
        #    pass
        rng = (x_max - x_min)

    pad = denom - (rng % denom)
    if pad == denom:
        pass
    elif x_min >= pad: # pad from left
        x_min -= pad
    elif (maxsz - x_max) >= pad: # pad from right
        x_max += pad
    else: # don't slice
        x_min, x_max = 0 , maxsz
    return x_min, x_max

@torch.jit.script
def slice_tensor(down_scale_factor : int, x_min: int, x_max: int, inp : torch.Tensor) \
        -> Tuple[torch.Tensor, int, int]:
    x_min, x_max = get_slice_range(down_scale_factor, x_min, x_max, inp.size(3))
    return inp[..., x_min:x_max].contiguous(), x_min, x_max

# This will be used to generate the onnx
class DenseConvsPipeline(torch.nn.Module):
    def __init__(self, backbone_3d, backbone_2d, dense_head, dettype):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head
        self.optimize_attr_convs = dense_head.model_cfg.OPTIMIZE_ATTR_CONVS
        self.dettype = dettype

    def forward(self, x_conv4 : torch.Tensor) -> List[torch.Tensor]:
        if self.dettype == 'PillarNet':
            x_conv5 = self.backbone_3d.forward_dense(x_conv4)
            data_dict = self.backbone_2d({"multi_scale_2d_features" :
                {"x_conv4": x_conv4, "x_conv5": x_conv5}})
        elif self.dettype == 'PointPillarsCP' or self.dettype == 'CenterPointVN':
            data_dict = {'spatial_features_2d': self.backbone_2d(x_conv4)}

        if self.optimize_attr_convs:
            outputs = self.dense_head.forward_pre(data_dict['spatial_features_2d'])
            shr_conv_outp = outputs[0]
            heatmaps = outputs[1:]

            topk_outputs = self.dense_head.forward_topk_trt(heatmaps)

            ys_all = [topk_outp[2] for topk_outp in topk_outputs]
            xs_all = [topk_outp[3] for topk_outp in topk_outputs]

            sliced_inp = self.dense_head.slice_shr_conv_outp(shr_conv_outp, ys_all, xs_all)
            outputs = self.dense_head.forward_sliced_inp_trt(sliced_inp)
            for topk_output in topk_outputs:
                outputs += topk_output

            return outputs
        else:
            return self.dense_head.forward_up_to_topk(data_dict['spatial_features_2d'])

class MultiVoxelCounter(torch.nn.Module):
    # Pass the args in cpu , pillar sizes should be [N,2], pc_range should be [6]
    grid_sizes: Final[List[List[int]]]
    num_slices: Final[List[int]]
    num_res : Final[int]
    slice_sz : Final[int]
    pillar_sizes : torch.Tensor
    pc_range_min: torch.Tensor
    pillar_sizes_cpu : torch.Tensor
    pc_range_min_cpu: torch.Tensor
    dettype: str

    def __init__(self, pillar_sizes : torch.Tensor, pc_ranges : torch.Tensor,
                 slice_sz: int = 32, dettype: str = 'PillarNet'):
        super().__init__()
        self.dettype = dettype

        # For 3D (CenterPointVN), keep all 3 dimensions; for 2D (PillarNet), use only XY
        if dettype == 'CenterPointVN' and pillar_sizes.size(1) > 2:
            # Keep 3D pillar sizes
            is_3d = True
            pillar_sizes_3d = pillar_sizes
            pillar_sizes_2d = pillar_sizes[:, :2]  # For backward compatibility
        else:
            # Use 2D pillar sizes
            is_3d = False
            if pillar_sizes.size(1) > 2:
                pillar_sizes = pillar_sizes[:, :2]
            pillar_sizes_2d = pillar_sizes
            pillar_sizes_3d = None

        self.num_res = len(pillar_sizes_2d)

        # Initialize grid sizes and other parameters
        if is_3d:
            grid_sizes = torch.empty((self.num_res, 3), dtype=torch.int)
            num_slices = [0] * self.num_res
            pc_range_mins = torch.empty((self.num_res, 3))
            for i, (ps, pc_range) in enumerate(zip(pillar_sizes_3d, pc_ranges)):
                xyz_length = pc_range[[3,4,5]] - pc_range[[0,1,2]]
                grid_sizes[i] = torch.round(xyz_length / ps)
                # num_slices is computed on H dimension (index 1)
                num_slices[i] = (grid_sizes[i, 1] // slice_sz).item()
                pc_range_mins[i] = pc_range[[0,1,2]]
        else:
            grid_sizes = torch.empty((self.num_res, 2), dtype=torch.int)
            num_slices = [0] * self.num_res
            pc_range_mins = torch.empty((self.num_res, 2))
            for i, (ps, pc_range) in enumerate(zip(pillar_sizes_2d, pc_ranges)):
                xy_length = pc_range[[3,4]] - pc_range[[0,1]]
                grid_sizes[i] = torch.round(xy_length / ps)
                num_slices[i] = (grid_sizes[i, 0] // slice_sz).item()
                pc_range_mins[i] = pc_range[[0,1]]

        self.slice_sz = slice_sz
        self.grid_sizes = grid_sizes.tolist()
        self.is_3d = is_3d

        if is_3d:
            self.pillar_sizes_cpu = pillar_sizes_3d
            self.pillar_sizes_2d_cpu = pillar_sizes_2d
        else:
            self.pillar_sizes_cpu = pillar_sizes_2d
            self.pillar_sizes_2d_cpu = pillar_sizes_2d

        self.pc_range_mins_cpu = pc_range_mins
        self.pillar_sizes = pillar_sizes_2d.cuda()
        self.pc_range_mins = pc_range_mins.cuda()
        if is_3d:
            self.pillar_sizes_3d = pillar_sizes_3d.cuda()
            self.pc_range_mins_3d = pc_range_mins.cuda()
        self.num_slices = num_slices

        print('pillar_sizes', self.pillar_sizes_cpu)
        #print('pc_range_mins', pc_range_mins)
        print('num_slices', num_slices)
        print('grid_sizes', self.grid_sizes)
        print('dettype', dettype)
        print('is_3d', is_3d)

        if dettype == 'PillarNet':
            self.voxel_calc_func = PillarRes18BackBone8x_pillar_calc
        elif dettype == 'CenterPointVN':
            self.voxel_calc_func = VoxelResBackBone8x_voxel_calc_3d


    #@torch.jit.export
    def forward_pillar(self, points_xy : torch.Tensor, first_res_idx : int = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        fri = first_res_idx
        cur_num_res = self.num_res - fri
        grid_sz = self.grid_sizes[fri] # get biggest grid
        batch_grid = torch.zeros([1, cur_num_res, grid_sz[0], grid_sz[1]],
                                      device=points_xy.device, dtype=torch.float32)

        expanded_pts = points_xy.unsqueeze(1).expand(-1, cur_num_res, -1)
        batch_point_coords = ((expanded_pts - self.pc_range_mins[fri:]) / \
                self.pillar_sizes[fri:]).int()

        inds = torch.arange(cur_num_res, device=points_xy.device).unsqueeze(0)
        inds = inds.expand(batch_point_coords.size(0), -1).flatten()
        batch_grid[:, inds, batch_point_coords[:, :, 1].flatten(),
                   batch_point_coords[:, :, 0].flatten()] = 1.0

        pc0, pillar_counts = self.voxel_calc_func(batch_grid, self.num_slices[fri])
        return pc0, pillar_counts.T

    #@torch.jit.export
    def forward_voxel(self, points_xyz : torch.Tensor, first_res_idx : int = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D version that handles XYZ points.
        Creates a 3D occupancy grid of shape (N, C, H, W, L) where C=num_res.
        Following max_pool3d convention: (N, C, D, H, W) -> in our case (N, C, Y, X, Z)
        where Y=height, X=width, Z=depth
        """
        fri = first_res_idx
        cur_num_res = self.num_res - fri
        grid_sz = self.grid_sizes[fri] # [X, Y, Z] sizes

        # Create 3D grid in NCDHW: (N=1, C=num_res, D=Z, H=Y, W=X)
        batch_grid = torch.zeros([1, cur_num_res, grid_sz[2], grid_sz[1], grid_sz[0]],
                                      device=points_xyz.device, dtype=torch.float32)

        expanded_pts = points_xyz.unsqueeze(1).expand(-1, cur_num_res, -1)
        batch_point_coords = ((expanded_pts - self.pc_range_mins_3d[fri:]) / \
                self.pillar_sizes_3d[fri:]).int()

        inds = torch.arange(cur_num_res, device=points_xyz.device).unsqueeze(0)
        inds = inds.expand(batch_point_coords.size(0), -1).flatten()

        # Set occupancy: grid[:, res_idx, D, H, W] = 1.0
        # NCDHW mapping: z->D (dim 2), y->H (dim 3), x->W (dim 4)
        batch_grid[:, inds,
                   batch_point_coords[:, :, 2].flatten(),  # Z -> D dimension
                   batch_point_coords[:, :, 1].flatten(),  # Y -> H dimension
                   batch_point_coords[:, :, 0].flatten()] = 1.0  # X -> W dimension

        pc0, voxel_counts = self.voxel_calc_func(batch_grid, self.num_slices[fri])
        return pc0, voxel_counts.T

    def forward(self, points_inds : torch.Tensor, first_res_idx : int = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dettype == 'PillarNet':
            return self.forward_pillar(points_inds, first_res_idx)
        else: # self.dettype == 'CenterPointVN':
            return self.forward_voxel(points_inds, first_res_idx)

    @torch.jit.export
    def get_minmax_inds(self, points_x : torch.Tensor) -> torch.Tensor:
        x_minmax = torch.empty((self.num_res, 2), dtype=torch.int)
        xmin, xmax = torch.aminmax(points_x)
        minmax = torch.cat((xmin.unsqueeze(-1), xmax.unsqueeze(-1))).cpu()
        for i in range(self.num_res):
            minmax_s = minmax - self.pc_range_mins_cpu[i, 0]
            minmax_s = (minmax_s / self.pillar_sizes_cpu[i]).int()
            x_minmax[i] = minmax_s // self.slice_sz
        return x_minmax

