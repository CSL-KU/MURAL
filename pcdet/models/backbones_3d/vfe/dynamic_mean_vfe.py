import torch
from typing import Tuple, Optional

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from typing import Tuple

class DynamicMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        all_pc_ranges = self.model_cfg.get('ALL_PC_RANGES', None)

        self.voxel_params = []
        for i, resdiv in enumerate(res_divs):
            if all_pc_ranges is not None:
                point_cloud_range = all_pc_ranges[i]
            voxel_size_tmp = [vs * resdiv for vs in voxel_size[:2]]
            grid_size_tmp = [int(gs / resdiv) for gs in grid_size[:2]]
            self.voxel_params.append((
                    voxel_size_tmp[0], #voxel_x
                    voxel_size_tmp[1], #voxel_y
                    voxel_size[2], #voxel_z constant
                    voxel_size_tmp[0] / 2 + point_cloud_range[0], #x_offset
                    voxel_size_tmp[1] / 2 + point_cloud_range[1], #y_offset
                    voxel_size[2] / 2 + point_cloud_range[2], #z_offset
                    grid_size_tmp[0] * grid_size_tmp[1] * grid_size[2], #scale_xyz
                    grid_size_tmp[1] * grid_size[2], #scale_yz
                    grid_size[2], #scale_z
                    torch.tensor(grid_size_tmp + [grid_size[2]]).cuda(), # grid_size
                    torch.tensor(voxel_size_tmp + [voxel_size[2]]).cuda(),
                    torch.tensor(point_cloud_range).cuda()
            ))
        self.set_params(0)

    def set_params(self, idx):
        self.voxel_x, self.voxel_y, self.voxel_z, \
                self.x_offset, self.y_offset, self.z_offset,  \
                self.scale_xyz, self.scale_yz, self.scale_z, \
                self.grid_size, self.voxel_size, self.point_cloud_range = self.voxel_params[idx]

    def adjust_voxel_size_wrt_resolution(self, res_idx):
        self.set_params(res_idx)

    def get_output_feature_dim(self):
        return self.num_point_features


    @torch.no_grad()
    def calc_point_coords(self, points : torch.Tensor) -> torch.Tensor:
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        return point_coords

    @torch.no_grad()
    def forward_gen_voxels(self, points : torch.Tensor, point_coords : Optional[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        if point_coords is None:
            point_coords = self.calc_point_coords(points)
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        num_out_inds = torch.max(unq_inv) + 1
        points_mean = torch.zeros((num_out_inds, points_data.size(1)),
                dtype=points_data.dtype, device=points_data.device)
        torch_scatter.scatter_mean(points_data, unq_inv, dim=0, out=points_mean)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        return voxel_coords.contiguous(), points_mean.contiguous()

    def forward(self, points : torch.Tensor, point_coords : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        voxel_coords, features = self.forward_gen_voxels(points, point_coords)
        return voxel_coords, features
