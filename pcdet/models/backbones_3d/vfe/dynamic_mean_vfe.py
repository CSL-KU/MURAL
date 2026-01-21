import torch
from typing import Tuple, Optional

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from typing import Tuple, List

class DynamicMeanVFE(VFETemplate):
    voxel_x : float
    voxel_y : float
    voxel_z : float
    x_offset : float
    y_offset : float
    z_offset : float
    scale_xyz: int
    scale_yz: int
    scale_z: int
    grid_size : torch.Tensor
    voxel_size : torch.Tensor
    point_cloud_range : torch.Tensor
    voxel_params : List[Tuple[float, float, float, float, float, float, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]]
    num_point_features: int

    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = int(num_point_features)
        res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        all_pc_ranges = self.model_cfg.get('ALL_PC_RANGES', None)

        self.voxel_params = []
        for i, resdiv in enumerate(res_divs):
            if all_pc_ranges is not None:
                point_cloud_range = all_pc_ranges[i]
            voxel_size_tmp = [vs * resdiv for vs in voxel_size[:2]]
            grid_size_tmp = [int(gs / resdiv) for gs in grid_size[:2]]
            self.voxel_params.append((
                    float(voxel_size_tmp[0]), #voxel_x
                    float(voxel_size_tmp[1]), #voxel_y
                    float(voxel_size[2]), #voxel_z constant
                    float(voxel_size_tmp[0] / 2 + point_cloud_range[0]), #x_offset
                    float(voxel_size_tmp[1] / 2 + point_cloud_range[1]), #y_offset
                    float(voxel_size[2] / 2 + point_cloud_range[2]), #z_offset
                    int(grid_size_tmp[0] * grid_size_tmp[1] * grid_size[2]), #scale_xyz
                    int(grid_size_tmp[1] * grid_size[2]), #scale_yz
                    int(grid_size[2]), #scale_z
                    torch.tensor(grid_size_tmp + [grid_size[2]]).cuda(), # grid_size
                    torch.tensor(voxel_size_tmp + [voxel_size[2]]).cuda(),
                    torch.tensor(point_cloud_range).cuda()
            ))
        self.set_params(0)

    @torch.jit.export
    def set_params(self, idx : int):
        self.voxel_x, self.voxel_y, self.voxel_z, \
                self.x_offset, self.y_offset, self.z_offset,  \
                self.scale_xyz, self.scale_yz, self.scale_z, \
                self.grid_size, self.voxel_size, self.point_cloud_range = self.voxel_params[idx]

    @torch.jit.export
    def adjust_voxel_size_wrt_resolution(self, res_idx : int):
        self.set_params(res_idx)

    @torch.jit.export
    def get_output_feature_dim(self) -> int:
        return self.num_point_features

    @torch.jit.export
    def calc_point_coords(self, points : torch.Tensor) -> torch.Tensor:
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        return point_coords

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

        num_out_inds = int((torch.max(unq_inv) + 1).item())
        points_mean = torch.zeros([num_out_inds, int(points_data.size(1))],
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
