from functools import partial

import torch.nn as nn
import torch
from typing import Tuple

from ...utils.spconv_utils import replace_feature, spconv
from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm1d

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

#static method
def VoxelResBackBone8x_voxel_calc_3d(bev_img_3d : torch.Tensor, num_slices : int) \
        -> Tuple[torch.Tensor,torch.Tensor]:
    """
    3D version that handles inputs in NCDHW format.
    Max pooling parameters match the VoxelResBackBone8x architecture:
    - conv2: stride=2, padding=1
    - conv3: stride=2, padding=1
    - conv4: stride=2, padding=(0, 1, 1) - asymmetric padding on depth dimension

    Args:
        bev_img_3d: Input tensor of shape (N, C, D, H, W)
        num_slices: Number of slices used to partition the depth (D) dimension

    Returns:
        p0_: Tensor of shape (num_res, num_slices)
        stacked: Tensor of shape (4, num_res) containing counts at different stages
    """
    bi_sz = bev_img_3d.shape  # N, C, D, H, W
    p0_ = bev_img_3d.view(
        bi_sz[0], bi_sz[1], bi_sz[2] , bi_sz[3], num_slices, bi_sz[4] // num_slices
    ).sum(dim=[0, 2, 3, 5])

    # Apply 3D max pooling matching the architecture's conv parameters
    # conv2: kernel_size=3, stride=2, padding=1 (all dimensions)
    x1 = torch.nn.functional.max_pool3d(bev_img_3d, kernel_size=3, stride=2, padding=1)

    # conv3: kernel_size=3, stride=2, padding=1 (all dimensions)
    x2 = torch.nn.functional.max_pool3d(x1, kernel_size=3, stride=2, padding=1)

    # conv4: kernel_size=3, stride=2, padding=(0, 1, 1) - no padding on depth (D), padding=1 on H and W
    # Note: max_pool3d expects padding as (padD, padH, padW)
    x3 = torch.nn.functional.max_pool3d(x2, kernel_size=3, stride=2, padding=(0, 1, 1))

    dims = [0, 2, 3, 4]  # sum over N, D, H, W, num_slices  -> leave channels
    p1 = x1.sum(dim=dims)
    p2 = x2.sum(dim=dims)
    p3 = x3.sum(dim=dims)
    p0 = p0_.sum(dim=1) # remember p0_ is (num_res, num_slices)
    return p0_, torch.stack((p0, p1, p2, p3))  # (num_res, num_slices), (num_layer, num_res)


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        self.resdiv_mask = self.model_cfg.get('RESDIV_MASK', [True] * len(self.res_divs))
        norm_method = self.model_cfg.get('NORM_METHOD', 'Batch')
        if norm_method == 'ResAwareBatch':
            norm_fn = partial(ResAwareBatchNorm1d, res_divs=self.res_divs, \
                    resdiv_mask=self.resdiv_mask, eps=1e-3, momentum=0.01)
        else:
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        # this is the first convolution where input size changes
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

        # Grouped with respect to having same input size
        self.num_layer_groups = 4

    def get_inds_dividers(self, tile_size_voxels):
        # numbers here are determined with respect to strides
        return [tile_size_voxels / float(s) for s in (2,4,8)]

    def sparse_outp_downscale_factor(self):
        return 8 # 3 convs with stride of 2

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        record_time = batch_dict.get('record_time', False)
        record_vcounts = batch_dict.get('record_int_vcounts', False)
        # int: intermediary
        record_int_vcoords = batch_dict.get('record_int_vcoords', False)
        record_int_indices = batch_dict.get('record_int_indices', False)

        if record_time:
            events=[torch.cuda.Event(enable_timing=True)]
            events[-1].record()
        if record_vcounts:
            num_voxels=[voxel_coords.size(0)]
        if record_int_vcoords:
            vcoords = []
            tile_size_voxels = batch_dict['tile_size_voxels']
            num_tiles = batch_dict['num_tiles']
        if record_int_indices:
            vinds = []

        resdiv = batch_dict.get('resolution_divider', 1)
        sparse_shape = [self.sparse_shape[0]] + [int(s/resdiv) for s in self.sparse_shape[1:]]
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2_0 = self.conv2[0][0](x_conv1)
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv2_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv2_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv2_0.indices[:, -1], tile_size_voxels // 2, \
                    rounding_mode='trunc')
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)[:num_tiles]
            vcoords.append(voxel_dist.unsqueeze(0))

        # since next to ops are not spconv op
        x_conv2_0 = x_conv2_0.replace_feature(self.conv2[0][1](x_conv2_0.features))
        x_conv2_0 = x_conv2_0.replace_feature(self.conv2[0][2](x_conv2_0.features))
        x_conv2_1 = self.conv2[1](x_conv2_0)
        x_conv2 = self.conv2[2](x_conv2_1)
        x_conv3_0 = self.conv3[0][0](x_conv2)
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv3_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv3_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv3_0.indices[:, -1], tile_size_voxels // 4, \
                    rounding_mode='trunc')
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)[:num_tiles]
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv3_0 = x_conv3_0.replace_feature(self.conv3[0][1](x_conv3_0.features))
        x_conv3_0 = x_conv3_0.replace_feature(self.conv3[0][2](x_conv3_0.features))
        x_conv3_1 = self.conv3[1](x_conv3_0)
        x_conv3 = self.conv3[2](x_conv3_1)
        x_conv4_0 = self.conv4[0][0](x_conv3)
        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv4_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv4_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv4_0.indices[:, -1], tile_size_voxels // 8, \
                    rounding_mode='trunc')
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)[:num_tiles]
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv4_0 = x_conv4_0.replace_feature(self.conv4[0][1](x_conv4_0.features))
        x_conv4_0 = x_conv4_0.replace_feature(self.conv4[0][2](x_conv4_0.features))
        x_conv4_1 = self.conv4[1](x_conv4_0)
        x_conv4 = self.conv4[2](x_conv4_1)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
            batch_dict['bb3d_layer_time_events'] = events
        if record_vcounts:
            batch_dict['bb3d_num_voxels'] = num_voxels
        if record_int_indices:
            batch_dict['bb3d_intermediary_vinds'] = vinds
        if record_int_vcoords:
            batch_dict['bb3d_intermediary_vcoords'] = vcoords

        return batch_dict
