from typing import Sequence, Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn


def _make_scratch(in_shape: Sequence[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()

    out1 = out_shape
    out2 = out_shape
    out3 = out_shape
    out4 = out_shape

    if expand:
        out1 = out_shape
        out2 = out_shape * 2
        out3 = out_shape * 4
        out4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, use_bn: bool) -> None:
        super().__init__()
        self.use_bn = use_bn
        self.activation = activation

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        use_bn: bool,
        align_corners: bool = True,
        expand: bool = False,
        default_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.expand = expand
        self.default_size = default_size

        self.out_conv = nn.Conv2d(
            features,
            features // 2 if expand else features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.residual_1 = ResidualConvUnit(features, activation, use_bn)
        self.residual_2 = ResidualConvUnit(features, activation, use_bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        output = x
        if skip is not None:
            output = self.skip_add.add(output, self.residual_1(skip))

        output = self.residual_2(output)

        if size is not None:
            target = {"size": size}
        elif self.default_size is not None:
            target = {"size": self.default_size}
        else:
            target = {"scale_factor": 2.0}

        output = F.interpolate(output, mode='bilinear', align_corners=self.align_corners, **target)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features: int, use_bn: bool, size: Optional[Tuple[int, int]] = None) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=False),
        use_bn=use_bn,
        align_corners=True,
        expand=False,
        default_size=size,
    )


class DPTDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int | Sequence[int],
        features: int,
        out_channels: Sequence[int],
        use_bn: bool = False,
        use_cls_token: bool = True,
        first_resize_scale: int = 4,
    ) -> None:
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError('DPTDecoder expects four encoder stages.')

        self.use_cls_token = use_cls_token
        self.features = int(features)
        self.projection_channels = tuple(int(c) for c in out_channels)

        if isinstance(embed_dim, int):
            self.projects = nn.ModuleList([
                nn.Conv2d(embed_dim, c, kernel_size=1, stride=1, padding=0, bias=True)
                for c in self.projection_channels
            ])
        else:
            self.projects = nn.ModuleList([
                nn.Conv2d(e, c, kernel_size=1, stride=1, padding=0, bias=True)
                for e, c in zip(embed_dim, self.projection_channels)
            ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                self.projection_channels[0], self.projection_channels[0],
                kernel_size=first_resize_scale, stride=first_resize_scale, padding=0
            ),
            nn.ConvTranspose2d(self.projection_channels[1], self.projection_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(self.projection_channels[3], self.projection_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        if use_cls_token:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ])
        else:
            self.readout_projects = nn.ModuleList([nn.Identity() for _ in range(4)])

        self.scratch = _make_scratch(self.projection_channels, self.features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(self.features, use_bn)

    def _reshape_tokens(self, tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        return tokens.permute(0, 2, 1).reshape(batch, channels, patch_h, patch_w)

    def forward(
        self,
        features: Sequence[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
        patch_shape: Tuple[int, int] | List[Tuple[int, int]],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f'DPTDecoder expects four feature tensors, received {len(features)}.')

        processed: List[torch.Tensor] = []

        for idx, feat in enumerate(features):
            if isinstance(feat, (tuple, list)):
                patch_tokens = feat[0]
                cls_token = feat[1] if len(feat) > 1 else None
            else:
                patch_tokens = feat
                cls_token = None

            if self.use_cls_token and cls_token is not None:
                readout = cls_token.unsqueeze(1).expand_as(patch_tokens)
                patch_tokens = torch.cat((patch_tokens, readout), dim=-1)
                patch_tokens = self.readout_projects[idx](patch_tokens)

            if isinstance(patch_shape[0], int):
                patch_h, patch_w = patch_shape
            else:
                patch_h, patch_w = patch_shape[idx]

            spatial = self._reshape_tokens(patch_tokens, patch_h, patch_w)
            spatial = self.projects[idx](spatial)
            spatial = self.resize_layers[idx](spatial)
            processed.append(spatial)

        layer1, layer2, layer3, layer4 = processed

        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        path4 = self.scratch.refinenet4(layer4_rn, size=layer3_rn.shape[2:])
        path3 = self.scratch.refinenet3(path4, layer3_rn, size=layer2_rn.shape[2:])
        path2 = self.scratch.refinenet2(path3, layer2_rn, size=layer1_rn.shape[2:])
        path1 = self.scratch.refinenet1(path2, layer1_rn)

        return path1


class DPTHead(nn.Module):
    def __init__(self, in_channels: int, head_channels: int, use_bn: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(head_channels))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
