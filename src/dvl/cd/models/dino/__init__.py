import warnings
from typing import Sequence, Optional, Tuple, List

import torch
import math


_BACKBONE_ALIASES = {
    'vit_small_patch14_dinov2.lvd142m': 'dinov2_vits14',
    'vit_base_patch14_dinov2.lvd142m': 'dinov2_vitb14',
    'vit_large_patch14_dinov2.lvd142m': 'dinov2_vitl14',
    'vit_giant_patch14_dinov2.lvd142m': 'dinov2_vitg14',
}


def _default_stage_channels(embed_dim: int | Sequence[int]) -> Tuple[int, int, int, int]:
    if isinstance(embed_dim, Sequence):
        embed_dim = max(embed_dim)

    if embed_dim >= 1024:
        return 256, 512, 1024, 1024
    if embed_dim >= 768:
        return 256, 512, 768, 768
    if embed_dim >= 512:
        return 192, 384, 768, 768
    return 128, 256, 512, 512


class DINOv2FeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        model_name: str = 'vit_base_patch14_dinov2.lvd142m',
        pretrained: bool = True,
        in_channels: int = 3,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.build_dino(model_name=model_name, pretrained=pretrained)

        if getattr(self.model, 'chunked_blocks', False):
            warnings.warn('DINOv2 backbone is chunked; LoRA or fine-grained modifications may need adaptation.')

        patch_embed = getattr(self.model, 'patch_embed', None)
        if patch_embed is None:
            raise ValueError('Backbone does not expose a patch embedding module.')
        if hasattr(patch_embed, 'strict_img_size'):
            patch_embed.strict_img_size = False
        if hasattr(patch_embed, 'img_size'):
            patch_embed.img_size = None

        self.embed_dim = getattr(self.model, 'embed_dim', None)
        if self.embed_dim is None:
            raise ValueError('Unable to determine embedding dimension for backbone.')

        patch_size = getattr(patch_embed, 'patch_size', 14)
        if isinstance(patch_size, tuple):
            self.patch_size = int(patch_size[0])
        else:
            self.patch_size = int(patch_size)

        if in_channels != getattr(patch_embed.proj, 'in_channels', in_channels):
            self._adapt_input_channels(in_channels)

        depth = len(getattr(self.model, 'blocks', []))
        if depth == 0:
            raise ValueError('Backbone exposes no transformer blocks.')
        if hook_indices is None:
            if num_features <= 0:
                raise ValueError('num_features must be positive when hook_indices is not provided.')
            step = depth / float(num_features)
            indices = []
            for i in range(1, num_features + 1):
                idx = int(round(i * step) - 1)
                idx = max(0, min(depth - 1, idx))
                indices.append(idx)
            hook_indices = tuple(sorted(set(indices)))
        else:
            filtered = [int(idx) for idx in hook_indices if 0 <= int(idx) < depth]
            if not filtered:
                raise ValueError('Provided hook_indices are invalid for the selected backbone.')
            hook_indices = tuple(sorted(set(filtered)))

        self.hook_indices = hook_indices
        self.num_features = len(self.hook_indices)

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def build_dino(self, model_name: str, pretrained: bool):
        resolved = _BACKBONE_ALIASES.get(model_name, model_name)
        if not resolved.startswith('dinov2_'):
            raise ValueError(
                f'Unsupported backbone "{model_name}". Provide a DINOv2 identifier or a known alias.'
            )

        try:
            self.model = torch.hub.load('facebookresearch/dinov2', resolved, pretrained=pretrained)
        except Exception as exc:  # pragma: no cover - hub loading failures are environment specific
            raise RuntimeError(
                'Failed to load DINOv2 weights via torch.hub. Ensure torch>=1.13 and internet/connectivity '
                'or pre-download the weights with torch.hub.load_state_dict_from_url.'
            ) from exc

    def _adapt_input_channels(self, in_channels: int) -> None:
        proj = getattr(self.model.patch_embed, 'proj', None)
        if proj is None or not isinstance(proj, torch.nn.Conv2d):
            raise ValueError('Patch embedding projection is not a convolution; cannot adapt input channels.')
        if proj.in_channels == in_channels:
            return

        new_proj = torch.nn.Conv2d(
            in_channels,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=proj.bias is not None,
        )

        with torch.no_grad():
            weight = proj.weight
            current_in = weight.shape[1]
            if in_channels < current_in:
                weight = weight[:, :in_channels, :, :]
            elif in_channels > current_in:
                repeats = in_channels // current_in
                remainder = in_channels % current_in
                weight = weight.repeat(1, repeats, 1, 1)
                if remainder:
                    weight = torch.cat([weight, weight[:, :remainder, :, :]], dim=1)
                weight = weight * (current_in / float(in_channels))
            new_proj.weight.copy_(weight)
            if proj.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(proj.bias)

        self.model.patch_embed.proj = new_proj

    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')
        height, width = x.shape[-2:]

        outputs = self.model.get_intermediate_layers(
            x,
            n=self.hook_indices,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        sample = outputs[0][0] if isinstance(outputs[0], (tuple, list)) else outputs[0]
        num_tokens = sample.shape[1]

        patch_h = max(1, int(round(height / float(self.patch_size))))
        patch_w = max(1, num_tokens // patch_h)
        if patch_h * patch_w != num_tokens:
            patch_w = max(1, int(round(width / float(self.patch_size))))
            patch_h = max(1, num_tokens // patch_w)
        if patch_h * patch_w != num_tokens:
            patch_h = max(1, int(round(math.sqrt(num_tokens))))
            patch_w = max(1, num_tokens // patch_h)
        if patch_h * patch_w != num_tokens:
            raise RuntimeError('Failed to infer patch grid size from token count.')

        return list(outputs), (patch_h, patch_w)



class DINOv3FeatureExtractor(DINOv2FeatureExtractor):
    def build_dino(self, model_name: str, pretrained: bool):
        vit_types = ['dinov3_vitl16', 'dinov3_vit7b16']
        assert model_name in vit_types, model_name

        type2pth = {
            "dinov3_vitl16": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            "dinov3_vit7b16": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
        }

        self.model = torch.hub.load(
            repo_or_dir="./pretrained/dinov3",
            model=model_name,
            source='local',
            weights=f"./pretrained/dinov3/model_weights/{type2pth[model_name]}",
            pretrained=pretrained
        )
