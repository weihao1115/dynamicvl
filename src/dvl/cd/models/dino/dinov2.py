from typing import Sequence, Optional, Dict

import torch
from torch import nn
import torch.nn.functional as F

from dvl.cd.models.decoders.dpt import DPTDecoder, DPTHead
from dvl.cd.models.dino import DINOv2FeatureExtractor, _default_stage_channels
from dvl.cd.models.losses import dice_loss_with_logits


class DinoV2DPTChangeDetection(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = 35,
            backbone: str = 'vit_base_patch14_dinov2.lvd142m',
            decoder_channels: Optional[int] = None,
            head_channels: Optional[int] = None,
            hook_indices: Optional[Sequence[int]] = None,
            num_features: int = 4,
            pretrained_backbone: bool = True,
            freeze_backbone: bool = False,
            decoder_stage_channels: Optional[Sequence[int]] = None,
            use_bn: bool = True,
    ):
        super(DinoV2DPTChangeDetection, self).__init__()
        self.build_encoder(
            in_channels=in_channels,
            backbone=backbone,
            hook_indices=hook_indices,
            num_features=num_features,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )

        stage_channels = tuple(
            int(c) for c in (
                decoder_stage_channels if decoder_stage_channels is not None
                else _default_stage_channels(self.encoder.embed_dim)
            )
        )

        self.decoder_channels = int(decoder_channels) if decoder_channels is not None else stage_channels[0]
        self.head_channels = int(head_channels) if head_channels is not None else self.decoder_channels

        self.decoder = DPTDecoder(
            embed_dim=self.encoder.embed_dim,
            features=self.decoder_channels,
            out_channels=stage_channels,
            use_bn=use_bn,
            use_cls_token=True,
        )
        self.head = DPTHead(self.decoder_channels, self.head_channels, use_bn=use_bn)
        self.classifier = nn.Conv2d(self.head_channels, num_classes, kernel_size=1)

        self.hook_indices = self.encoder.hook_indices
        self.num_features = len(self.hook_indices)


    def build_encoder(
            self,
            in_channels: int,
            backbone: str,
            hook_indices: Optional[Sequence[int]],
            num_features: int,
            pretrained_backbone: bool,
            freeze_backbone: bool,
    ):
        self.encoder = DINOv2FeatureExtractor(
            model_name=backbone,
            pretrained=pretrained_backbone,
            in_channels=in_channels,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_backbone=freeze_backbone,
        )


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features, patch_shape = self.encoder(x)
        return self.decoder(features, patch_shape)


    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict:
        input_size = pixel_values.shape[-2:]
        decoded = self.forward_features(pixel_values)
        refined = self.head(decoded)
        logits = self.classifier(refined)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)

        ret_dict = dict(loss=torch.tensor(0.0))
        if self.training and labels is not None:
            ce_loss = F.cross_entropy(input=logits, target=labels.long(), ignore_index=255)
            dice_loss = dice_loss_with_logits(y_pred=logits, y_true=labels, ignore_index=255)
            loss = ce_loss + dice_loss
            ret_dict.update(loss=loss, ce_loss=ce_loss, dice_loss=dice_loss)

        ret_dict.update(logits=logits)
        return ret_dict
