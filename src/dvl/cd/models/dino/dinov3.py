from typing import Optional, Sequence

from dvl.cd.models import DinoV2DPTChangeDetection
from dvl.cd.models.dino import DINOv3FeatureExtractor


class DinoV3DPTChangeDetection(DinoV2DPTChangeDetection):
    def build_encoder(
            self,
            in_channels: int,
            backbone: str,
            hook_indices: Optional[Sequence[int]],
            num_features: int,
            pretrained_backbone: bool,
            freeze_backbone: bool,
    ):
        self.encoder = DINOv3FeatureExtractor(
            model_name=backbone,
            pretrained=pretrained_backbone,
            in_channels=in_channels,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_backbone=freeze_backbone,
        )
