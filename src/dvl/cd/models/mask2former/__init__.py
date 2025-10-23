from typing import Optional, Dict

import torch
from transformers import Mask2FormerForUniversalSegmentation, AutoConfig, Mask2FormerImageProcessor


class Mask2FormerChangeDetection(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = 35,
            backbone: str = "facebook/mask2former-swin-base-ade-semantic",
            pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes

        if pretrained_backbone:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            mask2former_config = AutoConfig.from_pretrained(backbone, num_labels=num_classes)
            self.model = Mask2FormerForUniversalSegmentation(config=mask2former_config)

        self.image_processor = Mask2FormerImageProcessor.from_pretrained(backbone)
        self.image_processor.num_labels = num_classes


        patch_embed = getattr(self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings, 'projection', None)
        if patch_embed is None:
            raise ValueError('Backbone does not expose a patch embedding module.')

        if in_channels != getattr(patch_embed, 'in_channels', in_channels):
            self._adapt_input_channels(in_channels)

        id2label = {idx: f'class_{idx}' for idx in range(num_classes)}
        label2id = {name: idx for idx, name in id2label.items()}
        self.model.config.id2label = id2label
        self.model.config.label2id = label2id
        self.model.config.ignore_index = 255

    def _adapt_input_channels(self, in_channels: int) -> None:
        proj = getattr(self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings, 'projection', None)
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

        self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_proj

    def post_process_instance_segmentation_logits(self, outputs, target_size: tuple[int, int]):
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=target_size, mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        return segmentation


    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        """
        pixel_values: [B, C, H, W]
        labels:       [B, H, W] , 值∈{0..num_classes-1} 或 255(ignore)
        """
        B, _, H, W = pixel_values.shape
        device = pixel_values.device

        self.image_processor.size.update(height=H, width=W)
        inputs = self.image_processor(
            images=torch.ones((B, 3, H, W), device=device, dtype=pixel_values.dtype),  # just a placeholder
            segmentation_maps=labels,
            return_tensors='pt'
        ).to(device)
        mask_labels = [item.to(device) for item in inputs.mask_labels]
        class_labels = [item.to(device) for item in inputs.class_labels]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=inputs.pixel_mask,  # [B,H,W]，1/0
            mask_labels=mask_labels,  # List[Tensor[N_i,H_i,W_i]] (float)
            class_labels=class_labels,  # List[Tensor[N_i]]
        )
        if not self.training:
            logits = self.post_process_instance_segmentation_logits(
                outputs=outputs, target_size=(H, W),
            )
            outputs.logits = logits
            outputs["logits"] = logits
        return outputs
