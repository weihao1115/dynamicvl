from dvl.cd.models.dino.dinov2 import DinoV2DPTChangeDetection
from dvl.cd.models.dino.dinov3 import DinoV3DPTChangeDetection
from dvl.cd.models.mask2former import Mask2FormerChangeDetection

model_libs = dict(
    dinov2_dpt_cd=DinoV2DPTChangeDetection,
    dinov3_dpt_cd=DinoV3DPTChangeDetection,
    mask2former_cd=Mask2FormerChangeDetection,
)
