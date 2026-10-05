"""
Loss functions for LoRATalk training stages (Landmark Generator and PAA Renderer).
"""

from losses.landmark_losses import (
    l1_loss_landmark,
    lmd_loss,
    velocity_loss_iplap,
    mouth_open_inner_torch,
    mouth_width_outer_torch,
    mouth_open_delta_loss,
    mouth_aperture_profile_torch,
    mouth_aperture_profile_loss,
)

from losses.renderer_losses import (
    mouth_l1_loss,
    lower_face_l1_loss,
    masked_l1,
    mouth_prior_mask,
    sketch_mouth_mask,
    sobel_edges,
    mouth_edge_l1,
    temporal_l1,
    mouth_temporal_l1,
    alignment_temporal_l1,
    warp_temporal_l1,
    sketch_aperture_proxy,
    image_mouth_activity,
    derivative_cosine_loss,
)

__all__ = [
    "l1_loss_landmark",
    "lmd_loss",
    "velocity_loss_iplap",
    "mouth_open_inner_torch",
    "mouth_width_outer_torch",
    "mouth_open_delta_loss",
    "mouth_aperture_profile_torch",
    "mouth_aperture_profile_loss",
    "mouth_l1_loss",
    "lower_face_l1_loss",
    "masked_l1",
    "mouth_prior_mask",
    "sketch_mouth_mask",
    "sobel_edges",
    "mouth_edge_l1",
    "temporal_l1",
    "mouth_temporal_l1",
    "alignment_temporal_l1",
    "warp_temporal_l1",
    "sketch_aperture_proxy",
    "image_mouth_activity",
    "derivative_cosine_loss",
]
