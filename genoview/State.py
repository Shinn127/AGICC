from dataclasses import dataclass
from typing import Any


@dataclass
class SceneResources:
    ground_model: Any
    ground_position: Any
    geno_model: Any
    pose_model: Any
    geno_position: Any
    bind_pos: Any
    bind_rot: Any


@dataclass
class MotionResources:
    clip_resource: str
    clip_name: str
    bvh_animation: Any
    parents: Any
    global_positions: Any
    global_rotations: Any
    trajectory_sample_offsets: Any
    bvh_frame_time: float
    motion_variant: str = "original"
    mirrored: bool = False
    mirror_axis: str = "x"
    base_pose_source: Any = None
    bootstrap_contact_data: Any = None
    terrain_provider: Any = None
    contact_data: Any = None
    body_proxy_layout: Any = None
    terrain_model: Any = None
    terrain_height_grid: Any = None
    motion_root_trajectory: Any = None
    terrain_adapted_root_trajectory: Any = None
    pose_source: Any = None
    label_result: Any = None
    terrain_sample_normals: Any = None


@dataclass
class RenderResources:
    light_dir: Any
    shadow_light: Any
    shadow_map: Any
    shadow_inv_resolution: Any
    gbuffer: Any
    lighted: Any
    ssao_front: Any
    ssao_back: Any


@dataclass
class DebugState:
    label_module_ptr: Any
    draw_bone_transforms_ptr: Any
    draw_flat_ground_ptr: Any
    draw_terrain_mesh_ptr: Any
    draw_root_trajectory_ptr: Any
    draw_trajectory_directions_ptr: Any
    draw_trajectory_velocity_ptr: Any
    draw_contacts_ptr: Any
    draw_bootstrap_contacts_ptr: Any
    draw_terrain_samples_ptr: Any
    draw_terrain_normals_ptr: Any
    draw_body_proxy_ptr: Any
    draw_terrain_penetration_ptr: Any
    draw_reconstructed_pose_ptr: Any
    draw_pose_model_local_ptr: Any
    draw_reconstruction_error_ptr: Any
    integrate_root_motion_ptr: Any
    selected_timeline_mode: str
    local_debug_origin: Any
    pose_model_color: Any
    selected_action_label: str
    transition_width: int
    annotation_status: str
    playback: Any
    module_dropdown_open: bool = False
    clip_dropdown_open: bool = False
    clip_dropdown_scroll: int = 0


@dataclass
class AsyncProgress:
    active: bool = False
    title: str = ""
    detail: str = ""
    progress: float = 0.0
    request_id: int = 0
    indeterminate: bool = False


@dataclass(frozen=True)
class AsyncProgressEvent:
    request_id: int
    title: str = ""
    detail: str = ""
    progress: float = 0.0
    active: bool = True
    indeterminate: bool = False


@dataclass
class PendingClipLoad:
    request_id: int
    clip_index: int
    clip_resource: str
    motion_variant: str
    future: Any


@dataclass
class PendingFeatureLoad:
    request_id: int
    feature_id: str
    clip_resource: str
    motion_variant: str
    future: Any


@dataclass
class FeatureLoadResult:
    feature_id: str
    clip_resource: str
    motion_variant: str
    motion: MotionResources


@dataclass
class AppState:
    screen_width: int
    screen_height: int
    shaders: Any
    scene: SceneResources
    motion: MotionResources
    clip_resources: list[str]
    clip_index: int
    render: RenderResources
    debug: DebugState
    features: Any
    async_progress: AsyncProgress
    async_progress_events: Any
    clip_load_executor: Any
    pending_clip_load: Any
    feature_load_executor: Any
    pending_feature_loads: Any
    camera: Any
    mirror_enabled: bool = False
    mirror_axis: str = "x"


@dataclass
class ShadowPassState:
    view_proj: Any
    clip_near_ptr: Any
    clip_far_ptr: Any


@dataclass
class CameraPassState:
    view: Any
    proj: Any
    inv_proj: Any
    inv_view_proj: Any
    clip_near_ptr: Any
    clip_far_ptr: Any


@dataclass
class FrameState:
    animation_frame: int
    frame_count: int
    clip_name: str
    clip_prior: str
    current_auto_label: str
    current_final_label: str
    auto_labels: Any
    auto_segments: Any
    final_labels: Any
    final_segments: Any
    soft_weights: Any
    hip_position: Any
    root_trajectory: Any = None
    terrain_root_trajectory_display: Any = None
    local_pose: Any = None
    local_pose_positions_offset: Any = None
    reconstructed_pose_world: Any = None
    pose_comparison_positions: Any = None
    pose_error_label: bytes = b"Recon Err"
    pose_position_error_mean: float = 0.0
    pose_position_error_max: float = 0.0
    bootstrap_frame_contacts: Any = None
    bootstrap_bvh_contact_positions: Any = None
    bootstrap_pose_contact_positions: Any = None
    frame_contacts: Any = None
    bvh_contact_positions: Any = None
    pose_contact_positions: Any = None
    terrain_height_at_focus: Any = None
    terrain_normal_at_focus: Any = None
    body_proxy_positions: Any = None
    body_proxy_radii: Any = None
    penetration_frame: Any = None
    penetration_count: int = 0
    max_penetration_depth: float = 0.0
