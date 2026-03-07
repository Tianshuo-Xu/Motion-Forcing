#!/usr/bin/env python
"""
Motion-Forcing Inference Demo (Gradio)

Point -> Shape -> Appearance Video Generation

Pipeline:
  1. Upload an image
  2. YOLO extracts segmentation map
  3. Click on an object to select it, click again to define motion arrow
  4. Arrow creates a circle video (min inscribed circles moving over time)
  5. Circle video is VAE-encoded as latent_point condition
  6. Model generates segmentation video, then RGB video (two-stage denoising)

Supported checkpoint formats:
  - HuggingFace model ID: e.g. 'TSXu/forcing_depth' (auto-downloaded/cached)
  - DeepSpeed:  checkpoint-XXXX/pytorch_model/mp_rank_00_model_states.pt
  - Safetensors: checkpoint dir containing *.safetensors (including sharded)
  - Single file: any .pt / .bin / .safetensors file
  - HuggingFace: directory with config.json + model weights

Usage:
    # Load from HuggingFace Hub (recommended)
    python gradio_demo.py \\
        --model_path THUDM/CogVideoX-5b-I2V \\
        --transformer_ckpt TSXu/MotionForcing_driving

    # Load with DeepSpeed checkpoint
    python gradio_demo.py \\
        --model_path THUDM/CogVideoX-5b-I2V \\
        --load_pretrained_weight /path/to/checkpoint-XXXX/pytorch_model/mp_rank_00_model_states.pt

    # Load with LoRA
    python gradio_demo.py \\
        --model_path THUDM/CogVideoX-5b-I2V \\
        --lora_path /path/to/lora_weights
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob as glob_module
import tempfile

import math

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import CubicSpline as _CubicSpline
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, T5EncoderModel

from models.cogvideox_transformer_MD import CogVideoXTransformer3DModel
from models.pipeline import CogVideoXImageToVideoPipeline

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

_VDA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Video-Depth-Anything'))
if _VDA_ROOT not in sys.path:
    sys.path.insert(0, _VDA_ROOT)
for _pkg in ['video_depth_anything', 'video_depth_anything/motion_module', 'video_depth_anything/util', 'utils']:
    _init = os.path.join(_VDA_ROOT, _pkg, '__init__.py')
    if not os.path.exists(_init):
        open(_init, 'a').close()
from video_depth_anything.video_depth import VideoDepthAnything



# ─── Constants ───────────────────────────────────────────────────────────────

PALETTE = np.random.RandomState(42).randint(0, 255, size=(1000, 3)).astype(np.uint8)
TARGET_HEIGHT = 480
TARGET_WIDTH = 720
DEFAULT_PROMPT = (
    "a realistic driving scenario with high visual quality, "
    "the overall scene is moving forward."
)


# ─── Checkpoint Loading Utilities ─────────────────────────────────────────────

def _load_state_dict_from_file(file_path):
    """
    Load a state dict from a single file (.safetensors / .pt / .bin).
    For DeepSpeed checkpoints the state dict is stored under the 'module' key.
    """
    print(f"  Loading weights from {file_path} ...")
    if file_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(file_path, device="cpu")

    state_dict = torch.load(file_path, map_location="cpu")
    if isinstance(state_dict, dict) and "module" in state_dict:
        state_dict = state_dict["module"]
    return state_dict


def _load_sharded_safetensors(ckpt_dir):
    """
    Load a sharded checkpoint (safetensors or .bin) indexed by model.safetensors.index.json.
    """
    import json as json_module

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        # Also try diffusers naming
        index_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors.index.json")

    with open(index_path) as f:
        index = json_module.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    print(f"  Loading {len(shard_files)} shards from {ckpt_dir} ...")

    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(ckpt_dir, shard_file)
        print(f"    {shard_file} ...", end=" ", flush=True)
        state_dict.update(_load_state_dict_from_file(shard_path))
        print("done")

    print(f"  Loaded {len(state_dict)} keys total.")
    return state_dict


def _resolve_hf_model_id(path):
    """
    If *path* looks like a HuggingFace model ID (e.g. 'TSXu/forcing_depth'),
    resolve it to a local snapshot directory via huggingface_hub.
    Returns the resolved local path (downloading if needed), or *path* unchanged.
    """
    if os.path.exists(path):
        return path

    # Heuristic: HF model IDs contain '/' but are not existing filesystem paths
    if "/" in path and not path.startswith("/"):
        try:
            from huggingface_hub import snapshot_download
            print(f"  Resolving HuggingFace model ID '{path}' ...")
            local_dir = snapshot_download(path)
            print(f"  Resolved to: {local_dir}")
            return local_dir
        except Exception as e:
            print(f"  Warning: failed to resolve '{path}' as HF model ID: {e}")

    return path


def _find_and_load_checkpoint(path):
    """
    Auto-detect checkpoint format at *path* (file or directory) and return state_dict.

    Supports:
      - HuggingFace model ID: e.g. 'TSXu/forcing_depth' (resolved via huggingface_hub)
      - Single file: .pt / .bin / .safetensors
      - Directory with sharded safetensors (model.safetensors.index.json)
      - Directory with DeepSpeed checkpoint (pytorch_model/mp_rank_00_model_states.pt)
      - Directory with single weight file
    """
    path = _resolve_hf_model_id(path)

    if os.path.isfile(path):
        return _load_state_dict_from_file(path)

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    # Sharded safetensors (index.json + multiple shard files)
    for index_name in ["model.safetensors.index.json",
                       "diffusion_pytorch_model.safetensors.index.json"]:
        if os.path.isfile(os.path.join(path, index_name)):
            return _load_sharded_safetensors(path)

    # DeepSpeed format
    ds_path = os.path.join(path, "pytorch_model", "mp_rank_00_model_states.pt")
    if os.path.isfile(ds_path):
        return _load_state_dict_from_file(ds_path)

    # Single safetensors
    for name in ["model.safetensors", "diffusion_pytorch_model.safetensors"]:
        p = os.path.join(path, name)
        if os.path.isfile(p):
            return _load_state_dict_from_file(p)

    # Single pytorch file
    for name in ["pytorch_model.bin", "model.bin", "diffusion_pytorch_model.bin"]:
        p = os.path.join(path, name)
        if os.path.isfile(p):
            return _load_state_dict_from_file(p)

    # Glob for any weight file
    for ext in ["*.safetensors", "*.pt", "*.bin"]:
        found = sorted(glob_module.glob(os.path.join(path, ext)))
        if found:
            return _load_state_dict_from_file(found[0])

    raise FileNotFoundError(
        f"Cannot find loadable checkpoint in {path}. "
        "Expected .pt / .safetensors / .bin files or sharded safetensors with index.json."
    )


def _apply_state_dict(model, state_dict):
    """Load state dict with helpful mismatch reporting (always non-strict)."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [load_state_dict] missing keys  : {len(missing)}")
        print(f"  [load_state_dict] unexpected keys: {len(unexpected)}")
        if missing:
            print(f"    first 5 missing : {missing[:5]}")
        if unexpected:
            print(f"    first 5 unexpected: {unexpected[:5]}")
    else:
        print("  [load_state_dict] all keys matched.")


# ─── Model Loading ───────────────────────────────────────────────────────────

def _load_pipeline_components(model_path, dtype):
    """
    Load pipeline components (tokenizer, text_encoder, vae, scheduler)
    individually from the HuggingFace model, WITHOUT downloading
    the large transformer weights.
    """
    from transformers import T5Tokenizer

    # print("  Loading tokenizer ...")
    # tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
    # print("  Loading text_encoder ...")
    # text_encoder = T5EncoderModel.from_pretrained(
    #     model_path, subfolder="text_encoder", torch_dtype=dtype
    # )
    print("  Skipping tokenizer & text_encoder (using cached prompt embeddings) ...")
    tokenizer = None
    text_encoder = None

    print("  Loading VAE ...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    )

    print("  Loading scheduler ...")
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, scheduler


def _build_transformer_from_config(model_path, dtype, checkpoint_sd=None):
    """
    Create a CogVideoXTransformer3DModel from the config at *model_path*
    (only downloads the tiny config.json, not the weights).

    If *checkpoint_sd* is provided, auto-detects CogVideoX version (1.0 vs 1.5)
    from weight shapes and patches the config accordingly.
    """
    config_dict = CogVideoXTransformer3DModel.load_config(
        model_path, subfolder="transformer"
    )

    if checkpoint_sd is not None:
        # Auto-detect CogVideoX 1.5 from patch_embed weight shape
        pe_key = "patch_embed.proj.weight"
        if pe_key in checkpoint_sd and checkpoint_sd[pe_key].ndim == 2:
            # CogVideoX 1.5 uses nn.Linear -> 2-D weight [embed_dim, in*p*p*p_t]
            # CogVideoX 1.0 uses nn.Conv2d  -> 4-D weight [embed_dim, in, p, p]
            config_dict["patch_size_t"] = 2
            print("  Auto-detected CogVideoX 1.5 architecture (patch_size_t=2)")

        # Auto-detect ofs_embed_dim
        ofs_key = "ofs_embedding.linear_1.weight"
        if ofs_key in checkpoint_sd:
            ofs_dim = checkpoint_sd[ofs_key].shape[1]
            config_dict["ofs_embed_dim"] = ofs_dim
            print(f"  Auto-detected ofs_embed_dim={ofs_dim}")

        # Auto-detect out_channels from proj_out
        po_key = "proj_out.bias"
        if po_key in checkpoint_sd:
            output_dim = checkpoint_sd[po_key].shape[0]
            patch_size = config_dict.get("patch_size", 2)
            patch_size_t = config_dict.get("patch_size_t", None) or 1
            out_ch = output_dim // (patch_size * patch_size * patch_size_t)
            if out_ch != config_dict.get("out_channels", 16):
                config_dict["out_channels"] = out_ch
                print(f"  Auto-detected out_channels={out_ch}")

        # Auto-detect use_learned_positional_embeddings
        if "patch_embed.pos_embedding" not in checkpoint_sd:
            config_dict["use_learned_positional_embeddings"] = False
            print("  Auto-detected use_learned_positional_embeddings=False")

    transformer = CogVideoXTransformer3DModel.from_config(config_dict)
    transformer = transformer.to(dtype=dtype)
    return transformer


def load_models(args):
    """
    Load the CogVideoX pipeline and YOLO model.

    When --transformer_ckpt or --load_pretrained_weight is given, pipeline
    components are loaded individually so the large base-transformer download
    is skipped entirely.
    """
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    has_local_ckpt = args.load_pretrained_weight or args.transformer_ckpt

    # 1. YOLO
    print(f"[1/4] Loading YOLO from {args.yolo_path} ...")
    yolo_model = YOLO(args.yolo_path)

    if has_local_ckpt:
        # ── Fast path: load components individually, skip transformer download ──

        # Load checkpoint first so we can auto-detect architecture
        ckpt_path = args.load_pretrained_weight or args.transformer_ckpt
        print(f"[2/4] Loading checkpoint from {ckpt_path} ...")
        sd = _find_and_load_checkpoint(ckpt_path)

        print(f"[3/4] Loading pipeline components from {args.model_path} "
              "(skipping transformer download) ...")
        tokenizer, text_encoder, vae, scheduler = _load_pipeline_components(
            args.model_path, dtype
        )

        print("  Building transformer (auto-detecting architecture from checkpoint) ...")
        transformer = _build_transformer_from_config(
            args.model_path, dtype, checkpoint_sd=sd
        )
        transformer.init_all_weights()

        # Zero-init layers that don't exist in the checkpoint so they act as no-ops.
        # ins_norm* are already zero-init, but flow_proj has random Kaiming init by default.
        for name, param in transformer.named_parameters():
            if name not in sd:
                torch.nn.init.zeros_(param)

        _apply_state_dict(transformer, sd)
        del sd

        # Ensure all params (including newly created ones from init_all_weights)
        # are in the correct dtype
        transformer = transformer.to(dtype=dtype)

        # Assemble pipeline
        print("[4/4] Assembling pipeline ...")
        scheduler = CogVideoXDPMScheduler.from_config(
            scheduler.config, timestep_spacing="trailing"
        )
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    else:
        # ── Normal path: load full pipeline from HF (downloads everything) ──
        print(f"[2/4] Loading full CogVideoX pipeline from {args.model_path} ...")
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            args.model_path, torch_dtype=dtype
        )
        print("[3/4] Initialising motion-specific layers ...")
        pipe.transformer.init_all_weights()
        print("[4/4] No checkpoint provided; using base model with zero-init motion layers.")
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

    # Optional LoRA
    if args.lora_path:
        print(f"  Loading LoRA weights from {args.lora_path} ...")
        pipe.load_lora_weights(
            args.lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="motion_lora",
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=0.5)

    # Load Video Depth Anything for high-quality depth
    print("[5/6] Loading Video Depth Anything ...")
    vda_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    depth_model = VideoDepthAnything(**vda_config)
    vda_ckpt = os.path.join(
        os.path.dirname(__file__),
        'Video-Depth-Anything', 'checkpoints', 'video_depth_anything_vits.pth'
    )
    if os.path.exists(vda_ckpt):
        depth_model.load_state_dict(torch.load(vda_ckpt, map_location='cpu'), strict=True)
    depth_model.eval()

    # Load VGGT for camera pose estimation
    print("[6/6] Loading VGGT model ...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
    vggt_model.float()
    vggt_model.eval()

    # Memory optimisations
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    depth_model.to(args.device)
    vggt_model.to(args.device)

    print("All models loaded successfully.")
    return pipe, yolo_model, depth_model, vggt_model


# ─── YOLO Segmentation ──────────────────────────────────────────────────────

def run_yolo_segmentation(yolo_model, image_np):
    """
    Run YOLO instance segmentation on a single image.

    Args:
        image_np: [H, W, 3] uint8 numpy array

    Returns:
        seg_overlay: [H, W, 3] uint8  coloured mask overlay
        masks_data:  list of (binary_mask_HW, track_id, rgb_color)
    """
    H, W = image_np.shape[:2]

    # Reset tracker state so IDs start fresh
    if yolo_model.predictor is not None:
        for tracker in yolo_model.predictor.trackers:
            tracker.reset()
            tracker.reset_id()

    result = yolo_model.track(
        source=image_np, persist=True, verbose=False,
        conf=0.2, iou=0.2, tracker="bytetrack.yaml",
    )[0]

    seg_overlay = np.zeros((H, W, 3), dtype=np.uint8)
    masks_data = []

    if result.masks is not None:
        ids = (
            result.boxes.id.cpu().numpy()
            if result.boxes.id is not None
            else np.arange(len(result.masks.data))
        )
        for i, mask_tensor in enumerate(result.masks.data):
            binary = mask_tensor.cpu().numpy().astype(np.uint8)
            binary = cv2.resize(binary, (W, H), interpolation=cv2.INTER_NEAREST)
            track_id = int(ids[i]) if i < len(ids) else i
            color = PALETTE[track_id % len(PALETTE)].tolist()
            masks_data.append((binary, track_id, color))

            coloured = np.zeros_like(seg_overlay)
            for c in range(3):
                coloured[:, :, c] = binary * color[c]
            seg_overlay = cv2.addWeighted(seg_overlay, 1, coloured, 0.5, 0)

    return seg_overlay, masks_data


def overlay_with_highlight(image_np, seg_overlay, masks_data,
                           selected_idx=None, paths_dict=None):
    """Blend image + seg overlay, draw all polyline paths, highlight selected mask."""
    blended = cv2.addWeighted(image_np, 0.6, seg_overlay, 0.4, 0)

    if paths_dict:
        circles = compute_object_circles(masks_data)
        for idx, disps in paths_dict.items():
            if not (0 <= idx < len(masks_data)):
                continue
            mask_bin, _, _ = masks_data[idx]
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(blended, contours, -1, (0, 200, 255), 2)
            cx, cy, _, _ = circles[idx]
            waypoints = [(cx, cy)] + [(cx + dx, cy + dy) for dx, dy in disps]
            cv2.circle(blended, (cx, cy), 6, (0, 255, 0), -1)
            for j in range(len(waypoints) - 1):
                p1 = (int(waypoints[j][0]), int(waypoints[j][1]))
                p2 = (int(waypoints[j + 1][0]), int(waypoints[j + 1][1]))
                cv2.arrowedLine(blended, p1, p2,
                                (255, 0, 0), 3, tipLength=0.25)

    if selected_idx is not None and 0 <= selected_idx < len(masks_data):
        mask_bin, _, _ = masks_data[selected_idx]
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blended, contours, -1, (255, 255, 0), 3)
        hl = np.zeros_like(blended)
        hl[mask_bin > 0] = [255, 255, 0]
        blended = cv2.addWeighted(blended, 0.85, hl, 0.15, 0)

    return blended


def find_mask_at_point(masks_data, x, y):
    """Return the index of the mask that contains pixel (x, y), or None."""
    for i, (mask_bin, _, _) in enumerate(masks_data):
        if 0 <= int(y) < mask_bin.shape[0] and 0 <= int(x) < mask_bin.shape[1]:
            if mask_bin[int(y), int(x)] > 0:
                return i
    return None


# ─── Circle Video Helpers ────────────────────────────────────────────────────

def compute_object_circles(masks_data):
    """
    Compute (center_x, center_y, radius, color) for each detected object
    from its binary mask bounding box.
    """
    if not masks_data:
        return []
    circles = []
    for binary_mask, track_id, color in masks_data:
        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0:
            circles.append((0, 0, 5, color))
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        bbox_w = x1 - x0
        bbox_h = y1 - y0
        radius = max(int(min(bbox_w, bbox_h) / 2), 5)
        circles.append((cx, cy, radius, color))
    return circles


def perspective_radius(base_radius, x, y, H, W,
                       min_scale=0.3, horiz_weight=0.15):
    """
    Scale radius to simulate driving-scene perspective:
      - vertical: bottom of frame (near camera) → large, top (far) → small
      - horizontal: centre of frame → slightly larger than edges
    """
    vert_scale = min_scale + (1.0 - min_scale) * (y / H)
    horiz_factor = 1.0 - horiz_weight * abs(x - W / 2.0) / (W / 2.0)
    scale = vert_scale * horiz_factor
    return max(int(base_radius * scale), 3)


def smooth_path_2d(waypoints, num_samples=200):
    """Resample 2D waypoints along a smooth cubic spline curve.

    Returns a densely-sampled list of (x, y) tuples that can be fed into
    ``interpolate_polyline`` for per-frame lookup.  With ≤ 2 control points
    the input is returned unchanged (linear is already smooth).
    """
    if len(waypoints) <= 2:
        return list(waypoints)
    dists = [0.0]
    for i in range(1, len(waypoints)):
        d = math.sqrt((waypoints[i][0] - waypoints[i - 1][0]) ** 2
                      + (waypoints[i][1] - waypoints[i - 1][1]) ** 2)
        dists.append(dists[-1] + d)
    total = dists[-1]
    if total == 0:
        return list(waypoints)
    s = np.array(dists) / total
    xs = np.array([w[0] for w in waypoints])
    ys = np.array([w[1] for w in waypoints])
    cs_x = _CubicSpline(s, xs)
    cs_y = _CubicSpline(s, ys)
    t_new = np.linspace(0, 1, num_samples)
    return [(float(cs_x(t)), float(cs_y(t))) for t in t_new]


def smooth_path_3d(waypoints_3d, num_samples=200):
    """Resample 3D waypoints (numpy arrays) along a smooth cubic spline curve."""
    if len(waypoints_3d) <= 2:
        return list(waypoints_3d)
    pts = np.array([np.asarray(w) for w in waypoints_3d])
    dists = [0.0]
    for i in range(1, len(pts)):
        dists.append(dists[-1] + np.linalg.norm(pts[i] - pts[i - 1]))
    total = dists[-1]
    if total == 0:
        return list(waypoints_3d)
    s = np.array(dists) / total
    splines = [_CubicSpline(s, pts[:, d]) for d in range(3)]
    t_new = np.linspace(0, 1, num_samples)
    return [np.array([float(sp(t)) for sp in splines]) for t in t_new]


def interpolate_polyline(waypoints, t):
    """Return (x, y) at fraction *t* ∈ [0, 1] along the polyline *waypoints*."""
    if len(waypoints) < 2:
        return waypoints[0]
    dists = [0.0]
    for i in range(1, len(waypoints)):
        d = math.sqrt((waypoints[i][0] - waypoints[i - 1][0]) ** 2
                      + (waypoints[i][1] - waypoints[i - 1][1]) ** 2)
        dists.append(dists[-1] + d)
    total = dists[-1]
    if total == 0:
        return waypoints[0]
    target = t * total
    for i in range(1, len(waypoints)):
        if dists[i] >= target:
            seg_len = dists[i] - dists[i - 1]
            if seg_len == 0:
                return waypoints[i - 1]
            local_t = (target - dists[i - 1]) / seg_len
            x = waypoints[i - 1][0] + local_t * (waypoints[i][0] - waypoints[i - 1][0])
            y = waypoints[i - 1][1] + local_t * (waypoints[i][1] - waypoints[i - 1][1])
            return (x, y)
    return waypoints[-1]


def interpolate_driving_path(waypoints, t):
    """
    Interpolate along a driving-style path at fraction *t* ∈ [0, 1].

    Forward (second element, screen-Y / tz) progresses linearly while
    lateral (first element, screen-X / tx) follows a smoothstep S-curve,
    both advancing simultaneously — like a real vehicle lane change where
    you steer gradually while maintaining forward speed.
    """
    if len(waypoints) < 2:
        return waypoints[0]

    seg_lens = []
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        seg_lens.append(abs(dy) + abs(dx))
    total = sum(seg_lens)
    if total == 0:
        return waypoints[0]

    target = t * total
    acc = 0.0
    for i, seg_len in enumerate(seg_lens, start=1):
        if acc + seg_len >= target:
            x0, y0 = waypoints[i - 1]
            x1, y1 = waypoints[i]
            dx = x1 - x0
            dy = y1 - y0
            if seg_len == 0:
                return (x0, y0)
            local_t = (target - acc) / seg_len

            s = 3.0 * local_t ** 2 - 2.0 * local_t ** 3
            return (x0 + s * dx, y0 + local_t * dy)
        acc += seg_len

    return waypoints[-1]


def interpolate_polyline_3d(waypoints_3d, t):
    """Return 3D point at fraction *t* ∈ [0, 1] along the 3D polyline *waypoints_3d*."""
    if len(waypoints_3d) < 2:
        return waypoints_3d[0].copy()
    dists = [0.0]
    for i in range(1, len(waypoints_3d)):
        d = np.linalg.norm(waypoints_3d[i] - waypoints_3d[i - 1])
        dists.append(dists[-1] + d)
    total = dists[-1]
    if total == 0:
        return waypoints_3d[0].copy()
    target = t * total
    for i in range(1, len(waypoints_3d)):
        if dists[i] >= target:
            seg_len = dists[i] - dists[i - 1]
            if seg_len == 0:
                return waypoints_3d[i - 1].copy()
            local_t = (target - dists[i - 1]) / seg_len
            return waypoints_3d[i - 1] + local_t * (waypoints_3d[i] - waypoints_3d[i - 1])
    return waypoints_3d[-1].copy()


def build_raw_seg_frame(masks_data, H, W):
    """
    Create [H, W, 3] uint8 seg image matching training format
    (direct color assignment, no alpha blending).
    """
    seg = np.zeros((H, W, 3), dtype=np.uint8)
    for binary_mask, track_id, color in masks_data:
        for c in range(3):
            seg[:, :, c] = np.where(binary_mask > 0, color[c], seg[:, :, c])
    return seg


def build_circle_video_frames(circles, paths_dict, num_frames, H, W,
                              depth_np=None):
    """
    Create [H, W, 3] circle frames for objects in *paths_dict* only.
    Radius scales by depth ratio (original_depth / current_depth) when
    depth_np is provided; falls back to pixel-position heuristic otherwise.
    """
    def _sample_depth(px, py):
        py_c = max(0, min(int(round(py)), H - 1))
        px_c = max(0, min(int(round(px)), W - 1))
        d = float(depth_np[py_c, px_c])
        if d <= 0:
            valid = depth_np[depth_np > 0]
            d = float(valid.mean()) if valid.size > 0 else 1.0
        return d

    frames = []
    for t in range(num_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frac = t / max(num_frames - 1, 1)
        for i, (cx, cy, base_r, color) in enumerate(circles):
            if i not in paths_dict:
                continue
            waypoints = [(cx, cy)] + [(cx + dx, cy + dy)
                                      for dx, dy in paths_dict[i]]
            waypoints = smooth_path_2d(waypoints)
            pos = interpolate_polyline(waypoints, frac)
            draw_cx, draw_cy = int(pos[0]), int(pos[1])
            if depth_np is not None:
                d_0 = _sample_depth(cx, cy + base_r)
                d_t = _sample_depth(draw_cx, draw_cy + base_r)
                r = max(int(base_r * d_0 / d_t), 3)
            else:
                r = perspective_radius(base_r, draw_cx, draw_cy, H, W)
            cv2.circle(frame, (draw_cx, draw_cy), r, color, -1)
        frames.append(frame)
    return frames


def build_circle_video_frames_with_camera(
    circles, paths_dict, num_frames, H, W,
    raw_depth, intrinsics, extrinsics,
    item_ease=1.4,
):
    """
    Build circle video frames with item trajectories treated as world-space motion,
    reprojected through each frame's (moving) camera.

    Pixel-space waypoints are unprojected to 3D via depth + frame-0 camera, then
    for each frame the 3D position is reprojected through the current camera pose.
    This makes item motion and camera motion orthogonal: e.g. if ego and another
    car both move forward at the same speed, their relative image position is
    unchanged; if ego turns left, the other car drifts rightward on screen.
    """
    depth_np = raw_depth.squeeze().cpu().float().numpy()  # [H, W]
    K = intrinsics[0, 0].cpu().double().numpy()           # [3, 3]
    K_inv = np.linalg.inv(K)

    R_0 = extrinsics[0, 0, :3, :3].cpu().double().numpy()
    t_0 = extrinsics[0, 0, :3, 3].cpu().double().numpy()

    def _get_depth(px, py):
        py_c = max(0, min(int(round(py)), H - 1))
        px_c = max(0, min(int(round(px)), W - 1))
        d = float(depth_np[py_c, px_c])
        if d <= 0:
            valid = depth_np[depth_np > 0]
            d = float(valid.mean()) if valid.size > 0 else 1.0
        return d

    def _unproject_to_world(px, py, depth):
        p_pix = np.array([px, py, 1.0], dtype=np.float64)
        p_cam = K_inv @ p_pix * depth
        return R_0.T @ (p_cam - t_0)

    world_trajectories = {}
    bottom_trajectories = {}
    d0_bottoms = {}
    for i, (cx, cy, base_r, color) in enumerate(circles):
        if i not in paths_dict:
            continue
        d_center = _get_depth(cx, cy)
        d_bottom = _get_depth(cx, cy + base_r)
        d0_bottoms[i] = d_bottom

        center_wps = [_unproject_to_world(cx, cy, d_center)]
        bottom_wps = [_unproject_to_world(cx, cy + base_r, d_bottom)]
        for dx, dy in paths_dict[i]:
            wp_x, wp_y = cx + dx, cy + dy
            d_wp = _get_depth(wp_x, wp_y)
            d_wp_bot = _get_depth(wp_x, wp_y + base_r)
            center_wps.append(_unproject_to_world(wp_x, wp_y, d_wp))
            bottom_wps.append(_unproject_to_world(wp_x, wp_y + base_r, d_wp_bot))
        world_trajectories[i] = smooth_path_3d(center_wps)
        bottom_trajectories[i] = smooth_path_3d(bottom_wps)

    exited = set()
    frames = []
    for t in range(num_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frac_linear = t / max(num_frames - 1, 1)
        frac = 1.0 - (1.0 - frac_linear) ** item_ease

        R_t = extrinsics[0, t, :3, :3].cpu().double().numpy()
        t_t = extrinsics[0, t, :3, 3].cpu().double().numpy()

        for i, (cx, cy, base_r, color) in enumerate(circles):
            if i not in world_trajectories or i in exited:
                continue
            p_world = interpolate_polyline_3d(world_trajectories[i], frac)

            p_cam = R_t @ p_world + t_t
            if p_cam[2] <= 0:
                exited.add(i)
                continue
            p_pix = K @ p_cam
            draw_cx = int(round(p_pix[0] / p_pix[2]))
            draw_cy = int(round(p_pix[1] / p_pix[2]))

            p_bot_world = interpolate_polyline_3d(bottom_trajectories[i], frac)
            p_bot_cam = R_t @ p_bot_world + t_t
            d_t_bot = max(p_bot_cam[2], 1e-6)
            r = max(int(base_r * d0_bottoms[i] / d_t_bot), 3)

            clipped_x = max(0, min(draw_cx + r, W)) - max(0, min(draw_cx - r, W))
            clipped_y = max(0, min(draw_cy + r, H)) - max(0, min(draw_cy - r, H))
            visible_ratio = (clipped_x * clipped_y) / max((2 * r) ** 2, 1)
            if visible_ratio < 0.7:
                if t > 0:
                    exited.add(i)
                continue

            cv2.circle(frame, (draw_cx, draw_cy), r, color, -1)

        frames.append(frame)
    return frames


def render_circle_preview(circles, paths_dict, H, W):
    """Render a static preview: polyline path + perspective-scaled circles."""
    preview = np.zeros((H, W, 3), dtype=np.uint8)
    for i, (cx, cy, base_r, color) in enumerate(circles):
        if i not in paths_dict:
            continue
        waypoints = [(cx, cy)] + [(cx + dx, cy + dy)
                                  for dx, dy in paths_dict[i]]
        r = perspective_radius(base_r, cx, cy, H, W)
        cv2.circle(preview, (cx, cy), r, color, -1)
        for j in range(len(waypoints) - 1):
            p1 = (int(waypoints[j][0]), int(waypoints[j][1]))
            p2 = (int(waypoints[j + 1][0]), int(waypoints[j + 1][1]))
            cv2.line(preview, p1, p2, (255, 255, 255), 2)
            cv2.circle(preview, p2, 4, (255, 255, 255), -1)
        if len(waypoints) >= 2:
            p_prev = (int(waypoints[-2][0]), int(waypoints[-2][1]))
            p_end = (int(waypoints[-1][0]), int(waypoints[-1][1]))
            r_end = perspective_radius(base_r, p_end[0], p_end[1], H, W)
            cv2.arrowedLine(preview, p_prev, p_end,
                            (255, 255, 255), 3, tipLength=0.25)
            cv2.circle(preview, p_end, r_end, color, 2)
    return preview


# ─── Camera Depth Helpers ────────────────────────────────────────────────────

@torch.no_grad()
def estimate_depth_single_frame(depth_model, vggt_model, image_tensor, device="cuda"):
    """
    Use Video Depth Anything for visual depth and VGGT for raw depth + intrinsics.

    Args:
        depth_model: VideoDepthAnything model (eval mode)
        vggt_model: VGGT model (eval mode)
        image_tensor: [3, H, W] in [0, 255]
    Returns:
        depth_visual: [1, 3, H, W] normalised to [0, 255] (from VDA)
        raw_depth:    [1, 1, H, W] raw depth for geometry (from VGGT)
        intrinsics:   [1, 1, 3, 3] (from VGGT)
    """
    C, H, W = image_tensor.shape

    # ── VDA depth (single-frame, higher visual quality) ──
    frame_np = image_tensor.permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)  # [H, W, 3]
    original_dtype_vda = next(depth_model.parameters()).dtype
    depth_model.float()
    depths_np, _ = depth_model.infer_video_depth(
        frame_np[None],  # expects [T, H, W, 3]
        target_fps=8, input_size=518,
        device=str(depth_model.pretrained.blocks[0].attn.qkv.weight.device),
    )
    depth_model.to(original_dtype_vda)
    d_vda = depths_np[0]
    d_vda = (d_vda - d_vda.min()) / (d_vda.max() - d_vda.min() + 1e-8) * 255.0
    depth_visual = torch.from_numpy(d_vda).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1, 3, H, W]

    # ── VGGT for raw depth + intrinsics (for 3D geometry, model kept in float32) ──
    frame_01 = image_tensor.float().unsqueeze(0) / 255.0  # [1, 3, H, W]
    pad_h = (14 - H % 14) % 14
    pad_w = (14 - W % 14) % 14
    if pad_h > 0 or pad_w > 0:
        frame_01 = F.pad(frame_01, (0, pad_w, 0, pad_h), mode='reflect')
    H_pad, W_pad = frame_01.shape[2], frame_01.shape[3]

    frame_01 = frame_01.to(device)
    vggt_model.float()
    with torch.cuda.amp.autocast(enabled=False):
        predictions = vggt_model(frame_01.unsqueeze(0))

    raw_depth = predictions["depth"][:, :, :H, :W, 0]  # [1, 1, H, W]
    pose_enc = predictions["pose_enc"]
    _, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H_pad, W_pad))

    return depth_visual.to(device), raw_depth.to(device), intrinsics.to(device)


def speed_weights_to_progress(weights, num_frames, half_T):
    """Convert keyframe speed weights to a per-frame progress curve in [0, 1].

    Args:
        weights: list/array of relative speed values at evenly-spaced keyframes.
                 All-ones gives uniform (linear) progress.
        num_frames: total frame count.
        half_T: active motion boundary; progress stays 1.0 for t >= half_T.

    Returns:
        np.ndarray of shape [num_frames] with values in [0, 1].
    """
    weights = np.array(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.01)
    n = max(min(half_T, num_frames), 2)

    key_t = np.linspace(0, 1, len(weights))
    frame_t = np.linspace(0, 1, n)
    speeds = np.interp(frame_t, key_t, weights)

    intervals = (speeds[:-1] + speeds[1:]) / 2.0
    cumsum = np.cumsum(intervals)
    total = cumsum[-1]
    progress = np.zeros(n)
    if total > 0:
        progress[1:] = cumsum / total

    full = np.ones(num_frames)
    full[:n] = progress
    return full


def build_camera_trajectory(tx, ty, tz, ry, num_frames, half_T, progress_curve=None):
    """
    Build smooth camera extrinsics from user-specified translation and yaw.

    Frame 0 is identity. Motion is linearly interpolated up to half_T,
    then held constant (those frames are zero-padded in warping anyway).

    Args:
        tx: total left/right translation (positive = camera moves right)
        ty: total up/down translation (positive = camera moves down, OpenCV convention)
        tz: total forward/backward translation (positive = forward)
        ry: total yaw rotation in degrees (positive = camera looks left in w2c convention)
        num_frames: total frame count
        half_T: first-half boundary

    Returns:
        extrinsics: [1, num_frames, 3, 4]
    """
    extrinsics = torch.zeros(1, num_frames, 3, 4)
    for t in range(num_frames):
        if progress_curve is not None:
            alpha = float(progress_curve[t])
        else:
            alpha = min(t / max(half_T - 1, 1), 1.0)
        r = Rotation.from_euler('y', ry * alpha, degrees=True)
        R = torch.tensor(r.as_matrix(), dtype=torch.float32)
        cam_center = torch.tensor([tx * alpha, ty * alpha, tz * alpha],
                                  dtype=torch.float32)
        extrinsics[0, t, :3, :3] = R
        extrinsics[0, t, :3, 3] = -(R @ cam_center)
    return extrinsics


def warp_depth_with_cameras(depth_map_visual, raw_depth_0, extrinsics, intrinsics,
                            num_frames, half_T, static_mask=None):
    """
    Forward-splat first-frame depth via 3D reprojection (aligned with training).

    Args:
        depth_map_visual: [B, 3, H, W] in [0, 255]
        raw_depth_0:      [B, 1, H, W] raw depth for geometry
        extrinsics:       [B, T, 3, 4]
        intrinsics:       [B, T, 3, 3] or [B, 1, 3, 3]
        num_frames:       total frames
        half_T:           frames >= half_T are zero-padded (training behavior)
        static_mask:      [B, 1, H, W], 1=static, 0=dynamic objects
    Returns:
        camera_depth_video: [B, T, 3, H, W] in [-1, 1]
    """
    B, C, H, W = depth_map_visual.shape
    device = depth_map_visual.device
    dtype = depth_map_visual.dtype

    if static_mask is not None:
        depth_map_visual = depth_map_visual * static_mask
        raw_depth_0 = raw_depth_0 * static_mask

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    ones = torch.ones_like(u_coords)
    pixel_flat = torch.stack([u_coords, v_coords, ones], dim=0).view(3, H * W)
    pixel_flat = pixel_flat.unsqueeze(0).expand(B, -1, -1)

    K_0 = intrinsics[:, 0]
    K_0_inv = torch.inverse(K_0.float()).to(dtype)
    R_0 = extrinsics[:, 0, :3, :3]
    t_0 = extrinsics[:, 0, :3, 3:]

    rays = torch.bmm(K_0_inv, pixel_flat)
    depth_flat = raw_depth_0.reshape(B, 1, H * W)
    p_cam0 = rays * depth_flat
    R_0_T = R_0.transpose(1, 2)
    p_world = torch.bmm(R_0_T, p_cam0 - t_0)

    src_flat = depth_map_visual.reshape(B, C, H * W)
    zero_frame = torch.zeros(B, 1, C, H, W, device=device, dtype=dtype)
    warped_frames = []
    for t in range(num_frames):
        if t >= half_T:
            warped_frames.append(zero_frame)
            continue

        R_t = extrinsics[:, t, :3, :3]
        t_t = extrinsics[:, t, :3, 3:]
        K_t = intrinsics[:, t] if intrinsics.shape[1] > 1 else intrinsics[:, 0]

        p_cam_t = torch.bmm(R_t, p_world) + t_t
        p_pixel_t = torch.bmm(K_t, p_cam_t)

        z_t = p_pixel_t[:, 2:3].clamp(min=1e-6)
        u_t = (p_pixel_t[:, 0:1] / z_t).squeeze(1)
        v_t = (p_pixel_t[:, 1:2] / z_t).squeeze(1)

        target_x = u_t.round().long()
        target_y = v_t.round().long()

        valid = ((target_x >= 0) & (target_x < W) &
                 (target_y >= 0) & (target_y < H) &
                 (p_cam_t[:, 2] > 0))
        target_idx = (target_y * W + target_x).clamp(0, H * W - 1)

        out_flat = torch.zeros(B, C, H * W + 1, device=device, dtype=dtype)
        valid_flat = valid.unsqueeze(1).expand_as(src_flat)
        idx_flat = torch.where(
            valid_flat, 
            target_idx.unsqueeze(1).expand_as(src_flat), 
            torch.tensor(H * W, device=device, dtype=torch.long)
        )
        out_flat.scatter_reduce_(2, idx_flat, src_flat * valid_flat.float(), reduce='amax', include_self=True)
        out_flat = out_flat[:, :, :H * W]
        warped_frames.append(out_flat.view(B, C, H, W).unsqueeze(1))

    camera_depth_video = torch.cat(warped_frames, dim=1)
    camera_depth_video = camera_depth_video / 127.5 - 1.0
    return camera_depth_video


# ─── Camera Path Helpers ─────────────────────────────────────────────────────

CAM_ORIGIN_OFFSET_Y = 30  # pixels above bottom edge


def render_cam_path_canvas(image_np, cam_waypoints, H, W):
    """Render interactive camera path overlay on a dimmed copy of the input image."""
    if image_np is not None:
        canvas = (image_np.astype(np.float32) * 0.3).astype(np.uint8)
    else:
        canvas = np.full((H, W, 3), 30, dtype=np.uint8)

    origin_x = W // 2
    origin_y = H - CAM_ORIGIN_OFFSET_Y

    # Grid
    for gx in range(0, W, W // 8):
        cv2.line(canvas, (gx, 0), (gx, H), (50, 50, 50), 1)
    for gy in range(0, H, H // 6):
        cv2.line(canvas, (0, gy), (W, gy), (50, 50, 50), 1)

    # Origin marker (ego vehicle)
    cv2.circle(canvas, (origin_x, origin_y), 12, (0, 255, 0), -1)
    cv2.circle(canvas, (origin_x, origin_y), 14, (255, 255, 255), 2)
    cv2.putText(canvas, "EGO", (origin_x - 18, origin_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if cam_waypoints:
        pts = [(origin_x, origin_y)] + [(int(x), int(y)) for x, y in cam_waypoints]

        # Mirror the forward-kick used in build_camera_trajectory_from_path
        first = pts[1]
        dx = first[0] - origin_x
        dy = origin_y - first[1]
        d_first = math.sqrt(dx ** 2 + dy ** 2)
        kick_px = max(d_first * 0.2, 15)
        pts.insert(1, (origin_x, int(origin_y - kick_px)))

        smoothed = smooth_path_2d(pts)
        if len(smoothed) >= 2:
            smooth_int = [(int(round(x)), int(round(y))) for x, y in smoothed]
            for j in range(len(smooth_int) - 1):
                cv2.line(canvas, smooth_int[j], smooth_int[j + 1],
                         (0, 200, 255), 3)
            cv2.arrowedLine(canvas, smooth_int[-2], smooth_int[-1],
                            (0, 200, 255), 3, tipLength=0.3)

        # Show only user-clicked waypoints (skip synthetic kick at pts[1])
        user_pts = pts[2:]
        for i, pt in enumerate(user_pts, 1):
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 6, (255, 100, 0), -1)
            cv2.putText(canvas, str(i),
                        (int(pt[0]) + 8, int(pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    return canvas


def build_camera_trajectory_from_path(cam_waypoints, num_frames, half_T, H, W,
                                      cam_scale=0.15, progress_curve=None):
    """
    Convert pixel-space camera waypoints to driving-style per-frame extrinsics.

    Lateral displacement on the canvas produces yaw rotation (steering)
    rather than sideways translation, matching real driving behaviour.
    The heading at each frame is derived from the path tangent.
    """
    if not cam_waypoints:
        return build_camera_trajectory(0, 0, cam_scale, 0, num_frames, half_T,
                                       progress_curve=progress_curve)

    origin_x = W // 2
    origin_y = H - CAM_ORIGIN_OFFSET_Y
    scale = cam_scale / H

    ctrl = [(0.0, 0.0)]
    for wx, wy in cam_waypoints:
        tx = (wx - origin_x) * scale
        tz = (origin_y - wy) * scale
        ctrl.append((tx, tz))

    # Insert a forward-kick point so the path starts heading straight (yaw≈0).
    # Without this, a single left-forward click would give constant yaw from
    # frame 0 instead of the expected gradual steering.
    first = ctrl[1]
    d_first = math.sqrt(first[0] ** 2 + first[1] ** 2)
    kick_z = max(d_first * 0.2, scale * 15)
    ctrl.insert(1, (0.0, kick_z))

    ctrl = smooth_path_2d(ctrl)
    print(f"  [cam_path] scale={scale:.6f}  "
          f"final_tx={ctrl[-1][0]:.4f}  final_tz={ctrl[-1][1]:.4f}")

    extrinsics = torch.zeros(1, num_frames, 3, 4)
    eps = 1e-4
    yaw_gain = 0.15

    for t in range(num_frames):
        if progress_curve is not None:
            frac = float(progress_curve[t])
        else:
            frac = min(t / max(half_T - 1, 1), 1.0)
        tx_t, tz_t = interpolate_polyline(ctrl, frac)

        frac_fwd = min(frac + eps, 1.0)
        frac_bwd = max(frac - eps, 0.0)
        p_fwd = interpolate_polyline(ctrl, frac_fwd)
        p_bwd = interpolate_polyline(ctrl, frac_bwd)
        dx = p_fwd[0] - p_bwd[0]
        dz = p_fwd[1] - p_bwd[1]

        heading = math.atan2(dx, dz) if (abs(dx) > 1e-10 or abs(dz) > 1e-10) else 0.0
        yaw = -heading * yaw_gain

        r = Rotation.from_euler('y', yaw)
        R = torch.tensor(r.as_matrix(), dtype=torch.float32)
        cam_center = torch.tensor([tx_t, 0.0, tz_t], dtype=torch.float32)
        extrinsics[0, t, :3, :3] = R
        extrinsics[0, t, :3, 3] = -(R @ cam_center)

    return extrinsics


# ─── Condition Construction ──────────────────────────────────────────────────

@torch.no_grad()
def build_conditions(image_pil, masks_data, paths_dict,
                     cam_waypoints=None, causal=True,
                     cam_speed_weights=None,
                     num_frames_pipeline=50, pipe=None, depth_model=None,
                     vggt_model=None, device="cuda", dtype=torch.bfloat16):
    """
    Build all tensors the pipeline needs from user inputs.

    Args:
        cam_waypoints: list of (x, y) pixel coords for camera path, or None/empty
                       for default gentle forward motion.

    Returns:
        image_input, latent_point, latent_camera_depth,
        circle_preview, circle_frames_np, cam_depth_frames_np
    """
    H, W = TARGET_HEIGHT, TARGET_WIDTH
    vae = pipe.vae

    circles = compute_object_circles(masks_data)

    vae_sf_t = pipe.vae_scale_factor_temporal
    patch_size_t = pipe.transformer.config.patch_size_t or 1
    total_latent = (num_frames_pipeline - 1) // vae_sf_t + 4
    if patch_size_t > 1 and total_latent % patch_size_t != 0:
        total_latent += patch_size_t - total_latent % patch_size_t
    s = (total_latent - 2) // 2
    circle_raw_frames = (s - 1) * vae_sf_t + 1

    # ── Depth estimation (needed before circle video for world-space reprojection) ──
    image_resized = image_pil.resize((W, H), Image.LANCZOS)
    image_np_f = np.array(image_resized).astype(np.float32)
    image_tensor_255 = torch.from_numpy(image_np_f).permute(2, 0, 1)  # [3, H, W]

    depth_visual, raw_depth, intrinsics = estimate_depth_single_frame(
        depth_model, vggt_model, image_tensor_255, device=device,
    )

    # Training-aligned static mask: remove dynamic instances (YOLO masks from frame 0)
    # before warping so moving objects leave natural black trails/holes.
    if masks_data:
        dynamic_mask_np = np.zeros((H, W), dtype=np.float32)
        for binary_mask, _, _ in masks_data:
            dynamic_mask_np = np.maximum(dynamic_mask_np, (binary_mask > 0).astype(np.float32))
        static_mask = torch.from_numpy(1.0 - dynamic_mask_np).unsqueeze(0).unsqueeze(0).to(device)
    else:
        static_mask = torch.ones(1, 1, H, W, device=device, dtype=torch.float32)

    # ── Camera trajectory (needed before circle video for reprojection) ──
    half_T = (circle_raw_frames + 1) // 2 if causal else circle_raw_frames

    # Adaptive scale: proportional to median scene depth so camera motion
    # produces ~30-pixel shifts at typical depth, regardless of depth units.
    valid_depth = raw_depth[raw_depth > 0]
    median_d = valid_depth.median().item() if valid_depth.numel() > 0 else 1.0
    cam_scale = 0.6 * median_d
    print(f"  [cam] median_depth={median_d:.4f}  cam_scale={cam_scale:.4f}")

    progress_curve = None
    if cam_speed_weights is not None:
        progress_curve = speed_weights_to_progress(
            cam_speed_weights, circle_raw_frames, half_T,
        )

    if cam_waypoints and len(cam_waypoints) > 0:
        extrinsics = build_camera_trajectory_from_path(
            cam_waypoints, num_frames=circle_raw_frames, half_T=half_T,
            H=H, W=W, cam_scale=cam_scale, progress_curve=progress_curve,
        ).to(device)
    else:
        extrinsics = build_camera_trajectory(
            0, 0, cam_scale, 0,
            num_frames=circle_raw_frames, half_T=half_T,
            progress_curve=progress_curve,
        ).to(device)
    intrinsics = intrinsics.to(device)

    # ── Circle video: unproject item waypoints to world space, reproject per frame ──
    circle_frames_np = build_circle_video_frames_with_camera(
        circles, paths_dict, circle_raw_frames, H, W,
        raw_depth, intrinsics, extrinsics,
    )
    circle_preview = render_circle_preview(
        circles, paths_dict, H, W,
    )

    # Encode circle video -> latent_point
    circle_tensor = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1) for f in circle_frames_np
    ]).float().unsqueeze(0)
    circle_tensor = circle_tensor.permute(0, 2, 1, 3, 4) / 127.5 - 1.0
    circle_tensor = circle_tensor.to(vae.device, dtype=vae.dtype)
    latent_point = (
        vae.encode(circle_tensor).latent_dist.sample()
        * vae.config.scaling_factor
    )
    latent_point = latent_point.permute(0, 2, 1, 3, 4)

    # ── Warp depth with camera trajectory ──
    camera_depth_video = warp_depth_with_cameras(
        depth_visual.float().to(device),
        raw_depth.float().to(device),
        extrinsics.float(),
        intrinsics.float(),
        num_frames=circle_raw_frames,
        half_T=half_T,
        static_mask=static_mask.float(),
    )  # [1, T, 3, H, W] in [-1, 1]

    # VAE-encode camera_depth_video -> latent_camera_depth
    cam_depth_for_vae = camera_depth_video.permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]
    cam_depth_for_vae = cam_depth_for_vae.to(vae.device, dtype=vae.dtype)
    latent_camera_depth = (
        vae.encode(cam_depth_for_vae).latent_dist.sample()
        * vae.config.scaling_factor
    )
    latent_camera_depth = latent_camera_depth.permute(0, 2, 1, 3, 4)

    # Camera depth frames for preview
    cam_depth_frames_np = (
        camera_depth_video.add(1).mul(127.5).byte()
        .cpu().squeeze(0).permute(0, 2, 3, 1).numpy()
    )  # [T, H, W, 3]

    # ── Build depth first-frame condition (matching training: video_cond[:, :, :1]) ──
    depth_frame_01 = depth_visual.float() / 127.5 - 1.0  # [1, 3, H, W] -> [-1, 1]
    depth_frame_cond = depth_frame_01.unsqueeze(2)  # [1, 3, 1, H, W]

    # Reference image [-1, 1]
    ref_img = (
        torch.from_numpy(image_np_f).permute(2, 0, 1).float() / 127.5 - 1.0
    ).unsqueeze(0).unsqueeze(2).to(depth_frame_cond.device)  # [1, 3, 1, H, W]

    # image_input = [depth_first_frame, rgb_image] -> [1, 3, 2, H, W]
    image_input = torch.cat([depth_frame_cond, ref_img], dim=2)

    # Move to device
    image_input = image_input.to(device, dtype)
    latent_point = latent_point.to(device, dtype)
    latent_camera_depth = latent_camera_depth.to(device, dtype)

    return (image_input, latent_point, latent_camera_depth,
            circle_preview, circle_frames_np, cam_depth_frames_np)


# ─── Video Generation ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_video(
    pipe, image_input, latent_point, latent_camera_depth,
    prompt, num_frames, num_inference_steps, guidance_scale, seed,
):
    """
    Run the two-stage pipeline: motion (shape) -> RGB (appearance).

    Returns:
        list of PIL images (video frames)
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Load cached prompt embeddings instead of using raw string prompt
    embeds_path = os.path.join(os.path.dirname(__file__), "models", "prompt_embeds.pt")
    if os.path.exists(embeds_path):
        cached_embeds = torch.load(embeds_path)
        prompt_embeds = cached_embeds["prompt_embeds"].to(pipe.device, dtype=pipe.transformer.dtype)
        negative_prompt_embeds = cached_embeds["negative_prompt_embeds"].to(pipe.device, dtype=pipe.transformer.dtype)
        
        video_frames = pipe(
            num_frames=num_frames,
            height=TARGET_HEIGHT,
            width=TARGET_WIDTH,
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image_input,
            latent_point=latent_point,
            latent_camera_depth=latent_camera_depth,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
    else:
        # Fallback to string prompt if cache not found
        video_frames = pipe(
            num_frames=num_frames,
            height=TARGET_HEIGHT,
            width=TARGET_WIDTH,
            prompt=prompt,
            image=image_input,
            latent_point=latent_point,
            latent_camera_depth=latent_camera_depth,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]

    return video_frames


# ─── Gradio Interface ────────────────────────────────────────────────────────

def create_demo(pipe, yolo_model, depth_model, vggt_model, device, dtype):
    """Build and return the Gradio Blocks app."""

    # ── Callbacks ────────────────────────────────────────────────────────

    def on_upload(image_pil):
        """Upload -> resize -> YOLO segmentation."""
        if image_pil is None:
            return (
                None, None, None, None, None, None,
                "Upload an image to begin.",
                None, None, None, None, 0, {},
                "No paths defined.",
                None, [],                     # display_cam, cam_waypoints
            )
        image_resized = image_pil.resize(
            (TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS
        )
        image_np = np.array(image_resized)

        seg_overlay, masks_data = run_yolo_segmentation(yolo_model, image_np)
        blended = overlay_with_highlight(image_np, seg_overlay, masks_data)

        raw_seg = build_raw_seg_frame(masks_data, TARGET_HEIGHT, TARGET_WIDTH)
        cv2.imwrite("/tmp/seg_latest.png", cv2.cvtColor(raw_seg, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/tmp/img_latest.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print("  [saved] /tmp/seg_latest.png  /tmp/img_latest.png")

        cam_canvas = render_cam_path_canvas(
            image_np, [], TARGET_HEIGHT, TARGET_WIDTH,
        )

        n = len(masks_data)
        status = (
            f"Detected {n} object{'s' if n != 1 else ''}. "
            "Click on an object to select it."
        )
        return (
            Image.fromarray(blended),     # display_seg
            None,                         # flow_preview
            None,                         # depth_video_output
            None,                         # rgb_video_output
            None,                         # point_video_output
            None,                         # cam_depth_output
            status,
            image_np, seg_overlay, masks_data,
            None, 0, {},                  # selected_idx, click_phase, paths
            "No paths defined.",          # paths_info
            Image.fromarray(cam_canvas),  # display_cam
            [],                           # cam_waypoints
        )

    def _paths_info_text(d):
        if not d:
            return "No paths defined."
        return "\n".join(
            f"Object #{k}: {len(v)} waypoint{'s' if len(v) != 1 else ''}"
            for k, v in sorted(d.items())
        )

    def on_seg_click(
        evt: gr.SelectData,
        image_np, seg_overlay, masks_data,
        selected_idx, click_phase,
        arrow_start_state, paths_dict,
    ):
        """
        Phase 0 -> click on object to select it.
        Phase 1 -> each click adds a waypoint to the current path.
        'Finish Annotation' button commits the path and returns to Phase 0.
        """
        if paths_dict is None:
            paths_dict = {}

        NOOP = (
            gr.update(), gr.update(), gr.update(),
            "No objects detected. Upload a different image.",
            selected_idx, click_phase, arrow_start_state, paths_dict,
        )
        if masks_data is None or len(masks_data) == 0:
            return NOOP

        x, y = evt.index[0], evt.index[1]

        if click_phase == 0:
            # ── Select an object ──
            idx = find_mask_at_point(masks_data, x, y)
            if idx is None:
                blended = overlay_with_highlight(
                    image_np, seg_overlay, masks_data,
                    paths_dict=paths_dict,
                )
                return (
                    Image.fromarray(blended), gr.update(), gr.update(),
                    "No object at that position. Click on a coloured region.",
                    selected_idx, 0, None, paths_dict,
                )

            # Starting a new path for this object (replaces any existing one)
            new_paths = dict(paths_dict)
            new_paths[idx] = []

            blended = overlay_with_highlight(
                image_np, seg_overlay, masks_data,
                selected_idx=idx, paths_dict=new_paths,
            )
            cv2.circle(blended, (int(x), int(y)), 8, (0, 255, 0), -1)

            # Use the object's bounding-box center as the arrow origin so
            # that subsequent waypoint displacements (dx, dy) align with
            # where the user actually clicks on screen.
            circles = compute_object_circles(masks_data)
            obj_cx, obj_cy = circles[idx][0], circles[idx][1]

            return (
                Image.fromarray(blended), gr.update(), gr.update(),
                f"Object #{idx} selected. Click to add waypoints, "
                "then press Finish Annotation.",
                idx, 1, (obj_cx, obj_cy), new_paths,
            )

        else:
            # ── Phase 1: add waypoint to current path ──
            arrow_start = arrow_start_state
            if arrow_start is None:
                return (
                    gr.update(), gr.update(), gr.update(),
                    "Error: no start point. Click an object first.",
                    selected_idx, 0, None, paths_dict,
                )

            sx, sy = arrow_start
            dx, dy = x - sx, y - sy

            new_paths = dict(paths_dict)
            cur = list(new_paths.get(selected_idx, []))
            cur.append((dx, dy))
            new_paths[selected_idx] = cur

            blended = overlay_with_highlight(
                image_np, seg_overlay, masks_data,
                selected_idx=selected_idx, paths_dict=new_paths,
            )

            circles = compute_object_circles(masks_data)
            preview_np = render_circle_preview(
                circles, new_paths, TARGET_HEIGHT, TARGET_WIDTH,
            )

            return (
                Image.fromarray(blended),
                Image.fromarray(preview_np),
                gr.update(value=_paths_info_text(new_paths)),
                f"Waypoint added for Obj#{selected_idx} "
                f"({len(cur)} pts). Click more or Finish Annotation.",
                selected_idx, 1, arrow_start, new_paths,
            )

    def on_finish_annotation(
        image_np, seg_overlay, masks_data,
        selected_idx, paths_dict,
    ):
        """Commit current annotation and return to Phase 0."""
        if paths_dict is None:
            paths_dict = {}

        # Remove the entry if no waypoints were added
        if selected_idx is not None and selected_idx in paths_dict:
            if len(paths_dict[selected_idx]) == 0:
                paths_dict = {k: v for k, v in paths_dict.items()
                              if k != selected_idx}

        blended = overlay_with_highlight(
            image_np, seg_overlay, masks_data, paths_dict=paths_dict,
        )

        n = len(paths_dict)
        status = (
            f"Annotation finished for Obj#{selected_idx}. "
            f"{n} object{'s' if n != 1 else ''} annotated. "
            "Click another object or Generate."
            if selected_idx is not None
            else "Click on an object to annotate."
        )
        circles = compute_object_circles(masks_data) if masks_data else []
        preview_np = render_circle_preview(
            circles, paths_dict, TARGET_HEIGHT, TARGET_WIDTH,
        ) if paths_dict else None
        preview = Image.fromarray(preview_np) if preview_np is not None else None

        return (
            Image.fromarray(blended),
            preview,
            _paths_info_text(paths_dict),
            status,
            None, 0, None, paths_dict,
        )

    def on_generate_point(
        image_np, masks_data, paths_dict,
        arrow_scale, cam_waypoints, num_frames, causal,
        s0, s1, s2, s3, s4, s5,
    ):
        """Build point/circle video and camera depth video (fast, no diffusion)."""
        if image_np is None:
            return gr.update(), gr.update(), gr.update(), "Please upload an image first."

        if not paths_dict:
            paths_dict = {}
        if not masks_data:
            masks_data = []

        cam_speed_weights = [s0, s1, s2, s3, s4, s5]
        is_uniform = all(abs(w - 1.0) < 1e-6 for w in cam_speed_weights)

        scaled_paths = {
            idx: [(dx * arrow_scale, dy * arrow_scale) for dx, dy in wps]
            for idx, wps in paths_dict.items()
        }
        image_pil = Image.fromarray(image_np)

        (image_input, latent_point, latent_camera_depth,
         _, circle_frames_np, cam_depth_frames_np) = build_conditions(
            image_pil, masks_data, scaled_paths,
            cam_waypoints=cam_waypoints if cam_waypoints else None,
            causal=bool(causal),
            cam_speed_weights=None if is_uniform else cam_speed_weights,
            num_frames_pipeline=int(num_frames),
            pipe=pipe, depth_model=depth_model, vggt_model=vggt_model,
            device=device, dtype=dtype,
        )

        point_video_frames = [Image.fromarray(f) for f in circle_frames_np]
        tmp_point = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(point_video_frames, tmp_point.name, fps=8)

        cam_depth_pils = [Image.fromarray(f) for f in cam_depth_frames_np]
        tmp_cam = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(cam_depth_pils, tmp_cam.name, fps=8)

        conditions = (image_input, latent_point, latent_camera_depth)
        return (tmp_point.name, tmp_cam.name, conditions,
                "Condition videos ready. Click Generate Video to run diffusion.")

    def on_generate_video(
        conditions_state, num_frames, num_steps, guidance, seed,
        progress=gr.Progress(track_tqdm=True),
    ):
        """Run diffusion using pre-built conditions from on_generate_point."""
        if conditions_state is None:
            return gr.update(), gr.update(), "Generate Point Video first."

        image_input, latent_point, latent_camera_depth = conditions_state

        try:
            frames = generate_video(
                pipe, image_input, latent_point, latent_camera_depth,
                DEFAULT_PROMPT, int(num_frames), int(num_steps),
                float(guidance), int(seed),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return gr.update(), gr.update(), f"Generation failed: {e}"

        depth_frames = []
        rgb_frames = []
        for frame in frames:
            w, h = frame.size
            half_h = h // 2
            depth_frames.append(frame.crop((0, 0, w, half_h)))
            rgb_frames.append(frame.crop((0, half_h, w, h)))

        tmp_depth = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(depth_frames, tmp_depth.name, fps=8)
        tmp_rgb = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(rgb_frames, tmp_rgb.name, fps=8)
        return tmp_depth.name, tmp_rgb.name, "Video generated successfully!"

    def on_reset(image_np, seg_overlay, masks_data):
        """Reset all selections and paths."""
        if image_np is None:
            return (None, None, "Upload an image.",
                    None, 0, None, {}, "No paths defined.")
        blended = overlay_with_highlight(image_np, seg_overlay, masks_data)
        return (
            Image.fromarray(blended), None,
            "Selection reset. Click on an object to start over.",
            None, 0, None, {}, "No paths defined.",
        )

    def on_cam_click(evt: gr.SelectData, image_np, cam_waypoints):
        """Add a waypoint to the camera path."""
        if cam_waypoints is None:
            cam_waypoints = []
        x, y = evt.index[0], evt.index[1]
        cam_waypoints = list(cam_waypoints) + [(x, y)]
        canvas = render_cam_path_canvas(
            image_np, cam_waypoints, TARGET_HEIGHT, TARGET_WIDTH,
        )
        n = len(cam_waypoints)
        return (
            Image.fromarray(canvas),
            cam_waypoints,
            f"{n} waypoint{'s' if n != 1 else ''} defined. "
            "Click more or Generate.",
        )

    def on_cam_undo(image_np, cam_waypoints):
        """Remove the last camera waypoint."""
        if not cam_waypoints:
            canvas = render_cam_path_canvas(
                image_np, [], TARGET_HEIGHT, TARGET_WIDTH,
            )
            return Image.fromarray(canvas), [], "No waypoints to undo."
        cam_waypoints = list(cam_waypoints)[:-1]
        canvas = render_cam_path_canvas(
            image_np, cam_waypoints, TARGET_HEIGHT, TARGET_WIDTH,
        )
        n = len(cam_waypoints)
        msg = (f"{n} waypoint{'s' if n != 1 else ''} remaining."
               if n > 0 else "All waypoints removed. Default forward motion will be used.")
        return Image.fromarray(canvas), cam_waypoints, msg

    def on_cam_reset(image_np):
        """Clear the camera path entirely."""
        canvas = render_cam_path_canvas(
            image_np, [], TARGET_HEIGHT, TARGET_WIDTH,
        )
        return (
            Image.fromarray(canvas),
            [],
            "Camera path cleared. Default forward motion will be used.",
        )

    # ── Layout ───────────────────────────────────────────────────────────

    with gr.Blocks(
        title="Motion Forcing Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Motion Forcing: Point -> Shape -> Appearance\n"
            "Upload an image, select an object, draw a motion arrow, "
            "and generate a video."
        )

        # ── Hidden states ──
        st_image_np = gr.State(None)
        st_seg_overlay = gr.State(None)
        st_masks_data = gr.State(None)
        st_selected_idx = gr.State(None)
        st_click_phase = gr.State(0)
        st_arrow_start = gr.State(None)
        st_paths = gr.State({})
        st_conditions = gr.State(None)
        st_cam_waypoints = gr.State([])

        # ── Row 1: Stage 1 / Stage 2 / Stage 3 ──
        with gr.Row():
            # Stage 1
            with gr.Column(scale=1):
                gr.Markdown("### Stage 1: Upload Image")
                inp_image = gr.Image(type="pil", label="Input Image", height=280)
                flow_preview = gr.Image(label="Circle Motion Preview", height=200)

            # Stage 2
            with gr.Column(scale=1):
                gr.Markdown(
                    "### Stage 2: Annotate Object Motion\n"
                    "**1st click** on an object = select it.  \n"
                    "**Subsequent clicks** = add waypoints (curve)."
                )
                display_seg = gr.Image(
                    label="Segmentation (click here)",
                    interactive=False, height=360,
                )
                with gr.Row():
                    btn_finish = gr.Button("Next Object", variant="primary")
                    btn_reset = gr.Button("Reset All", variant="secondary")

            # Stage 3
            with gr.Column(scale=1):
                gr.Markdown(
                    "### Stage 3: Camera Motion\n"
                    "Click on the canvas to draw the ego-vehicle trajectory.  \n"
                    "**Up** = forward, **Left/Right** = lateral."
                )
                display_cam = gr.Image(
                    label="Camera Path (click to add waypoints)",
                    interactive=False, height=360,
                )
                with gr.Row():
                    btn_cam_undo = gr.Button("Undo Last", variant="secondary")
                    btn_cam_reset = gr.Button("Reset Camera", variant="secondary")

        # Hidden status outputs (still needed for event wiring)
        status_text = gr.Textbox(visible=False, value="Upload an image to begin.")
        cam_status = gr.Textbox(
            visible=False,
            value="No camera path. Default forward motion will be used.",
        )

        # ── Row 2: Step 4 Parameters (left) + Advanced Ego Speed (right) ──
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Step 4: Parameters")
                paths_info = gr.Textbox(
                    label="Defined Paths",
                    value="No paths defined.",
                    interactive=False, lines=3,
                )
                arrow_scale = gr.State(1.0)
                num_frames = gr.State(66)
                with gr.Row():
                    num_steps = gr.Slider(
                        5, 50, value=20, step=1, label="Inference Steps",
                    )
                with gr.Row():
                    guidance = gr.Slider(
                        1.0, 15.0, value=6.0, step=0.5, label="Guidance Scale",
                    )
                    seed = gr.Number(label="Seed", value=42, precision=0)
                with gr.Row():
                    btn_gen_point = gr.Button(
                        "1. Generate Point Video", variant="secondary", size="lg",
                    )
                    btn_gen_video = gr.Button(
                        "2. Generate Depth & RGB Video", variant="primary", size="lg",
                    )

            with gr.Column(scale=2):
                with gr.Accordion("Advanced Ego Speed", open=False):
                    gr.Markdown(
                        "Adjust relative speed at each keyframe "
                        "(1.0 = normal, <1 = slower, >1 = faster)."
                    )
                    causal_flag = gr.Checkbox(
                        label="Causal (depth warp first 50% only)",
                        value=False,
                    )
                    with gr.Row():
                        spd_0 = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                          label="Start")
                        spd_1 = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                          label="20%")
                    with gr.Row():
                        spd_2 = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                          label="40%")
                        spd_3 = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                          label="60%")
                    with gr.Row():
                        spd_4 = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                          label="80%")
                        spd_5 = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                          label="End")

        gr.Markdown("### Condition Previews")
        with gr.Row():
            point_video_output = gr.Video(label="Point/Circle Video", height=360)
            cam_depth_output = gr.Video(label="Camera Depth Video", height=360)

        gr.Markdown("### Generated Output")
        with gr.Row():
            depth_video_output = gr.Video(label="Generated Depth Video", height=360)
            rgb_video_output = gr.Video(label="Generated RGB Video", height=360)

        # ── Event wiring ─────────────────────────────────────────────────

        # Upload triggers segmentation + camera canvas init
        inp_image.change(
            fn=on_upload,
            inputs=[inp_image],
            outputs=[
                display_seg, flow_preview,
                depth_video_output, rgb_video_output,
                point_video_output, cam_depth_output,
                status_text,
                st_image_np, st_seg_overlay, st_masks_data,
                st_selected_idx, st_click_phase, st_paths,
                paths_info,
                display_cam, st_cam_waypoints,
            ],
        )

        # Click on segmentation image: select object or add waypoint
        display_seg.select(
            fn=on_seg_click,
            inputs=[
                st_image_np, st_seg_overlay, st_masks_data,
                st_selected_idx, st_click_phase, st_arrow_start, st_paths,
            ],
            outputs=[
                display_seg, flow_preview, paths_info,
                status_text,
                st_selected_idx, st_click_phase, st_arrow_start, st_paths,
            ],
        )

        # Click on camera canvas: add camera waypoint
        display_cam.select(
            fn=on_cam_click,
            inputs=[st_image_np, st_cam_waypoints],
            outputs=[display_cam, st_cam_waypoints, cam_status],
        )

        # Undo last camera waypoint
        btn_cam_undo.click(
            fn=on_cam_undo,
            inputs=[st_image_np, st_cam_waypoints],
            outputs=[display_cam, st_cam_waypoints, cam_status],
        )

        # Reset camera path
        btn_cam_reset.click(
            fn=on_cam_reset,
            inputs=[st_image_np],
            outputs=[display_cam, st_cam_waypoints, cam_status],
        )

        # Finish current object annotation
        btn_finish.click(
            fn=on_finish_annotation,
            inputs=[
                st_image_np, st_seg_overlay, st_masks_data,
                st_selected_idx, st_paths,
            ],
            outputs=[
                display_seg, flow_preview, paths_info,
                status_text,
                st_selected_idx, st_click_phase, st_arrow_start, st_paths,
            ],
        )

        # Reset all
        btn_reset.click(
            fn=on_reset,
            inputs=[st_image_np, st_seg_overlay, st_masks_data],
            outputs=[
                display_seg, flow_preview, status_text,
                st_selected_idx, st_click_phase, st_arrow_start,
                st_paths, paths_info,
            ],
        )

        # Generate condition videos (fast, no diffusion)
        btn_gen_point.click(
            fn=on_generate_point,
            inputs=[
                st_image_np, st_masks_data, st_paths,
                arrow_scale, st_cam_waypoints, num_frames, causal_flag,
                spd_0, spd_1, spd_2, spd_3, spd_4, spd_5,
            ],
            outputs=[point_video_output, cam_depth_output,
                     st_conditions, status_text],
        )

        # Generate full video (diffusion)
        btn_gen_video.click(
            fn=on_generate_video,
            inputs=[st_conditions, num_frames, num_steps, guidance, seed],
            outputs=[depth_video_output, rgb_video_output, status_text],
        )

    return demo


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    default_yolo = os.path.join(SCRIPT_DIR, "weights", "yolo11l-seg.pt")

    p = argparse.ArgumentParser(
        description="Motion Forcing Gradio Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from HuggingFace Hub (recommended, auto-downloads & caches)
  python gradio_demo.py --transformer_ckpt TSXu/forcing_depth

  # Load local checkpoint (only downloads tokenizer/text_encoder/vae, NOT transformer)
  python gradio_demo.py --transformer_ckpt /path/to/your/checkpoint_dir

  # DeepSpeed checkpoint (single .pt file)
  python gradio_demo.py \\
      --load_pretrained_weight /path/to/checkpoint-20000/pytorch_model/mp_rank_00_model_states.pt

  # Base model only (no fine-tuned weights, downloads full model)
  python gradio_demo.py --model_path THUDM/CogVideoX-5b-I2V
""",
    )
    p.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b-I2V",
        help="HF model ID or local path for CogVideoX base. When a local checkpoint "
             "is given via --transformer_ckpt or --load_pretrained_weight, only "
             "tokenizer/text_encoder/vae are loaded from here (skips transformer download).",
    )
    p.add_argument(
        "--yolo_path", type=str,
        default=default_yolo,
        help="Path to YOLO segmentation model (default: %(default)s)",
    )
    p.add_argument(
        "--load_pretrained_weight", type=str, default=None,
        help="Path to fine-tuned weight file or checkpoint directory. "
             "Supports DeepSpeed (.pt with 'module' key), safetensors, "
             "and plain PyTorch state dicts. Same flag as the training script.",
    )
    p.add_argument(
        "--transformer_ckpt", type=str, default="TSXu/MotionForcing_driving",
        help="Path or HuggingFace model ID for the fine-tuned transformer checkpoint. "
             "Accepts a local directory, a local file, or a HF model ID like 'TSXu/forcing_depth' "
             "(auto-downloaded/cached). Auto-detects DeepSpeed / safetensors / sharded format. "
             "(default: %(default)s)",
    )
    p.add_argument(
        "--lora_path", type=str, default=None,
        help="Path to LoRA weights directory (optional)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    p.add_argument(
        "--cpu_offload", action="store_true",
        help="Use sequential CPU offload to save GPU memory",
    )
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    pipe, yolo_model, depth_model, vggt_model = load_models(args)
    demo = create_demo(pipe, yolo_model, depth_model, vggt_model, device=args.device, dtype=dtype)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
