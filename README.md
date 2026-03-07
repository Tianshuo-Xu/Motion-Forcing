# Motion-Forcing

**Point → Shape → Appearance** video generation via motion forcing.

Given a single image, users can interactively select objects, draw motion trajectories, and control camera movement to generate realistic driving-scene videos.

## Project Structure

```
Motion-Forcing/
├── gradio_demo.py              # Gradio inference demo (main entry)
├── models/
│   ├── __init__.py
│   ├── pipeline.py             # CogVideoXImageToVideoPipeline (two-stage denoising)
│   ├── cogvideox_transformer_MD.py  # CogVideoXTransformer3DModel (motion-forcing)
│   └── normalization.py        # Custom norm layers
├── third_party/
│   └── Video-Depth-Anything/   # Depth estimation (git submodule)
├── weights/
│   └── yolo11l-seg.pt          # YOLO segmentation weights (download manually)
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/Tianshuo-Xu/Motion-Forcing.git
cd Motion-Forcing
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model weights

**YOLO segmentation model:**

```bash
mkdir -p weights
# Download yolo11l-seg.pt into weights/
wget -P weights/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt
```

**Video-Depth-Anything weights:**

```bash
cd third_party/Video-Depth-Anything
bash get_weights.sh
cd ../..
```

**CogVideoX base model & fine-tuned transformer** are automatically downloaded from HuggingFace Hub on first run:
- Base: `THUDM/CogVideoX-5b-I2V`
- Transformer: `TSXu/driving_forcing_80k`

## Usage

### Quick Start (recommended)

```bash
python gradio_demo.py
```

This loads the default configuration:
- Base model: `THUDM/CogVideoX-5b-I2V`
- Fine-tuned transformer: `TSXu/driving_forcing_80k`

### Custom Checkpoints

```bash
# HuggingFace model ID
python gradio_demo.py \
    --model_path THUDM/CogVideoX-5b-I2V \
    --transformer_ckpt TSXu/driving_forcing_80k

# Local checkpoint directory
python gradio_demo.py \
    --transformer_ckpt /path/to/finetuned/transformer_dir

# DeepSpeed checkpoint
python gradio_demo.py \
    --load_pretrained_weight /path/to/checkpoint-XXXX/pytorch_model/mp_rank_00_model_states.pt

# With LoRA
python gradio_demo.py \
    --lora_path /path/to/lora_weights
```

### Options

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `THUDM/CogVideoX-5b-I2V` | Base CogVideoX model (HF ID or local path) |
| `--transformer_ckpt` | `TSXu/driving_forcing_80k` | Fine-tuned transformer (HF ID or local path) |
| `--load_pretrained_weight` | `None` | Direct weight file path (DeepSpeed / .pt / .safetensors) |
| `--lora_path` | `None` | LoRA weights directory |
| `--yolo_path` | `weights/yolo11l-seg.pt` | YOLO segmentation model path |
| `--device` | `cuda` | Device |
| `--dtype` | `bfloat16` | Model precision (`bfloat16` / `float16`) |
| `--cpu_offload` | `False` | Enable sequential CPU offload to save VRAM |
| `--port` | `7860` | Gradio server port |
| `--share` | `False` | Create public Gradio link |

## Workflow

1. **Upload** an image → YOLO automatically segments objects
2. **Click** on an object to select it
3. **Click** additional waypoints to draw a motion path
4. **Finish Annotation** → repeat for more objects
5. **Camera Path** (optional) → click on the camera canvas to control ego-vehicle trajectory
6. **Generate Point Video** → builds circle-motion and camera-depth condition videos
7. **Generate Video** → runs two-stage diffusion (shape → appearance)

## Models Used

| Model | Source | Purpose |
|---|---|---|
| CogVideoX-5b-I2V | [THUDM/CogVideoX-5b-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V) | Base video generation |
| Motion-Forcing Transformer | [TSXu/driving_forcing_80k](https://huggingface.co/TSXu/driving_forcing_80k) | Fine-tuned transformer |
| YOLO11l-seg | [ultralytics](https://github.com/ultralytics/ultralytics) | Instance segmentation |
| Video-Depth-Anything | [DepthAnything](https://github.com/DepthAnything/Video-Depth-Anything) | Monocular depth estimation |
| VGGT-1B | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | Camera pose estimation |

## License

MIT
