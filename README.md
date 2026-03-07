<div align="center">

# Motion Forcing: A Decoupled Framework for Robust Video Generation in Motion Dynamics

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome)](https://tianshuo-xu.github.io/Motion-Forcing/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/TSXu/MotionForcing_driving)

**Tianshuo Xu<sup>1</sup>, Zhifei Chen<sup>1</sup>, Leyi Wu<sup>1</sup>, Hao Lu<sup>1</sup>, Ying-cong Chen<sup>1,2\*</sup>**

<sup>1</sup>HKUST (GZ) &nbsp;&nbsp; <sup>2</sup>HKUST &nbsp;&nbsp; \* corresponding author

</div>

Motion Forcing decouples physical reasoning from visual synthesis via a hierarchical **Point → Shape → Appearance** paradigm, enabling precise and physically consistent video generation from a single image and user-drawn trajectories. Given sparse motion anchors, the model first generates dynamic depth (Shape), then renders high-fidelity RGB frames (Appearance) — bridging the gap between control signals and complex scene dynamics.

---

## Visualization

### Driving Ego-Action Control

**Turn Left**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/driving_ego_action--ours-left.mp4

**Turn Right**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/driving_ego_action--ours-right.mp4

**Speed Up**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/driving_ego_action--ours-fast.mp4

**Slow Down**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/driving_ego_action--ours-slow.mp4

### Complex Driving Scenarios

**Dangerous Cut-In**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/more_driving_scene1--ours-dangerous-cut-in-trend.mp4

**Double Cut-In**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/more_driving_scene1--ours-double-cut-in.mp4

**Right Cut-In**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/more_driving_scene1--ours-right-cut-in.mp4

**Left Cut-In & Brake**

https://github.com/Tianshuo-Xu/Motion-Forcing/releases/download/v0.1-assets/more_driving_scene1--ours-left-cut-in-and-brake.mp4

---

## TODO

- [x] Inference code
- [x] Gradio demo
- [x] Pretrained checkpoint
- [ ] Data processing pipeline (coming soon)
- [ ] Training code (coming soon)

---

## Setup

```bash
git clone --recurse-submodules https://github.com/Tianshuo-Xu/Motion-Forcing.git
cd Motion-Forcing
pip install -r requirements.txt
```

Build VGGT:

```bash
git clone git@github.com:facebookresearch/vggt.git 
cd vggt
pip install -e .
```

Download depth estimation weights:

```bash
cd Video-Depth-Anything
bash get_weights.sh

```

Download YOLO segmentation weights into `weights/yolo11l-seg.pt` (used for interactive object selection in the demo).

CogVideoX base model and the fine-tuned transformer ([`TSXu/MotionForcing_driving`](https://huggingface.co/TSXu/MotionForcing_driving)) are downloaded automatically from HuggingFace on first run.

---

## Run the Demo

```bash
python gradio_demo.py
```

Open `http://localhost:7860`. Upload an image, click objects to draw trajectories, then generate.



## Acknowledgements

We thank the authors of [CogVideoX](https://github.com/THUDM/CogVideo), [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything), [VGGT](https://huggingface.co/facebook/VGGT-1B), and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for their outstanding open-source contributions.

---

## Citation

```bibtex
@inproceedings{xu2026motionforcing,
  title     = {Motion Forcing: A Decoupled Framework for Robust Video Generation in Motion Dynamics},
  author    = {Xu, Tianshuo and Chen, Zhifei and Wu, Leyi and Lu, Hao and Chen, Ying-cong},
  booktitle = {arXiv},
  year      = {2026}
}
```
