# SketchyGAN — Sketch to Image Synthesis

A PyTorch replication and extension of [SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis](https://arxiv.org/abs/1801.02753) (Chen & Hays, CVPR 2018). This project implements a GAN-based pipeline that synthesizes realistic images from freehand sketches across multiple object categories.

---

## Overview

Sketch-to-image synthesis is a challenging image translation task because sketches are sparse, abstract, and lack color and texture — yet must map to photorealistic outputs. SketchyGAN addresses this by introducing **Masked Residual Units (MRUs)**, a novel network block that injects the input sketch at multiple scales throughout both the generator and discriminator, improving structural fidelity and diversity of outputs.

This repository includes:
- Full replication of the SketchyGAN generator and discriminator using MRU blocks
- Data preprocessing and augmentation pipeline for paired sketch-image datasets
- Inception V4 feature extraction for perceptual loss computation
- Single-category training mode (`src_single`) for rapid experimentation
- Jupyter notebooks for replication experiments and balloon-category generation

---

## Architecture

### Masked Residual Unit (MRU)
The core building block of SketchyGAN. Unlike standard ResNet blocks, MRUs take **two inputs**: a feature map from the previous layer and the original input sketch (resized to the current scale). A learned internal mask selects which sketch features to inject, enabling the network to preserve structural detail from the sketch at every resolution.

```
Input Sketch (resized) ──┐
                          ▼
Feature Map ──► [Conv] ──► [Mask Gate] ──► Output Feature Map
```

### Generator
- Encoder-decoder architecture built entirely from MRU blocks
- Skip connections between encoder and decoder stages (similar to U-Net)
- Input sketch is fed into every MRU block along the forward path
- Outputs a photorealistic RGB image at the same resolution as the input sketch

### Discriminator
- Also built with MRU blocks (sketch-conditioned)
- Takes both the generated/real image and the input sketch as inputs
- Penalizes outputs that are realistic but do not match the sketch structure

### Loss Function
Multi-term objective combining:
- **Adversarial loss** — standard GAN minimax objective
- **Perceptual loss** — feature-level L2 distance using Inception V4 activations
- **Reconstruction loss** — pixel-level L1 loss for structural consistency

---

## Repository Structure

```
sketch-Image/
├── data_processing/          # Paired sketch-image dataset loading and augmentation
├── inception_v4_model/       # Inception V4 feature extractor for perceptual loss
├── src_single/               # Single-category training pipeline
├── main_single.py            # Training entry point (single category)
├── SketchyGAN_replication.ipynb   # Full replication notebook
├── sketchygan_balloon.ipynb       # Balloon category generation experiment
└── README.md
```

---

## Setup

### Requirements
```bash
pip install torch torchvision numpy pillow matplotlib
```

### Dataset
This project uses paired sketch-image data. You can use:
- The [Sketchy Dataset](http://sketchy.eye.gatech.edu/) (75,471 sketches, 125 categories)
- Custom paired sketch-image folders in the format expected by `data_processing/`

### Training (Single Category)
```bash
python main_single.py --category <category_name> --epochs 100 --batch_size 16
```

---

## Results

The model was trained and evaluated on single-category sketch-image pairs. The generator successfully learns to synthesize plausible object images from freehand sketches, preserving pose and structural layout from the input while generating realistic texture and color.

> Qualitative results and generated samples are available in `SketchyGAN_replication.ipynb`.

---

## Key Implementation Details

- **MRU blocks** are implemented from scratch in PyTorch following the original paper specification
- **Inception V4** is used as a frozen feature extractor for perceptual loss — weights are not updated during GAN training
- Single-category mode (`src_single`) reduces training complexity and allows faster iteration on individual object classes
- Data augmentation includes random horizontal flipping and sketch edge perturbation to improve generalization

---

## Reference

```
@inproceedings{chen2018sketchygan,
  title={SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis},
  author={Chen, Wengling and Hays, James},
  booktitle={CVPR},
  year={2018}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
