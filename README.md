<div align="center">

# DynamicFusionNet

**Dynamic layer, view, and scale fusion for medical image segmentation**

[Paper](https://doi.org/10.1186/s40644-026-01150-5) · [Installation](#installation) · [Quick start](#quick-start) · [Citation](#citation)

</div>

PyTorch model implementation accompanying **“Dynamic layer_view_scale fusion for multimodal medical segmentation”**, published in *Cancer Imaging* (2026).

DynamicFusionNet combines dense feature fusion, multi-scale contextual refinement, global–local feature modulation, and gated skip connections in a 2D encoder–decoder. The paper evaluates segmentation across MRI, OCT, and CT images.

## Architecture

<p align="center">
  <a href="assets/architecture.png">
    <img src="assets/architecture.png" width="700" alt="DynamicFusionNet architecture from Figure 2 of the paper, including dense fusion, contextual refinement, global-local fusion, and skip attention.">
  </a>
  <br>
  <sub>Figure 2 from the paper. Click the image to view the high-resolution version.</sub>
</p>

## Qualitative results

<table>
  <tr>
    <th align="center">BraTS 2021 &amp; PARSE 2022</th>
    <th align="center">RETOUCH (OCT)</th>
    <th align="center">Grad-CAM</th>
  </tr>
  <tr>
    <td align="center" valign="top"><a href="assets/brats-parse.png"><img src="assets/brats-parse.png" width="240" alt="Figure 3: segmentation predictions and ground-truth overlays on BraTS 2021 and PARSE 2022."></a></td>
    <td align="center" valign="top"><a href="assets/retouch.png"><img src="assets/retouch.png" width="240" alt="Figure 4: qualitative retinal-fluid segmentation comparison on RETOUCH."></a></td>
    <td align="center" valign="top"><a href="assets/gradcam.png"><img src="assets/gradcam.png" width="240" alt="Figure 5: predictions and Grad-CAM visualizations on BraTS 2021, PARSE 2022, and RETOUCH."></a></td>
  </tr>
  <tr>
    <td align="center"><sub>Paper Fig. 3 · Predictions and labels</sub></td>
    <td align="center"><sub>Paper Fig. 4 · Method comparison</sub></td>
    <td align="center"><sub>Paper Fig. 5 · Activation maps</sub></td>
  </tr>
</table>

Click any preview to open the full-resolution figure. Figures are reproduced from [Ding et al., *Cancer Imaging* (2026)](https://doi.org/10.1186/s40644-026-01150-5), under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). The figure panels and labels are preserved.

## Model release

This repository provides the model in [`DynamicFusionNet.py`](DynamicFusionNet.py), its installation dependencies, and paper figures. Training pipelines, datasets, and pretrained weights are not included.

## Installation

The paper reports **Python 3.12, PyTorch 2.7.0, and CUDA 12.6**. The following setup uses those versions and installs the dependencies required by the released model.

```bash
git clone https://github.com/mh-zhou/DynamicFusionNet.git
cd DynamicFusionNet

conda create -n dynamicfusionnet python=3.12 -y
conda activate dynamicfusionnet

# NVIDIA GPU / CUDA 12.6 build
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

<details>
<summary>CPU-only installation</summary>

After creating and activating the same Python environment, use the CPU build instead:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

</details>

The model imports `thop` and `torchsummary`, so both are included in `requirements.txt`. TorchVision is not required by the released model file.

## Quick start

```python
import torch
from DynamicFusionNet import DynamicFusionNet

model = DynamicFusionNet(img_ch=1, output_ch=2).eval()
x = torch.randn(1, 1, 224, 224)

with torch.inference_mode():
    logits = model(x)

print(logits.shape)  # torch.Size([1, 2, 224, 224])
```

- **Input:** `[batch, channels, height, width]`. Set `img_ch` to the number of input channels; use spatial dimensions divisible by 16.
- **Output:** `[batch, output_ch, height, width]` logits. Apply softmax or sigmoid according to your segmentation formulation.
- This example checks the model interface with random initialization. Load your trained weights before using predictions.

A CPU forward pass with this input/output shape was checked using Python 3.14.0 and PyTorch 2.11.0+cpu. The paper-version environment above is provided separately; it was not reinstalled for that check.

## Citation

If you use DynamicFusionNet in your research or include it as a baseline in comparative experiments, please cite:

J. Ding, M. Zhou, J. Zhao, L. Hou, D. Wang, Y. Duan, and H. Zhang, “Dynamic layer_view_scale fusion for multimodal medical segmentation,” *Cancer Imaging*, 2026. doi: [10.1186/s40644-026-01150-5](https://doi.org/10.1186/s40644-026-01150-5).

```bibtex
@article{ding2026dynamicfusionnet,
  author  = {Ding, Jiao and Zhou, Meihua and Zhao, Jiarui and Hou, Long and Wang, Dingguo and Duan, Yi and Zhang, Hao},
  title   = {Dynamic layer\_view\_scale fusion for multimodal medical segmentation},
  journal = {Cancer Imaging},
  year    = {2026},
  doi     = {10.1186/s40644-026-01150-5},
  url     = {https://doi.org/10.1186/s40644-026-01150-5}
}
```
