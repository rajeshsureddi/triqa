
# TRIQA: IMAGE QUALITY ASSESSMENT BY CONTRASTIVE PRETRAINING ON ORDERED DISTORTION TRIPLETS

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

### 🌟 **ICIP 2025 Spotlight** (Double-blind):
- **arXiv**: [https://arxiv.org/pdf/2507.12687](https://arxiv.org/pdf/2507.12687)
- **IEEE Xplore**: [https://ieeexplore.ieee.org/abstract/document/11084443](https://ieeexplore.ieee.org/abstract/document/11084443)

### 🚀 Hugging Face Demo: [https://huggingface.co/spaces/S-Rajesh/triqa-iqa]

TRIQA combines content-aware and quality-aware features from ConvNeXt models to predict image quality scores on a 1-5 scale.

## Quick Start

1. **Install dependencies:**
```bash
pip install torch torchvision pillow numpy scikit-learn
```

2. **Run on an image:**
```bash
python triqa_simple.py --image sample_image/233045618.jpg
```

3. **Use in Python:**
```python
from triqa_simple import TRIQA

predictor = TRIQA()
predictor.load_models()
score = predictor.predict_quality('image.jpg')
print(f"Quality: {score:.2f}/5.0")
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA GPU (recommended)

## Model Files

Download the required model files from Drive and place them in the appropriate directories:

### Required Files:
- `feature_models/convnext_tiny_22k_224.pth` - Content-aware model (170MB)
- `feature_models/triqa_quality_aware.pth` - Quality-aware model (107MB)  
- `Regression_Models/KonIQ_scaler.save` - Feature scaler
- `Regression_Models/KonIQ_TRIQA.save` - Regression model (111MB)

### Box Links:
- [Download Model Files](https://utexas.box.com/s/8aw6axc2lofouja65uc726lca8b1cduf) - Place in `feature_models/` and `Regression_Models/` directories

## How It Works

1. **Preprocessing**: Resize image to two scales (original + half-size)
2. **Feature Extraction**: Extract content and quality features using ConvNeXt models
3. **Prediction**: Combine features and predict quality score using regression model

## Files

- `triqa_simple.py` - Main implementation
- `example.py` - Usage example
- `convnext_original.py` - Original ConvNeXt (content features)
- `convnext_finetune.py` - Fine-tuned ConvNeXt (quality features)
- `feature_models/` - Pre-trained weights
- `Regression_Models/` - Regression models

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11084443,
  author={Sureddi, Rajesh and Zadtootaghaj, Saman and Barman, Nabajeet and Bovik, Alan C.},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)}, 
  title={Triqa: Image Quality Assessment by Contrastive Pretraining on Ordered Distortion Triplets}, 
  year={2025},
  volume={},
  number={},
  pages={1744-1749},
  keywords={Image quality;Training;Deep learning;Contrastive learning;Predictive models;Feature extraction;Distortion;Data models;Synthetic data;Image Quality Assessment;Contrastive Learning},
  doi={10.1109/ICIP55913.2025.11084443}}
```



## License

MIT License
