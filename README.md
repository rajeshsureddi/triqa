<<<<<<< HEAD
# TRIQA: Image Quality Assessment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

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

Download the required model files from Google Drive and place them in the appropriate directories:

### Required Files:
- `feature_models/convnext_tiny_22k_224.pth` - Content-aware model (170MB)
- `feature_models/triqa_quality_aware.pth` - Quality-aware model (107MB)  
- `Regression_Models/KonIQ_scaler.save` - Feature scaler
- `Regression_Models/KonIQ_TRIQA.save` - Regression model (111MB)

### Google Drive Links:
- [Download Model Files](https://drive.google.com/your-link-here) - Place in `feature_models/` and `Regression_Models/` directories

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

## License

MIT License
=======
# TRIQA Coming Soon
>>>>>>> 23627d3f2e6b8ac348f23aa4747288ce12b2b1f6
