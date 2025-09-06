#!/usr/bin/env python3
"""TRIQA Example Usage"""

import os
from triqa_simple import TRIQA

def main():
    predictor = TRIQA()
    predictor.load_models()
    
    sample_image = "sample_image/25239707.jpg"
    if os.path.exists(sample_image):
        score = predictor.predict_quality(sample_image)
        print(f"Quality Score: {score:.2f}/5.0")
    else:
        print("Sample image not found")

if __name__ == "__main__":
    main()