#!/usr/bin/env python3
"""
TRIQA: Image Quality Assessment
==============================

A simplified image quality assessment tool that predicts quality scores on a 1-5 scale.
"""

import os
import torch
import numpy as np
import pickle
from PIL import Image
import argparse

from convnext_original import ConvNeXt as ConvNeXtOriginal
from convnext_finetune import ConvNeXt


def get_activation(name, activations):
    """Hook function to capture activations."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def register_hooks(model):
    """Register hooks for each layer in the model."""
    activations = {}
    for name, module in model.named_modules():
        module.register_forward_hook(get_activation(name, activations))
    return activations


class TRIQA:
    """TRIQA Image Quality Assessment"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.content_model = None
        self.quality_model = None
        self.scaler = None
        self.regression_model = None
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess_image(self, image):
        """Preprocess image for model input."""
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        return torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    
    def load_models(self):
        """Load all required models."""
        print("Loading models...")
        
        # Load content-aware model (using original ConvNeXt)
        self.content_model = ConvNeXtOriginal(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        content_state_dict = torch.load('feature_models/convnext_tiny_22k_224.pth', map_location=self.device)['model']
        content_state_dict = {k: v for k, v in content_state_dict.items() if not k.startswith('head.')}
        self.content_model.load_state_dict(content_state_dict, strict=False)
        self.content_model.to(self.device).eval()
        
        # Load quality-aware model
        self.quality_model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        quality_state_dict = torch.load('feature_models/triqa_quality_aware.pth', map_location=self.device)
        self.quality_model.load_state_dict(quality_state_dict, strict=True)
        self.quality_model.to(self.device).eval()
        
        # Register hooks for feature extraction
        self.content_activations = register_hooks(self.content_model)
        self.quality_activations = register_hooks(self.quality_model)
        
        # Load scaler and regression model
        with open('Regression_Models/KonIQ_scaler.save', 'rb') as f:
            self.scaler = pickle.load(f)
        with open('Regression_Models/KonIQ_TRIQA.save', 'rb') as f:
            self.regression_model = pickle.load(f)
        
        print("Models loaded successfully!")
    
    def predict_quality(self, image_path):
        """Predict image quality score on 1-5 scale."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_half = image.resize((image.size[0]//2, image.size[1]//2), Image.LANCZOS)
        
        img_full = self.preprocess_image(image).to(self.device)
        img_half = self.preprocess_image(image_half).to(self.device)
        
        with torch.no_grad():
            # Extract content features using hooks
            _ = self.content_model(img_full)
            content_full = self.content_activations['norm'].cpu().numpy().flatten()
            
            _ = self.content_model(img_half)
            content_half = self.content_activations['norm'].cpu().numpy().flatten()
            
            content_features = np.concatenate([content_full, content_half])
            
            # Extract quality features using hooks
            _ = self.quality_model(img_full)
            quality_full = self.quality_activations['norm'].cpu().numpy().flatten()
            
            _ = self.quality_model(img_half)
            quality_half = self.quality_activations['norm'].cpu().numpy().flatten()
            
            quality_features = np.concatenate([quality_full, quality_half])
            
            # Combine features and predict
            combined_features = np.concatenate([content_features, quality_features])
            normalized_features = self.scaler.transform(combined_features.reshape(1, -1))
            quality_score = self.regression_model.predict(normalized_features)[0]
        
        return quality_score


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='TRIQA Image Quality Assessment')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found!")
        return
    
    # Initialize and load models
    predictor = TRIQA()
    predictor.load_models()
    
    # Predict quality score
    quality_score = predictor.predict_quality(args.image)
    print(f"Quality Score: {quality_score:.2f}/5.0")


if __name__ == "__main__":
    main()