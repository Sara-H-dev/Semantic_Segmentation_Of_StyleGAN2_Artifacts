# Semantic Segmentation of StyleGAN2 Artifacts
This repository contains the implementation and training setup for **semantic segmentation of visual artifacts** in StyleGAN2-generated face images. The project is based on a **MS-UNet architecture** with a **Swin Transformer backbone**, implemented in PyTorch. 

## Repository Structure 

### dataset/
Contains functions to: - Parse **CVAT XML files** and convert them into **binary artifact masks**. > ⚠️ The **dataset itself is not included** in this repository.

### list/ 
Contains text files defining dataset splits: 
- train.txt – file paths for training data
- val.txt – file paths for validation data
- test.txt – file paths for testing data
- 
Scripts are provided to automatically split data into these lists. 

### loss/

Includes several **loss functions** for segmentation tasks. 
This project uses a **Dynamic Loss**, which combines BCE and Tversky. 

### network/ 
Implements the **MS-UNet** architecture. Unlike the original implementation, this version integrates the SwinTransformerBlock from torchvision.models.swin_transformer.

### scripts/
Contains various utility scripts: 
- CSV handling (writing and reading metric logs)
- Graph generation for training/validation curves
- Validation functions to compute validation loss and metrics (Dice, IoU, etc.)
---

## Key Scripts 

### config.yaml 
Central configuration file for: 
- Model parameters
- Dataset paths
- Training settings (learning rate, batch size, etc.)
- Loss configuration and scheduler settings

### config.py 
Stores configuration objects and utility functions for loading parameters from config.yaml. 

### train.py and trainer.py 
- train.py initializes training and calls trainer.py.
- trainer.py handles the full training loop, validation, and logging.
---
  
## Summary 
This project provides a complete training pipeline for artifact segmentation in GAN-generated images, focusing on:
- Modular dataset preparation
- Flexible configuration
- Transformer-based MS-UNet architecture
- Dynamic loss function integration
- Automated logging and visualization
