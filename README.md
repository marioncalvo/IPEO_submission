# IPEO Project - Hurricane Damage Classification

## Project Description
This project aims to classify satellite images to detect hurricane damage using deep learning. We trained and compared two convolutional neural network models (ResNet34 and MobileNetV3) and further improved the best model (ResNet34) using data augmentation and probability calibration techniques.

## Project Structure
```
├── ipeo_code.ipynb              # Main notebook: data loading, model training, calibration
├── inference.ipynb              # Inference notebook: predictions on test images
├── data_utils.py                # Utility functions for data normalization
├── environment.yml              # Python dependencies
├── README.md                    # This file
├── models/                      # Trained model weights and statistics
│   ├── resnet_final.pth        # Best ResNet34 model (with data augmentation)
│   ├── resnet_test.pth         # ResNet34 model (baseline)
│   ├── mobilenet_test.pth      # MobileNetV3 model (baseline)
│   ├── stats_final.json        # Training statistics for ResNet34
│   └── stats_test.json         # Training statistics for baseline models
├── test_sample/                 # Sample images for inference
└── ipeo_hurricane_for_students/ # Dataset directory
    ├── train/
    ├── validation/
    └── test/
```
