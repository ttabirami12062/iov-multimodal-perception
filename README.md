# Multi-Modal Visual Features Perception System for Internet of Vehicles (IoV)

> Published at IEEE ESCI 2024 — International Conference on Emerging Smart Computing and Informatics, Pune, India (Mar 5–7, 2024)

## Overview

This project implements a multi-modal visual perception pipeline for autonomous vehicles in the Internet of Vehicles (IoV) framework. Instead of relying on a single sensor input, the system fuses data from LiDAR, Radar, Ultrasonic, and Vision cameras to achieve robust, real-time object detection and scene understanding — even under variable environmental conditions.

The core insight: single-modal methods miss context that multi-sensor fusion captures. This system combines per-sensor CNN feature extraction with an R-CNN + SPPNet fusion layer to produce accurate, reliable detections across all input types.

## Published Paper

**Multi-Modal Visual Features Perception Technology for Internet of Vehicles (IoV)**  
Rakesh Kumar Mahendran, Abirami Thiyagarajan, Angelin Gracia A, Kumar P  
Department of Computer Science and Engineering, Rajalakshmi Engineering College, Chennai, India  
IEEE ESCI 2024 | DOI: [10.1109/ESCI59607.2024.10497246](https://doi.org/10.1109/ESCI59607.2024.10497246)

## Architecture

```
LiDAR Data        Radar Data        Ultrasonic Data     Vision Camera Data
     |                 |                   |                    |
CNN Feature      CNN Feature         CNN Feature          CNN Feature
Extraction       Extraction          Extraction           Extraction
     |                 |                   |                    |
     └─────────────────┴───────────────────┴────────────────────┘
                                   |
                            Fusion Layer
                                   |
                        SPPNet for Sensor Fusion
                                   |
                            Output Layer
                        (Object Detection + Scene Classification)
```

## Pipeline Modules

**1. Data Collection**  
Sensor data is acquired from LiDAR, Radar, Ultrasonic, and Vision cameras. Each sensor captures a different aspect of the environment — distance, velocity, depth, and visual context.

**2. Preprocessing and Alignment**  
Raw sensor inputs are preprocessed and aligned to ensure format consistency and compatibility across modalities before feature extraction.

**3. Feature Extraction**  
Per-sensor CNN feature extraction isolates the unique characteristics of each input stream. SPPNet (Spatial Pyramid Pooling Network) is then applied to generate fixed-size representations regardless of input size or resolution differences.

**4. Multimodal Feature Fusion**  
Extracted features from all sensors are merged into a single unified representation — combining the environmental picture each sensor sees into one complete view.

**5. Model Training (R-CNN)**  
A Region-based Convolutional Neural Network (R-CNN) is trained on the fused multimodal dataset. Network parameters are iteratively adjusted to optimize object detection accuracy across all sensor inputs.

**6. Validation and Evaluation**  
The system is evaluated using precision, recall, and precision-recall curves on validation datasets, followed by real-world testing to confirm robustness under varied conditions.

## Models Used

- **R-CNN** — Region-based CNN for object detection on fused multimodal feature maps
- **SPPNet** — Spatial Pyramid Pooling Network for fixed-size feature representation across variable input sizes
- **YOLOv8** — Used in the GUI demo for real-time inference (segmentation + detection side-by-side)

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Tkinter (GUI)
- CNN, R-CNN, SPPNet

## GUI Demo

The project includes a desktop GUI (`gui.py`) that runs dual-model inference on any road image:

- **Left panel** — Segmentation model output with pixel-level masks
- **Right panel** — YOLOv8 detection with bounding boxes and confidence scores

**To run:**

```bash
# Install dependencies
pip install ultralytics ttkthemes pillow opencv-python torch torchvision

# Launch GUI
python gui.py
```

Load any road/traffic image and click "Detect Objects" to see both models run simultaneously.

## Results

The multi-modal fusion approach demonstrated a **25% improvement in object detection performance** over single-modal baselines, measured across precision and recall metrics on real-world and validation datasets.

## Repository Structure

```
├── gui.py                  # Desktop GUI — dual model inference
├── gui_combined.py         # Combined GUI variant
├── object.py               # Object detection script
├── predict.py              # Inference script
├── train.py                # Model training script
├── data.yaml               # Dataset configuration
├── segment/                # Segmentation training runs
├── train/                  # Training labels
├── test/                   # Test labels
├── valid/                  # Validation labels
└── road-3186188_640.jpg    # Sample test image
```

## Authors

- **Abirami Thiyagarajan** — [GitHub](https://github.com/ttabirami12062) | [Portfolio](https://ttabirami12062.github.io)
- Angelin Gracia A

Rajalakshmi Engineering College, Chennai, India
