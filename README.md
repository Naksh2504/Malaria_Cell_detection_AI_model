# Malaria Detection AI Platform

An automated deep learning screening system that detects *Plasmodium* parasites in microscopic thin blood smear cell images with **95.81% validation accuracy**.

Powered by a fine-tuned MobileNetV2 Convolutional Neural Network (CNN), the platform classifies cell micrographs as either **Parasitized** or **Uninfected**, providing calibrated probability scoring and diagnostic metrics.

---

## Key Features

- **High Accuracy**: Achieves 95.81% validation accuracy on thin blood smear test partitions.
- **MobileNetV2 Architecture**: Leverages pre-trained ImageNet representations for lightweight, high-speed inference (< 45 ms per specimen).
- **Two-Phase Transfer Learning**: First trains custom classification dense layers, followed by fine-tuning of deeper convolutional layers.
- **Data Augmentation**: Robust generalization trained with random horizontal/vertical flips and rotations.
- **Interactive Web Interface**: Streamlined browser-based diagnostic station with drag-and-drop support, clipboard paste, real specimen tests, and printable clinical summaries.

---

## Model Architecture & Training

1. **Base Network**: MobileNetV2 (pretrained on ImageNet)
2. **Input Dimensions**: 128 &times; 128 &times; 3 RGB
3. **Normalization**: Rescaled to `[-1.0, 1.0]` tensor range
4. **Classifier Head**:
   - Global Average Pooling 2D
   - Dropout (0.2)
   - Dense Sigmoid binary output
5. **Training Methodology**:
   - **Phase 1 (Feature Extraction)**: Base layers frozen; Adam optimizer ($\text{lr} = 10^{-3}$) for 15 epochs.
   - **Phase 2 (Fine-Tuning)**: Unfreezing layers from index 100 onwards with reduced learning rate ($\text{lr} = 10^{-5}$) for 15 additional epochs.

---

## Quickstart

### 1. Installation

Ensure Python 3.9+ is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Start the Flask server:

```bash
python app.py
```

Or on Windows:
```bash
run.bat
```

### 3. Open in Browser

Navigate to:
```
http://127.0.0.1:5000
```

---

## Project Structure

```text
├── app.py              # Flask server and inference API
├── index.html          # Web frontend UI
├── train.py            # Model training & fine-tuning script
├── malaria_model.h5    # Pre-trained MobileNetV2 weights (9.4 MB)
├── requirements.txt    # Python package dependencies
├── run.bat             # Windows startup script
├── samples/            # Real reference blood smear test images
└── README.md           # Documentation
```
