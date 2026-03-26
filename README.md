# 🎙️ VoiceClone: Personalized TTS Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)

**VoiceClone** is a complete pipeline for fine-tuning the **SpeechT5** model on your own voice. This project provides a user-friendly interface for recording datasets, scripts for model fine-tuning, and a web-based application for real-time inference.

## ✨ Features

- **Built-in Recorder**: A Gradio-based web interface to easily record voice samples and generate a structured dataset.
- **SpeechT5 Fine-Tuning**: Leverages State-of-the-Art (SOTA) encoder-decoder models for high-quality voice cloning.
- **X-Vector Support**: Implemented speaker embeddings to capture unique vocal characteristics.
- **Interactive Inference**: A clean Gradio UI to test your digital twin immediately after training.
- **Hugging Face Integrated**: Full compatibility with `transformers`, `datasets`, and `accelerate`.

## 🛠️ Tech Stack

- **Core**: Python 3.8+
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Data Handling**: Hugging Face Datasets, Librosa, Soundfile
- **UI/UX**: Gradio
- **Optimizations**: Accelerate, TensorBoard

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/voicemodel.git
cd voicemodel
pip install -r requirements.txt
```

### 2. Prepare the Environment
Download the base models (SpeechT5 and Vocoder):
```bash
python src/download_models.py
```

### 3. Record Your Voice
Launch the recording studio:
```bash
python src/record_data.py
```
*Note: Aim for at least 50-100 high-quality sentences for better results.*

### 4. Fine-Tune the Model
Train your voice model:
```bash
python src/train.py
```

### 5. Inference (Test Your Voice)
Run the final application:
```bash
python src/app.py
```

## 📁 Project Structure

```text
├── data/               # Voice recordings and metadata (ignored by git)
├── models/             # Fine-tuned model checkpoints (ignored by git)
├── src/
│   ├── record_data.py  # Gradio app for data collection
│   ├── dataset.py      # Dataset loading and preprocessing
│   ├── train.py        # Main training script
│   └── app.py          # Inference application
├── requirements.txt    # Project dependencies
└── README.md           # You are here!
```

## 📈 Roadmap & Improvements
- [ ] Add multi-speaker support.
- [ ] Implement VITS/GPT-SoVITS for even higher fidelity.
- [ ] Support for live streaming inference.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---
*Created as part of a portfolio project for AI/ML experimentation.*
