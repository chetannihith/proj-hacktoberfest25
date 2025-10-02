# Handwritten Digit Recognizer

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![PIL](https://img.shields.io/badge/PIL-8.0+-green.svg)](https://pillow.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning system for recognizing handwritten digits (0-9) from uploaded images. This project implements multiple ML algorithms with advanced image preprocessing capabilities and provides both individual and batch processing functionality.

## 🚀 Features

- **Multi-Model Architecture**: Neural Network (MLPClassifier) and Random Forest models
- **Advanced Image Processing**: Automatic resizing, grayscale conversion, and normalization
- **Batch Processing**: Process multiple images simultaneously
- **Confidence Scoring**: Get prediction confidence percentages
- **Model Persistence**: Save and load trained models
- **Sample Data Generation**: Create test digit images
- **Multiple Input Formats**: Supports PNG, JPG, JPEG, BMP, TIFF formats
- **Comprehensive Evaluation**: Detailed accuracy metrics and confusion matrices

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies
pip install numpy matplotlib scikit-learn pillow


### Alternative: Using requirements.txt
pip install -r requirements.txt


**requirements.txt:**
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
pillow>=8.0.0


## 📁 Project Structure
handwritten-digit-recognizer/
│
├── digit_recognizer.py # Main application file
├── digit_recognizer_model.pkl # Saved trained models
├── sample_digits/ # Generated sample images
│ ├── digit_0.png
│ ├── digit_1.png
│ ├── ...
│ └── digit_9.png
├── README.md # Project documentation
└── requirements.txt # Python dependencies


## 🚀 Quick Start

### Basic Usage
Initialize the recognizer
from digit_recognizer import HandwrittenDigitRecognizer

recognizer = HandwrittenDigitRecognizer()

Load sample data and train models
X, y = recognizer.load_sample_data(n_samples=5000)
training_results = recognizer.train_models(X, y)

Predict from an uploaded image
result = recognizer.predict_digit('path/to/your/digit_image.png')
print(f"Predicted Digit: {result['digit']}")
print(f"Confidence: {result['confidence']:.1%}")

Process multiple images
image_files = ['digit1.png', 'digit2.png', 'digit3.png']
batch_results = recognizer.batch_predict(image_files)


### Running the Complete Demo
Run the complete system demonstration
recognizer = HandwrittenDigitRecognizer()
demo_results = recognizer.demonstrate_complete_system()


## 📊 Model Performance

The system trains and compares two models:

| Model | Architecture | Typical Accuracy |
|-------|-------------|------------------|
| Neural Network | 3-layer MLP (256-128-64) | 85-95% |
| Random Forest | 50 estimators, max_depth=15 | 80-90% |

## 🖼️ Image Processing Pipeline

1. **Format Detection**: Automatically handles various image formats
2. **Resizing**: Converts images to 28x28 pixels (MNIST standard)
3. **Grayscale Conversion**: Ensures single-channel input
4. **Normalization**: Scales pixel values to [0,1] range
5. **Inversion**: Handles white-on-black vs black-on-white images
6. **Preprocessing**: Applies noise reduction and enhancement

## 💻 API Reference

### Core Methods

#### `predict_digit(image_input, model_type=None)`
Predicts digit from image input.

**Parameters:**
- `image_input`: File path (string) or numpy array
- `model_type`: 'neural_network' or 'random_forest' (optional)

**Returns:**
{
'digit': int, # Predicted digit (0-9)
'confidence': float, # Confidence score (0-1)
'probabilities': list, # Probability for each digit
'model_used': str, # Model used for prediction
'processed_image': array # Preprocessed 28x28 image
}


#### `batch_predict(image_list)`
Process multiple images simultaneously.

#### `train_models(X, y)`
Train both neural network and random forest models.

## 🔧 Configuration

### Model Parameters

**Neural Network:**
- Hidden layers: (256, 128, 64)
- Activation: ReLU
- Solver: Adam
- Learning rate: Adaptive
- Early stopping: Enabled

**Random Forest:**
- Estimators: 50
- Max depth: 15
- Min samples split: 5

## 📈 Usage Examples

### Single Image Prediction

Load and predict from file
result = recognizer.predict_digit('handwritten_digit.png')
if result:
print(f"Digit: {result['digit']} (Confidence: {result['confidence']:.2%})")


### Batch Processing

Process folder of images
import os
image_folder = "test_images/"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
if f.endswith(('.png', '.jpg', '.jpeg'))]

results = recognizer.batch_predict(image_files)


### Model Persistence

Save trained model
recognizer.save_model('my_digit_model.pkl')

Load existing model
new_recognizer = HandwrittenDigitRecognizer()
new_recognizer.load_model('my_digit_model.pkl')


## 🧪 Testing

### Create Sample Images

Generate test digit images
sample_files = recognizer.create_sample_images('test_digits/')

Test recognition accuracy
for img_path in sample_files:
result = recognizer.predict_digit(img_path)
expected = int(os.path.basename(img_path).split('_').split('.'))
actual = result['digit']
print(f"Expected: {expected}, Got: {actual}")


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- MNIST dataset for training data reference
- scikit-learn community for ML algorithms
- PIL/Pillow for image processing capabilities

## 🔮 Future Enhancements

- [ ] CNN implementation with TensorFlow/PyTorch
- [ ] Real-time webcam digit recognition
- [ ] Web interface with Flask/Django
- [ ] Mobile app integration
- [ ] Support for multiple digits in single image
- [ ] Data augmentation techniques

## 📊 Project Status

- ✅ Core ML pipeline complete
- ✅ Image processing implemented
- ✅ Batch processing functional
- ✅ Model persistence working
- ✅ Documentation complete

---

**Made with ❤️ by Bhavishy Agrawal**