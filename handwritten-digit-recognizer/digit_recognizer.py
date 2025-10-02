# digit_recognizer.py
"""
Handwritten Digit Recognition System
===================================

A comprehensive machine learning system for recognizing handwritten digits (0-9)
from uploaded images using multiple ML algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class HandwrittenDigitRecognizer:
    """
    Complete Handwritten Digit Recognition System
    
    Features:
    - Multiple ML Models (Neural Network, Random Forest)
    - Image Processing and Preprocessing
    - Model Training and Evaluation
    - Batch Processing
    - Model Persistence (Save/Load)
    - Sample Data Generation
    - Confidence Scoring
    """
    
    def __init__(self):
        """Initialize the digit recognizer system"""
        self.models = {
            'neural_network': None,
            'random_forest': None
        }
        self.scaler = StandardScaler()
        self.active_model = 'neural_network'
        self.is_trained = False
        
        print("🔢 Handwritten Digit Recognition System")
        print("=" * 50)
        print("✅ Multi-Model Architecture Ready")
        print("✅ Image Processing Pipeline Ready")  
        print("✅ Batch Processing Ready")
        print("=" * 50)
    
    def load_sample_data(self, n_samples=5000):
        """Load a sample of MNIST-like data for demonstration"""
        print("📥 Loading sample data...")
        
        # Create synthetic data that mimics MNIST structure
        np.random.seed(42)
        
        # Generate synthetic digit-like data
        X = np.random.rand(n_samples, 784) * 255
        y = np.random.randint(0, 10, n_samples)
        
        # Add some structure to make it more realistic
        for i in range(n_samples):
            digit = y[i]
            img = X[i].reshape(28, 28)
            
            # Add digit-specific patterns
            if digit == 0:  # Circle-like
                center = (14, 14)
                for x in range(28):
                    for y_coord in range(28):
                        dist = np.sqrt((x-center[0])**2 + (y_coord-center[1])**2)
                        if 8 <= dist <= 12:
                            img[x, y_coord] = 255
            elif digit == 1:  # Vertical line
                img[:, 12:16] = 255
            elif digit == 8:  # Two circles
                for center in [(10, 14), (18, 14)]:
                    for x in range(28):
                        for y_coord in range(28):
                            dist = np.sqrt((x-center[0])**2 + (y_coord-center[1])**2)
                            if 4 <= dist <= 6:
                                img[x, y_coord] = 255
            
            X[i] = img.flatten()
        
        # Normalize to [0, 1]
        X = X / 255.0
        
        print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each")
        return X, y
    
    def create_neural_network(self):
        """Create optimized neural network for digit recognition"""
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=200,
            learning_rate='adaptive',
            max_iter=200,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
        return model
    
    def create_random_forest(self):
        """Create random forest model"""
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        return model
    
    def train_models(self, X, y):
        """Train both models on the provided data"""
        print("🚀 Training Models...")
        print("-" * 30)
        
        # Split data for training and testing
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale data for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train Neural Network
        print("🧠 Training Neural Network...")
        nn_model = self.create_neural_network()
        nn_model.fit(X_train_scaled, y_train)
        nn_pred = nn_model.predict(X_test_scaled)
        nn_accuracy = accuracy_score(y_test, nn_pred)
        
        self.models['neural_network'] = nn_model
        results['neural_network'] = nn_accuracy
        
        print(f"   ✅ Neural Network Accuracy: {nn_accuracy:.4f}")
        
        # Train Random Forest
        print("🌳 Training Random Forest...")
        rf_model = self.create_random_forest()
        rf_model.fit(X_train, y_train)  # No scaling needed
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = rf_accuracy
        
        print(f"   ✅ Random Forest Accuracy: {rf_accuracy:.4f}")
        
        # Set best model as active
        best_model = 'neural_network' if nn_accuracy > rf_accuracy else 'random_forest'
        self.active_model = best_model
        
        print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()}")
        print(f"   Best Accuracy: {results[best_model]:.4f}")
        
        self.is_trained = True
        return results
    
    def preprocess_image(self, image_input):
        """Preprocess image for digit recognition"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                img = Image.open(image_input).convert('L')
                img_array = np.array(img)
            elif isinstance(image_input, np.ndarray):
                img_array = image_input.copy()
                if len(img_array.shape) == 3:
                    # Convert to grayscale
                    img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                raise ValueError("Unsupported input type")
            
            # Resize to 28x28 if needed
            if img_array.shape != (28, 28):
                img_pil = Image.fromarray(img_array.astype(np.uint8))
                img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img_pil)
            
            # Normalize
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            # Invert if needed (white background to black)
            if img_array.mean() > 0.5:
                img_array = 1 - img_array
            
            # Flatten for model
            flattened = img_array.flatten()
            
            return flattened, img_array.reshape(28, 28)
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None, None
    
    def predict_digit(self, image_input, model_type=None):
        """Predict digit from image with confidence score"""
        if not self.is_trained:
            print("❌ Models not trained. Please train first.")
            return None
        
        if model_type is None:
            model_type = self.active_model
        
        model = self.models[model_type]
        if model is None:
            print(f"❌ {model_type} model not available")
            return None
        
        # Preprocess image
        processed_img, original_shape = self.preprocess_image(image_input)
        if processed_img is None:
            return None
        
        try:
            # Scale if using neural network
            if model_type == 'neural_network':
                model_input = self.scaler.transform(processed_img.reshape(1, -1))
            else:
                model_input = processed_img.reshape(1, -1)
            
            # Predict
            prediction = model.predict(model_input)[0]
            
            # Get confidence (probability)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(model_input)[0]
                confidence = np.max(probabilities)
                all_probs = probabilities
            else:
                # Fallback for models without probability
                all_probs = np.zeros(10)
                all_probs[prediction] = 1.0
                confidence = 1.0
            
            return {
                'digit': int(prediction),
                'confidence': float(confidence),
                'probabilities': all_probs.tolist(),
                'model_used': model_type,
                'processed_image': original_shape
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def create_sample_images(self, save_dir="sample_digits"):
        """Create sample digit images for testing"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"🎨 Creating sample digit images in '{save_dir}'...")
        
        # Simple digit patterns
        patterns = {
            0: [(10,5),(18,5),(18,23),(10,23),(10,5)],
            1: [(14,5),(14,23)],
            2: [(10,8),(18,8),(18,15),(10,15),(10,23),(18,23)],
            3: [(10,8),(18,8),(18,15),(12,15),(18,15),(18,23),(10,23)],
            4: [(10,5),(10,15),(18,15),(18,5),(18,23)],
            5: [(18,8),(10,8),(10,15),(18,15),(18,23),(10,23)],
            6: [(18,8),(10,8),(10,15),(18,15),(18,23),(10,23),(10,15)],
            7: [(10,8),(18,8),(16,23)],
            8: [(14,5),(14,12),(14,23),(10,8),(18,8),(10,20),(18,20)],
            9: [(18,23),(18,8),(10,8),(10,15),(18,15)]
        }
        
        created_files = []
        
        for digit, points in patterns.items():
            # Create 28x28 image
            img = Image.new('L', (28, 28), 0)  # Black background
            draw = ImageDraw.Draw(img)
            
            # Draw the digit pattern
            if len(points) > 1:
                draw.line(points, fill=255, width=2)
            
            # Save image
            filename = os.path.join(save_dir, f"digit_{digit}.png")
            img.save(filename)
            created_files.append(filename)
        
        print(f"✅ Created {len(created_files)} sample images")
        return created_files
    
    def batch_predict(self, image_list):
        """Process multiple images and return results"""
        if not self.is_trained:
            print("❌ Models not trained. Please train first.")
            return None
        
        print(f"📊 Processing {len(image_list)} images...")
        results = []
        
        for i, img_path in enumerate(image_list):
            print(f"[{i+1:2d}/{len(image_list)}] {os.path.basename(img_path)}")
            
            result = self.predict_digit(img_path)
            if result:
                results.append({
                    'filename': os.path.basename(img_path),
                    'predicted_digit': result['digit'],
                    'confidence': result['confidence'],
                    'model_used': result['model_used']
                })
        
        return results
    
    def save_model(self, filename="digit_recognizer_model.pkl"):
        """Save the trained models and scaler"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'active_model': self.active_model,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename="digit_recognizer_model.pkl"):
        """Load previously saved models"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.active_model = model_data['active_model']
            self.is_trained = model_data['is_trained']
            
            print(f"✅ Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def demonstrate_complete_system(self):
        """Run a complete demonstration of all system capabilities"""
        print("\n🚀 COMPLETE SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        # Step 1: Load data and train models
        print("\n📚 Step 1: Loading Data and Training Models")
        X, y = self.load_sample_data(n_samples=2000)
        training_results = self.train_models(X, y)
        
        # Step 2: Create sample images
        print(f"\n🎨 Step 2: Creating Sample Test Images")
        sample_images = self.create_sample_images()
        
        # Step 3: Test individual predictions
        print(f"\n🔍 Step 3: Individual Image Predictions")
        print("-" * 40)
        
        for img_path in sample_images[:5]:
            result = self.predict_digit(img_path)
            if result:
                filename = os.path.basename(img_path)
                expected = int(filename.split('_')[1].split('.')[0])
                predicted = result['digit']
                confidence = result['confidence']
                
                status = "✅" if predicted == expected else "❌"
                print(f"{status} {filename}: Expected {expected}, "
                      f"Got {predicted} ({confidence:.1%} confidence)")
        
        # Step 4: Batch processing
        print(f"\n📊 Step 4: Batch Processing Results")
        batch_results = self.batch_predict(sample_images)
        
        if batch_results:
            correct = sum(1 for r in batch_results 
                         if r['predicted_digit'] == int(r['filename'].split('_')[1].split('.')[0]))
            accuracy = correct / len(batch_results)
            
            print(f"\n🎯 Batch Processing Summary:")
            print(f"   Total Images: {len(batch_results)}")
            print(f"   Correct Predictions: {correct}")
            print(f"   Accuracy: {accuracy:.1%}")
            print(f"   Active Model: {self.active_model.replace('_', ' ').title()}")
        
        # Step 5: Save model
        print(f"\n💾 Step 5: Saving Trained Model")
        self.save_model()
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("✅ All System Features Demonstrated:")
        print("   • Data Loading and Preprocessing")
        print("   • Multi-Model Training (Neural Network + Random Forest)")
        print("   • Model Selection and Evaluation")
        print("   • Image Processing and Recognition")
        print("   • Individual and Batch Predictions")
        print("   • Confidence Scoring")
        print("   • Model Persistence")
        print("=" * 60)
        
        return {
            'training_results': training_results,
            'batch_results': batch_results,
            'sample_images': sample_images
        }
