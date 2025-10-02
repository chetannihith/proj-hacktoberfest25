#!/usr/bin/env python3
"""
Handwritten Digit Recognizer - Main Application
==============================================

A comprehensive machine learning system for recognizing handwritten digits (0-9)
from uploaded images with GUI interface and command-line support.

Author: AI Assistant
Date: October 2025
License: MIT
"""

import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import threading

# Import our digit recognizer class
from digit_recognizer import HandwrittenDigitRecognizer

class DigitRecognizerGUI:
    """
    GUI Application for Handwritten Digit Recognition
    """
    
    def __init__(self):
        self.recognizer = HandwrittenDigitRecognizer()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Main title
        title_label = tk.Label(
            self.root, 
            text="🔢 Handwritten Digit Recognition System", 
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Create main frames
        self.create_control_frame()
        self.create_display_frame()
        self.create_status_frame()
        
    def create_control_frame(self):
        """Create control buttons frame"""
        control_frame = tk.LabelFrame(
            self.root, 
            text="Controls", 
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#34495e'
        )
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Model training section
        model_frame = tk.Frame(control_frame, bg='#f0f0f0')
        model_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(
            model_frame,
            text="🚀 Train Models",
            command=self.train_models_threaded,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            model_frame,
            text="📁 Load Model",
            command=self.load_model,
            bg='#2ecc71',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Image processing section
        image_frame = tk.Frame(control_frame, bg='#f0f0f0')
        image_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(
            image_frame,
            text="📤 Upload Image",
            command=self.upload_image,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            image_frame,
            text="📂 Batch Process",
            command=self.batch_process,
            bg='#f39c12',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            image_frame,
            text="🎨 Create Samples",
            command=self.create_samples,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side=tk.LEFT, padx=5)
        
    def create_display_frame(self):
        """Create image and results display frame"""
        display_frame = tk.LabelFrame(
            self.root,
            text="Results",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#34495e'
        )
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Image display
        self.image_label = tk.Label(
            display_frame,
            text="Upload an image to get started",
            bg='white',
            relief=tk.SUNKEN,
            width=40,
            height=15
        )
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results text area
        results_frame = tk.Frame(display_frame, bg='#f0f0f0')
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(
            results_frame,
            height=20,
            width=40,
            font=('Courier', 10),
            bg='#ecf0f1'
        )
        
        scrollbar = tk.Scrollbar(results_frame, orient=tk.VERTICAL)
        scrollbar.config(command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_status_frame(self):
        """Create status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please train or load a model first")
        
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#34495e',
            fg='white',
            font=('Arial', 9)
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def update_results(self, text):
        """Update results text area"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        
    def train_models_threaded(self):
        """Train models in a separate thread to avoid GUI freezing"""
        def train_thread():
            self.status_var.set("Training models... This may take a few minutes")
            self.root.update()
            
            try:
                X, y = self.recognizer.load_sample_data(n_samples=3000)
                results = self.recognizer.train_models(X, y)
                
                result_text = "🎉 Model Training Complete!\n"
                result_text += "=" * 40 + "\n"
                for model_name, accuracy in results.items():
                    result_text += f"✅ {model_name.replace('_', ' ').title()}: {accuracy:.4f}\n"
                result_text += "=" * 40 + "\n"
                result_text += f"🏆 Active Model: {self.recognizer.active_model.replace('_', ' ').title()}\n"
                
                self.update_results(result_text)
                self.status_var.set("Models trained successfully! Ready for predictions")
                
            except Exception as e:
                messagebox.showerror("Training Error", f"Error during training: {str(e)}")
                self.status_var.set("Training failed")
                
        threading.Thread(target=train_thread, daemon=True).start()
        
    def load_model(self):
        """Load a pre-trained model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            success = self.recognizer.load_model(file_path)
            if success:
                self.status_var.set("Model loaded successfully!")
                self.update_results("✅ Model loaded and ready for predictions!")
            else:
                messagebox.showerror("Load Error", "Failed to load model")
                
    def upload_image(self):
        """Upload and process a single image"""
        if not self.recognizer.is_trained:
            messagebox.showwarning("No Model", "Please train or load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Display image
                img = Image.open(file_path)
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                # Predict digit
                result = self.recognizer.predict_digit(file_path)
                
                if result:
                    result_text = f"🔍 Image Analysis Results\n"
                    result_text += "=" * 40 + "\n"
                    result_text += f"📁 File: {os.path.basename(file_path)}\n"
                    result_text += f"🎯 Predicted Digit: {result['digit']}\n"
                    result_text += f"📊 Confidence: {result['confidence']:.2%}\n"
                    result_text += f"🤖 Model: {result['model_used'].replace('_', ' ').title()}\n\n"
                    
                    result_text += "📈 All Probabilities:\n"
                    result_text += "-" * 20 + "\n"
                    
                    for digit, prob in enumerate(result['probabilities']):
                        marker = "👉" if digit == result['digit'] else "  "
                        result_text += f"{marker} Digit {digit}: {prob:.4f} ({prob*100:.1f}%)\n"
                    
                    self.update_results(result_text)
                    self.status_var.set(f"Prediction: {result['digit']} ({result['confidence']:.1%} confidence)")
                else:
                    messagebox.showerror("Processing Error", "Failed to process image")
                    
            except Exception as e:
                messagebox.showerror("Upload Error", f"Error processing image: {str(e)}")
                
    def batch_process(self):
        """Process multiple images in a folder"""
        if not self.recognizer.is_trained:
            messagebox.showwarning("No Model", "Please train or load a model first!")
            return
            
        folder_path = filedialog.askdirectory(title="Select folder with digit images")
        
        if folder_path:
            try:
                # Get image files
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                image_files = []
                
                for file in os.listdir(folder_path):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(folder_path, file))
                
                if not image_files:
                    messagebox.showwarning("No Images", "No image files found in selected folder")
                    return
                
                self.status_var.set(f"Processing {len(image_files)} images...")
                self.root.update()
                
                # Process images
                results = self.recognizer.batch_predict(image_files)
                
                if results:
                    result_text = f"📊 Batch Processing Results\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"📁 Folder: {os.path.basename(folder_path)}\n"
                    result_text += f"📸 Total Images: {len(image_files)}\n"
                    result_text += f"✅ Successfully Processed: {len(results)}\n\n"
                    
                    result_text += "Results:\n"
                    result_text += "-" * 30 + "\n"
                    
                    for result in results:
                        conf_bar = "█" * max(1, int(result['confidence'] * 10))
                        result_text += f"{result['filename']:15} → {result['predicted_digit']} "
                        result_text += f"({result['confidence']:.1%}) {conf_bar}\n"
                    
                    self.update_results(result_text)
                    self.status_var.set(f"Processed {len(results)} images successfully!")
                    
            except Exception as e:
                messagebox.showerror("Batch Error", f"Error during batch processing: {str(e)}")
                
    def create_samples(self):
        """Create sample digit images"""
        try:
            sample_files = self.recognizer.create_sample_images()
            
            result_text = f"🎨 Sample Images Created\n"
            result_text += "=" * 30 + "\n"
            result_text += f"📁 Location: sample_digits/\n"
            result_text += f"📸 Files created: {len(sample_files)}\n\n"
            result_text += "Files:\n"
            result_text += "-" * 20 + "\n"
            
            for file_path in sample_files:
                result_text += f"✅ {os.path.basename(file_path)}\n"
            
            self.update_results(result_text)
            self.status_var.set("Sample images created successfully!")
            
        except Exception as e:
            messagebox.showerror("Sample Creation Error", f"Error creating samples: {str(e)}")
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def command_line_interface():
    """Command line interface for the digit recognizer"""
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognizer")
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--predict", type=str, help="Predict digit from image file")
    parser.add_argument("--batch", type=str, help="Process all images in folder")
    parser.add_argument("--model", type=str, help="Model file to load")
    parser.add_argument("--samples", action="store_true", help="Create sample images")
    parser.add_argument("--demo", action="store_true", help="Run complete demonstration")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    
    args = parser.parse_args()
    
    # Create recognizer instance
    recognizer = HandwrittenDigitRecognizer()
    
    # Load model if specified
    if args.model:
        print(f"Loading model: {args.model}")
        if not recognizer.load_model(args.model):
            print("Failed to load model")
            return
    
    # Train models if requested
    if args.train:
        print("Training models...")
        X, y = recognizer.load_sample_data(n_samples=5000)
        results = recognizer.train_models(X, y)
        print("Training complete!")
        
    # Create sample images if requested
    if args.samples:
        print("Creating sample images...")
        sample_files = recognizer.create_sample_images()
        print(f"Created {len(sample_files)} sample images")
        
    # Single image prediction
    if args.predict:
        if not recognizer.is_trained:
            print("Error: No trained model available. Use --train or --model first.")
            return
            
        result = recognizer.predict_digit(args.predict)
        if result:
            print(f"Image: {args.predict}")
            print(f"Predicted Digit: {result['digit']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Model Used: {result['model_used'].replace('_', ' ').title()}")
        else:
            print("Failed to process image")
            
    # Batch processing
    if args.batch:
        if not recognizer.is_trained:
            print("Error: No trained model available. Use --train or --model first.")
            return
            
        if not os.path.isdir(args.batch):
            print(f"Error: Directory {args.batch} not found")
            return
            
        # Get image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(args.batch):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(args.batch, file))
        
        if not image_files:
            print("No image files found in directory")
            return
            
        results = recognizer.batch_predict(image_files)
        print(f"Processed {len(results)} images:")
        for result in results:
            print(f"  {result['filename']}: {result['predicted_digit']} "
                  f"({result['confidence']:.1%})")
    
    # Run complete demonstration
    if args.demo:
        print("Running complete system demonstration...")
        recognizer.demonstrate_complete_system()
        
    # Launch GUI
    if args.gui or len(sys.argv) == 1:
        print("Launching GUI interface...")
        app = DigitRecognizerGUI()
        app.run()

def main():
    """Main entry point"""
    print("🔢 Handwritten Digit Recognizer")
    print("=" * 50)
    print("Multi-Model ML System for Digit Recognition")
    print("=" * 50)
    
    try:
        command_line_interface()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
