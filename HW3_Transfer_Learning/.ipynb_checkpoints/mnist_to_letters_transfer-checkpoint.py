#!/usr/bin/env python3
"""
MNIST to Letters Transfer Learning

This script demonstrates transfer learning from MNIST digit classification (0-9)
to letter classification (A, B, C, D, E) using TensorFlow.

The approach:
1. Train a CNN on full MNIST dataset (digits 0-9)
2. Remove final classification layer
3. Add new classification layer for letters (A, B, C, D, E)
4. Fine-tune on letter dataset

Dataset structure:
- Images/A/, Images/B/, Images/C/, Images/D/, Images/E/: Letter images
- Images/NotA/, Images/NotB/, etc.: Non-target letter images
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class MNISTToLettersTransfer:
    def __init__(self, data_dir="Images", img_size=(28, 28)):
        """
        Initialize the transfer learning pipeline.
        
        Args:
            data_dir (str): Directory containing letter images
            img_size (tuple): Target image size (28x28 to match MNIST)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.letters = ['A', 'B', 'C', 'D', 'E']
        self.num_classes = len(self.letters)
        
        # Models
        self.mnist_model = None
        self.base_model = None
        self.transfer_model = None
        
        # Data
        self.letter_data = None
        self.letter_labels = None
        
    def create_mnist_model(self):
        """
        Create and train CNN model on MNIST dataset.
        
        Returns:
            tf.keras.Model: Trained MNIST model
        """
        print("=" * 60)
        print("STEP 1: TRAINING CNN ON MNIST DATASET")
        print("=" * 60)
        
        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to add channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
        y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
        
        print(f"MNIST Training data shape: {x_train.shape}")
        print(f"MNIST Training labels shape: {y_train_cat.shape}")
        
        # Create CNN model for MNIST
        model = tf.keras.Sequential([
            # Convolutional layers
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax', name='mnist_output')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        # Train model
        print("\nTraining MNIST model...")
        history = model.fit(
            x_train, y_train_cat,
            batch_size=128,
            epochs=10,
            validation_data=(x_test, y_test_cat),
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
        print(f"\nMNIST Test Accuracy: {test_accuracy:.4f}")
        
        self.mnist_model = model
        return model
    
    def load_letter_dataset(self):
        """
        Load and preprocess letter dataset from Images folder.
        
        Returns:
            tuple: (images, labels) arrays
        """
        print("\n" + "=" * 60)
        print("STEP 2: LOADING LETTER DATASET")
        print("=" * 60)
        
        images = []
        labels = []
        
        # Load positive examples for each letter
        for i, letter in enumerate(self.letters):
            letter_dir = os.path.join(self.data_dir, letter)
            print(f"Loading {letter} images from {letter_dir}...")
            
            if os.path.exists(letter_dir):
                for img_file in os.listdir(letter_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(letter_dir, img_file)
                        
                        # Load and preprocess image
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Resize to 28x28 to match MNIST
                            img = cv2.resize(img, self.img_size)
                            # Normalize
                            img = img.astype('float32') / 255.0
                            
                            images.append(img)
                            labels.append(i)  # 0=A, 1=B, 2=C, 3=D, 4=E
            
            print(f"  Loaded {sum(1 for l in labels if l == i)} {letter} images")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Reshape images to add channel dimension
        images = images.reshape(-1, 28, 28, 1)
        
        print(f"\nTotal letter dataset:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label distribution: {np.bincount(labels)}")
        
        self.letter_data = images
        self.letter_labels = labels
        
        return images, labels
    
    def create_base_model(self):
        """
        Create base model from trained MNIST model (without final layer).
        
        Returns:
            tf.keras.Model: Base model for transfer learning
        """
        print("\n" + "=" * 60)
        print("STEP 3: CREATING BASE MODEL FOR TRANSFER LEARNING")
        print("=" * 60)
        
        if self.mnist_model is None:
            raise ValueError("MNIST model must be trained first")
        
        # Create base model by removing the final classification layer
        base_model = tf.keras.Model(
            inputs=self.mnist_model.input,
            outputs=self.mnist_model.layers[-3].output  # Output before final Dense layer
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        print("Base model created from MNIST features:")
        base_model.summary()
        
        self.base_model = base_model
        return base_model
    
    def create_transfer_model(self, base_model):
        """
        Create transfer learning model for letter classification.
        
        Args:
            base_model: Pre-trained base model
            
        Returns:
            tf.keras.Model: Transfer learning model
        """
        print("\n" + "=" * 60)
        print("STEP 4: CREATING TRANSFER LEARNING MODEL")
        print("=" * 60)
        
        # Create new model with letter classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax', name='letter_output')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Transfer learning model created:")
        model.summary()
        
        self.transfer_model = model
        return model
    
    def train_transfer_model(self, fine_tune=True):
        """
        Train the transfer learning model.
        
        Args:
            fine_tune (bool): Whether to fine-tune base model layers
        """
        print("\n" + "=" * 60)
        print("STEP 5: TRAINING TRANSFER LEARNING MODEL")
        print("=" * 60)
        
        if self.letter_data is None:
            self.load_letter_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.letter_data, self.letter_labels,
            test_size=0.3, random_state=42, stratify=self.letter_labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Phase 1: Train with frozen base model
        print("\nPhase 1: Training with frozen base model...")
        history1 = self.transfer_model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=10,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen base model
        if fine_tune:
            print("\nPhase 2: Fine-tuning with unfrozen base model...")
            
            # Unfreeze base model
            self.base_model.trainable = True
            
            # Use lower learning rate for fine-tuning
            self.transfer_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history2 = self.transfer_model.fit(
                X_train, y_train,
                batch_size=16,
                epochs=10,
                validation_data=(X_test, y_test),
                verbose=1
            )
        
        # Final evaluation
        test_loss, test_accuracy = self.transfer_model.evaluate(X_test, y_test, verbose=0)
        print(f"\nFinal Transfer Learning Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed evaluation
        self.evaluate_model(X_test, y_test)
        
        return self.transfer_model
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test images
            y_test: Test labels
        """
        print("\n" + "=" * 60)
        print("DETAILED MODEL EVALUATION")
        print("=" * 60)
        
        # Predictions
        predictions = self.transfer_model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Classification report
        letter_names = [f"Letter_{letter}" for letter in self.letters]
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_classes, target_names=letter_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.letters, yticklabels=self.letters)
        plt.title('Confusion Matrix - Transfer Learning Results')
        plt.xlabel('Predicted Letter')
        plt.ylabel('True Letter')
        plt.tight_layout()
        plt.savefig('transfer_learning_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Per-class accuracy
        print(f"\nPer-class accuracy:")
        for i, letter in enumerate(self.letters):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == i)
                print(f"  {letter}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
    
    def predict_letter(self, image_path):
        """
        Predict letter for a single image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (predicted_letter, confidence_scores)
        """
        if self.transfer_model is None:
            raise ValueError("Transfer model must be trained first")
        
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        # Predict
        predictions = self.transfer_model.predict(img)
        predicted_class = np.argmax(predictions[0])
        predicted_letter = self.letters[predicted_class]
        confidence = predictions[0][predicted_class]
        
        return predicted_letter, predictions[0], confidence
    
    def run_complete_pipeline(self):
        """
        Run the complete transfer learning pipeline.
        """
        print("🚀 MNIST TO LETTERS TRANSFER LEARNING PIPELINE")
        print("=" * 80)
        
        # Step 1: Train MNIST model
        self.create_mnist_model()
        
        # Step 2: Load letter dataset
        self.load_letter_dataset()
        
        # Step 3: Create base model
        base_model = self.create_base_model()
        
        # Step 4: Create transfer model
        self.create_transfer_model(base_model)
        
        # Step 5: Train transfer model
        self.train_transfer_model(fine_tune=True)
        
        print("\n🎉 TRANSFER LEARNING PIPELINE COMPLETED!")
        print("=" * 80)
        
        return self.transfer_model

def main():
    """Main function to run transfer learning experiment."""
    
    # Create transfer learning pipeline
    transfer_pipeline = MNISTToLettersTransfer(data_dir="Images")
    
    # Run complete pipeline
    model = transfer_pipeline.run_complete_pipeline()
    
    # Save the trained model
    model.save('mnist_to_letters_transfer_model.h5')
    print(f"\n✅ Model saved as 'mnist_to_letters_transfer_model.h5'")
    
    # Example prediction on a test image
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Test predictions on some images from the dataset
    test_images = [
        "Images/A/A1.jpg",
        "Images/B/B1.jpg", 
        "Images/C/C1.jpg",
        "Images/D/D6.jpg",
        "Images/E/E1.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            try:
                predicted_letter, all_probs, confidence = transfer_pipeline.predict_letter(img_path)
                print(f"\nImage: {img_path}")
                print(f"Predicted: {predicted_letter} (confidence: {confidence:.4f})")
                print(f"All probabilities: {dict(zip(transfer_pipeline.letters, all_probs))}")
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")

if __name__ == "__main__":
    main() 