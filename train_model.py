#!/usr/bin/env python3
"""
Plant Disease Detection Model Training Script
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class PlantDiseaseModel:
    def __init__(self, input_shape=(512, 512, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_model(self):
        """Create ensemble model with Xception and DenseNet121"""
        
        # Xception Model
        xception_base = tf.keras.applications.Xception(
            include_top=False, 
            weights='imagenet', 
            input_shape=self.input_shape
        )
        xception_base.trainable = False  # Freeze base layers initially
        
        xception_model = tf.keras.Sequential([
            xception_base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # DenseNet Model
        densenet_base = tf.keras.applications.DenseNet121(
            include_top=False, 
            weights='imagenet', 
            input_shape=self.input_shape
        )
        densenet_base.trainable = False  # Freeze base layers initially
        
        densenet_model = tf.keras.Sequential([
            densenet_base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Ensemble Model
        inputs = tf.keras.Input(shape=self.input_shape)
        xception_output = xception_model(inputs)
        densenet_output = densenet_model(inputs)
        
        # Average the predictions
        outputs = tf.keras.layers.Average()([xception_output, densenet_output])
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return self.model
    
    def create_data_generators(self, train_dir, validation_split=0.2, batch_size=16):
        """Create data generators for training"""
        
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def get_callbacks(self, model_path='model.h5'):
        """Get training callbacks"""
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, validation_generator, epochs=50):
        """Train the model"""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-k accuracy
        if 'top_k_categorical_accuracy' in history.history:
            axes[1, 0].plot(history.history['top_k_categorical_accuracy'], label='Training Top-K Accuracy')
            axes[1, 0].plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-K Accuracy')
            axes[1, 0].set_title('Model Top-K Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-K Accuracy')
            axes[1, 0].legend()
        
        # Learning rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_sample_dataset_structure():
    """Create sample dataset structure for demonstration"""
    
    base_dir = Path('dataset')
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    
    for class_name in classes:
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"Created dataset structure at: {base_dir.absolute()}")
    print("Please add your images to the respective class folders:")
    for class_name in classes:
        print(f"  - {base_dir / class_name}")

def main():
    """Main training function"""
    
    # Check if dataset exists
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        print("Dataset directory not found. Creating sample structure...")
        create_sample_dataset_structure()
        print("\nPlease add your training images and run the script again.")
        return
    
    # Initialize model
    model_trainer = PlantDiseaseModel()
    
    # Create model
    print("Creating ensemble model...")
    model = model_trainer.create_model()
    print(f"Model created with {model.count_params():,} parameters")
    
    # Create data generators
    print("Creating data generators...")
    try:
        train_gen, val_gen = model_trainer.create_data_generators(
            str(dataset_dir),
            validation_split=0.2,
            batch_size=16
        )
        
        print(f"Found {train_gen.samples} training images")
        print(f"Found {val_gen.samples} validation images")
        print(f"Classes: {list(train_gen.class_indices.keys())}")
        
    except Exception as e:
        print(f"Error creating data generators: {e}")
        print("Make sure your dataset directory contains subdirectories for each class with images.")
        return
    
    # Train model
    print("Starting training...")
    history = model_trainer.train(
        train_gen, 
        val_gen, 
        epochs=50
    )
    
    # Plot training history
    model_trainer.plot_training_history(history)
    
    # Save training history
    pd.DataFrame(history.history).to_csv('training_history.csv', index=False)
    
    print("Training completed!")
    print("Model saved as 'model.h5'")
    print("Training history saved as 'training_history.csv'")

if __name__ == "__main__":
    main()