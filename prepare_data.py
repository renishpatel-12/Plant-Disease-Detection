#!/usr/bin/env python3
"""
Data Preparation Script for Plant Disease Detection
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_directory_structure():
    """Create the required directory structure"""
    
    directories = [
        'dataset/healthy',
        'dataset/multiple_diseases', 
        'dataset/rust',
        'dataset/scab',
        'sample_images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print("Directory structure created:")
    for directory in directories:
        print(f"  - {directory}")

def generate_sample_images():
    """Generate sample images for testing (placeholder images)"""
    
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    colors = {
        'healthy': (0, 255, 0),      # Green
        'multiple_diseases': (255, 165, 0),  # Orange  
        'rust': (139, 69, 19),       # Brown
        'scab': (128, 128, 128)      # Gray
    }
    
    for class_name in classes:
        class_dir = Path(f'dataset/{class_name}')
        
        # Generate 10 sample images per class
        for i in range(10):
            # Create a simple colored image with some noise
            img_array = np.full((512, 512, 3), colors[class_name], dtype=np.uint8)
            
            # Add some random noise to make it look more realistic
            noise = np.random.randint(-30, 30, (512, 512, 3))
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Add some random shapes to simulate leaf patterns
            for _ in range(5):
                x, y = np.random.randint(50, 462, 2)
                radius = np.random.randint(20, 50)
                
                # Create circular patterns
                Y, X = np.ogrid[:512, :512]
                mask = (X - x)**2 + (Y - y)**2 <= radius**2
                
                # Darken the circular areas
                img_array[mask] = np.clip(img_array[mask] * 0.7, 0, 255).astype(np.uint8)
            
            # Save image
            img = Image.fromarray(img_array)
            img.save(class_dir / f'sample_{class_name}_{i+1:02d}.jpg')
    
    print(f"Generated sample images for all classes")

def validate_dataset():
    """Validate the dataset structure and count images"""
    
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        print("Dataset directory not found!")
        return False
    
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    total_images = 0
    
    print("\nDataset validation:")
    print("-" * 40)
    
    for class_name in classes:
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            count = len(image_files)
            total_images += count
            print(f"{class_name:20}: {count:4d} images")
        else:
            print(f"{class_name:20}: Directory not found!")
            return False
    
    print("-" * 40)
    print(f"{'Total':20}: {total_images:4d} images")
    
    if total_images == 0:
        print("\nNo images found! Please add images to the class directories.")
        return False
    
    return True

def show_sample_images():
    """Display sample images from each class"""
    
    dataset_dir = Path('dataset')
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, class_name in enumerate(classes):
        class_dir = dataset_dir / class_name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        if image_files:
            # Load and display first image
            img_path = image_files[0]
            img = Image.open(img_path)
            
            axes[i].imshow(img)
            axes[i].set_title(f'{class_name.replace("_", " ").title()}\n({len(image_files)} images)')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No images\nfound for\n{class_name}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(class_name.replace("_", " ").title())
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_csv_from_directory():
    """Create a CSV file from directory structure (if needed for other scripts)"""
    
    dataset_dir = Path('dataset')
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    
    data = []
    
    for class_name in classes:
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_file in image_files:
                # Create one-hot encoding
                row = {
                    'image_id': img_file.stem,
                    'image_path': str(img_file),
                    'healthy': 1 if class_name == 'healthy' else 0,
                    'multiple_diseases': 1 if class_name == 'multiple_diseases' else 0,
                    'rust': 1 if class_name == 'rust' else 0,
                    'scab': 1 if class_name == 'scab' else 0,
                    'class': class_name
                }
                data.append(row)
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv('dataset_info.csv', index=False)
        print(f"\nCreated dataset_info.csv with {len(df)} entries")
        print(df['class'].value_counts())
    else:
        print("No data found to create CSV")

def main():
    """Main function"""
    
    print("Plant Disease Detection - Data Preparation")
    print("=" * 50)
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # Check if we need to generate sample data
    if not validate_dataset():
        print("\n2. Generating sample images for demonstration...")
        generate_sample_images()
        print("   Sample images generated successfully!")
        print("   Replace these with your actual plant disease images.")
    
    # Validate dataset
    print("\n3. Validating dataset...")
    if validate_dataset():
        print("   Dataset validation successful!")
        
        # Show sample images
        print("\n4. Displaying sample images...")
        show_sample_images()
        
        # Create CSV file
        print("\n5. Creating dataset CSV...")
        create_csv_from_directory()
        
        print("\n" + "=" * 50)
        print("Data preparation completed!")
        print("\nNext steps:")
        print("1. Replace sample images with your actual plant disease images")
        print("2. Run 'python train_model.py' to train the model")
        print("3. Run 'streamlit run app.py' to test the web application")
    
if __name__ == "__main__":
    main()