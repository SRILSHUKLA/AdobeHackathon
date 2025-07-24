#!/usr/bin/env python3
import os
import pandas as pd
from src.data_preprocessing import DatasetCreator
from src.trainer import HeadingTrainer

def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Create dataset
    print("Creating training dataset...")
    creator = DatasetCreator()
    df = creator.create_training_data('data/train/pdfs', 'data/train/annotations')
    
    print(f"Created dataset with {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Train model
    print("\nTraining model...")
    trainer = HeadingTrainer()
    train_loader, val_loader = trainer.prepare_data(df)
    
    trainer.train(train_loader, val_loader, epochs=10)
    
    print("Training completed!")

if __name__ == "__main__":
    main()