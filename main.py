# main.py
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from torch.utils.data import DataLoader  

from model import AttentionSignLanguageClassifier
from real_time_translator import RealTimeSignLanguageTranslator
from trainer import SignLanguageDataset, SignLanguageTrainer

def create_sample_data():
    """Create sample training data - replace with your actual dataset"""
    # This is just example code - you'll need to collect real sign language data
    num_samples = 1000
    sequence_length = 30
    feature_size = 78
    num_classes = 10
    
    # Generate random sample data
    sequences = []
    labels = []
    
    class_names = ['hello', 'thank_you', 'please', 'sorry', 'yes', 'no', 'help', 'stop', 'good', 'bad']
    
    for i in range(num_samples):
        # Random sequence
        sequence = np.random.randn(sequence_length, feature_size)
        sequences.append(sequence)
        labels.append(np.random.randint(0, num_classes))
    
    return sequences, labels, class_names

def train_model():
    """Train the sign language model"""
    # Create or load your dataset
    sequences, labels, class_names = create_sample_data()
    
    # Create dataset and dataloaders
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_size = 78
    hidden_size = 128
    num_classes = len(class_names)
    
    model = AttentionSignLanguageClassifier(input_size, hidden_size, num_classes)
    
    # Train model
    trainer = SignLanguageTrainer(model)
    trainer.train(train_loader, val_loader, epochs=100)
    
    # Save class names
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    print("Training completed!")

def run_real_time_translation():
    """Run real-time sign language translation"""
    # Load class names
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    
    # Initialize translator
    translator = RealTimeSignLanguageTranslator(
        model_path='best_model.pth',
        class_names=class_names,
        sequence_length=30,
        confidence_threshold=0.7
    )
    
    # Run real-time translation
    translator.run_real_time()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("Training model...")
        train_model()
    else:
        print("Running real-time translation...")
        print("Press 'q' to quit")
        run_real_time_translation()