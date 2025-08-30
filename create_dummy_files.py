# create_dummy_files.py
import pickle
import torch
import numpy as np

# Create dummy class names
class_names = ['hello', 'thank_you', 'please', 'sorry', 'yes', 'no', 'help', 'stop', 'good', 'bad']

# Save class names
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

# Create a dummy model file (random weights)
from model import AttentionSignLanguageClassifier

input_size = 78
hidden_size = 128
num_classes = len(class_names)

model = AttentionSignLanguageClassifier(input_size, hidden_size, num_classes)
torch.save(model.state_dict(), 'best_model.pth')

print("Dummy files created successfully!")
print("- class_names.pkl")
print("- best_model.pth")
print("\nNow you can run: python main.py")