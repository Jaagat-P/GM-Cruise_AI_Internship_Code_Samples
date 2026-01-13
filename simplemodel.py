# Run in terminal
# pip3 install ultralytics 
# pip3 install onnx onnxruntime-gpu --user 

# Run this on Lambda Vector Pro machine to generate labeled images and begin fine-tuning ASAP.

from ultralytics import YOLO
import torch
import os 

# Load the ONNX model
model_path = '/Users/jzcjk4/Downloads/model.pt'
image_path = 'test.png'

model = YOLO('/Users/jzcjk4/Downloads/model.pt')

# Run inference on an image
results = model('test.png')
print(results)

# Save each result with a unique filename based on its index
for idx, result in enumerate(results):
    output_path = f"result_{idx}.jpg"
    result.save(filename=output_path)
    print(f"Saved: {output_path}")
