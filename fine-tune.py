from ultralytics import YOLO
import yaml
import os

def load_yaml_config(yaml_path):
    """Load YAML configuration file"""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def fine_tune_model(pretrained_model_path, yaml_path, epochs=100):
    """
    Fine-tune YOLOv8 model while preserving previous weights
    
    Args:
        pretrained_model_path: Path to pretrained .pt model
        yaml_path: Path to dataset YAML file
        epochs: Number of training epochs
    """
    # Load the pre-trained model
    model = YOLO(pretrained_model_path)
    
    # Configure training parameters
    training_args = {
        'data': yaml_path,          # Path to data YAML file
        'epochs': epochs,           # Number of epochs
        'imgsz': 640,              # Image size
        'batch': 16,               # Batch size
        'device': 0,               # GPU device (use -1 for CPU)
        'workers': 8,              # Number of worker threads
        'patience': 50,            # Early stopping patience
        'save': True,              # Save training results
        'exist_ok': True,          # Overwrite existing experiment
        'pretrained': True,        # Use pretrained weights
        'freeze': 10,              # Freeze first 10 layers to preserve knowledge
        'cache': False,            # Cache images for faster training
        'optimizer': 'AdamW',      # Optimizer
        'lr0': 0.001,             # Initial learning rate
        'lrf': 0.01,              # Final learning rate fraction
        'momentum': 0.937,         # SGD momentum/Adam beta1
        'weight_decay': 0.0005,    # Optimizer weight decay
        'warmup_epochs': 3.0,      # Warmup epochs
        'warmup_momentum': 0.8,    # Warmup initial momentum
        'warmup_bias_lr': 0.1,     # Warmup initial bias lr
        'close_mosaic': 10,        # Disable mosaic augmentation for final epochs
        'box': 7.5,               # Box loss gain
        'cls': 0.5,               # Cls loss gain
        'dfl': 1.5,               # DFL loss gain
        'label_smoothing': 0.0,    # Label smoothing epsilon
        'nbs': 64,                # Nominal batch size
    }
    
    # Start training
    try:
        results = model.train(**training_args)
        print("Training completed successfully!")
        print(f"Best model saved at: {results.best}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        
if __name__ == "__main__":
    # Paths
    yaml_path = "aircraft-skin-defects.v1-size-640-stock.yolov5pytorch\data.yaml"  # Path to your YAML file
    pretrained_model_path = "best.pt"  # Path to your pretrained model
    
    # Start fine-tuning
    fine_tune_model(pretrained_model_path, yaml_path)