# Configuration file for training of PyTorch models
import torch

BATCH_SIZE = 6      # Batch size (increase/decrease according to device memory)
LEARNING_RATE = 0.003   # Learning-rate of optimizer
NUM_WORKERS = 2     # Number of workers in DataLoaders
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Run on GPU if available

# location to save best model and plots
OUT_DIR = 'outputs'
