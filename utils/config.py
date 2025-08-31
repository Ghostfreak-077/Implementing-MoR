import torch

class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'