from models import BaseModel
import torch.nn as nn
import torch.nn.functional as f

class CNN(BaseModel):
    def __init__(self, filed = 80):
        super(CNN, self).__init__()
        self.filed = filed
        self.cov = nn.Sequential(
            nn.Conv1d(in_channels=self.filed,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=256, out_channels=self.filed, kernel_size=1),
        )

    def forward(self, x):
        x_out = self.cov(x)
        return x_out

