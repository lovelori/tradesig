import torch
import torch.nn as nn

class TorchNet(nn.Module):
    def __init__(self,window_size=32,win2=16):
        super(TorchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                            # Flatten 5x99 -> 495 features
            nn.Linear(4 * window_size, win2),
            nn.ReLU(),
            nn.Linear(win2, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Tanh()  # Ensures the output is in (-1, 1)
        )

    def forward(self, x):
        return self.net(x)



