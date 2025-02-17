import torch
import torch.nn as nn

class TorchNet(nn.Module):
    def __init__(self,window_size=32):
        super(TorchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                            # Flatten 5x99 -> 495 features
            nn.Linear(4 * window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Tanh()  # Ensures the output is in (-1, 1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    model = TorchNet()
    # Create a random tensor with shape (batch_size, 5, 99)
    x = torch.randn(1, 5, 99)
    y = model(x)
    print("Output:", y.item())