import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.num_actions = env.num_actions

        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1)

        self.fc1 = nn.Linear(in_features = 3136, out_features = 512)
        self.output_layer = nn.Linear(in_features = 512, out_features = self.num_actions)

    def forward(self, x):
        latent = self.extract_features(x)
        q_values = self.output_layer(latent)
        return q_values

    def extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim = 1)
        latent = F.relu(self.fc1(x))
        return latent