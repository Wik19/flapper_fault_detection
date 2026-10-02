import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusionCNN(nn.Module):
    # Added input_shape parameter. Default is your new 3.0s window shape!
    def __init__(self, input_shape=(7, 64, 188), num_classes=5, dropout_rate=0.3):
        super(EarlyFusionCNN, self).__init__()
        
        # ---------------------------------------------------
        # Convolutional Blocks
        # ---------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------------------------------
        # Dynamic Flatten Size Calculation
        # ---------------------------------------------------
        # We push a dummy tensor through the conv layers to see exactly what shape comes out.
        # This makes the network bulletproof against changing window sizes!
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            self.flatten_size = x.numel() # Counts total elements (256 * 4 * 11 = 11264)

        # ---------------------------------------------------
        # Fully Connected (Classifier) Head
        # ---------------------------------------------------
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, self.flatten_size)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        
        return x

if __name__ == "__main__":
    dummy_batch = torch.randn(32, 7, 64, 188)
    model = EarlyFusionCNN(input_shape=(7, 64, 188), num_classes=5)
    output = model(dummy_batch)
    print(f"Model initialized safely. Dynamically calculated flatten size: {model.flatten_size}")
    print(f"Output shape: {output.shape}")