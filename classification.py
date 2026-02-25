import torch
import torch.nn as nn
import torch.nn.functional as F
# Convert rf_data to b-mode
# Also consider envelope, raw data for future
# TODO
def bmode_conversion(rf_data):
    return 0

# Divides envelope into patches
# @param p_width, p_height = width and height of patches
# @param r_width, r_height = width and height of rf data
# TODO
def patch_extraction(envelope, p_width, p_height, r_width, r_height):
    return 0

# May consider 2 channels: envelope and bmode
class CNN(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.classifier = nn.Linear(64, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (3, 3))
        out = out.flatten(out, start_dim = 1)
        out = self.classifier(out)
        return out

# @param model: CNN
# @param dataloader: bmode_image, label 
# @param optimizer: ADAM
# @param criterion: binary cross entropy loss
# @param device: cuda if possible, else cpu
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for image, label in dataloader:
        optimizer.zero_grad()
        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/len(dataloader)
    

def main(rf_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.BCELoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    main()