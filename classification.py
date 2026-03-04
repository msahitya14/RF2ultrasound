import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchvision.transforms as T


# Convert rf_data to b-mode images
# Needs rf_folder to have True and False folders for classes
# TODO:
def get_images(rf_folder):
    return 0

# TODO: adjust batch size to fit with 
def get_loaders():
    transform = T.ToTensor()
    dataset = datasets.ImageFolder(root = "Data", transform= transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size = int(train_size/5), shuffle = True)
    test_loader = DataLoader(test_data, batch_size = len(dataset) - int(train_size/5), shuffle = False)
    return (train_loader, test_loader)

# Divides envelope into patches
# @param p_width, p_height = width and height of patches
# @param r_width, r_height = width and height of rf data
# TODO
def patch_extraction(envelope, p_width, p_height, r_width, r_height):
    return 0

# May consider 2 channels: envelope and bmode
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.classifier = nn.LazyLinear(1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, (3, 3))
        out = torch.flatten(out,1)
        out = self.classifier(out)
        return out

# @param model: CNN
# @param dataloader: bmode_image, label 
# @param optimizer: ADAM
# @param criterion: binary cross entropy loss
# @param device: cuda if possible, else cpu
# @return average loss
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    correct = 0
    samples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels.float().unsqueeze(1))

        loss.backward()
        optimizer.step()

        predict = (logits >= 0).long().squeeze(1)
        correct += (predict == labels).sum().item()
        samples += images.size(0)
    return correct/samples

# return accuracy of model
def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predict = (logits >= 0).long().squeeze(1)
            correct += (predict == labels).sum().item()
            samples += images.size(0)
    return correct/samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 5
    train_loader, test_loader = get_loaders()
    for i in range(epochs):
        train_error = train(model, train_loader, optimizer, criterion, device)
        test_error = evaluate(model, test_loader, criterion, device)
        print(f"Train accuracy for epoch {i}: {train_error}")
        print(f"Test accuracy for epoch {i}: {test_error}")


if __name__ == "__main__":
    main()