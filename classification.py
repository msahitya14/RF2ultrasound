import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# Convert rf_data to b-mode images
# Needs rf_folder to have True and False folders for classes
# TODO:
def get_images(rf_folder):
    return 0

# TODO: adjust batch size to fit with 
def get_loaders():
    dataset = datasets.ImageFolder(root = "Data")
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
        self.classifier = nn.Linear(32, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, (3, 3))
        out = out.flatten(out, start_dim = 1)
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
    running_loss = 0.0
    for image, label in dataloader:
        image.to(device)
        label.to(device)

        optimizer.zero_grad()
        logits = model(image)
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss/len(dataloader)

# return accuracy of model
def evaluate(model, dataloader, criterion, device):
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predict = outputs.argmax(dim = 1)
        correct += (predict == labels).sum().item()
        samples += images.size(0)
    return correct/samples


def main(rf_folder):
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