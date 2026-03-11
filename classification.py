import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as T
import medmnist
from medmnist import BreastMNIST, INFO



# Convert rf_data to b-mode images
# Needs rf_folder to have True and False folders for classes
# TODO:
def get_images(rf_folder):
    return 0

def get_med_loaders(batch_size):
    data_transform = T.ToTensor()
    train_dataset = BreastMNIST(split="train", transform=data_transform, download = True)
    test_dataset = BreastMNIST(split ="test", transform=data_transform, download = True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2* batch_size, shuffle=False)
    return (train_loader, test_loader)



# temporary loaders for cifar10, dogs and cats
def get_cifar_loaders():
    transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root="./pytorch_data",
        train=True,
        download=True,
        transform=transform
    )

    # keep only cats (3) and dogs (5)
    dataset = [
        (img, 0 if label == 3 else 1)
        for img, label in dataset
        if label in [3, 5]
    ]

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size = int(train_size/10), shuffle = True, download = True)
    test_loader = DataLoader(test_data, batch_size = len(dataset) - int(train_size/10), shuffle = False, download = True)
    return (train_loader, test_loader)

# TODO: adjust batch size to fit 
def get_loaders():
    transform = T.ToTensor()
    dataset = datasets.ImageFolder(root = "Data", transform= transform)
    train_size = int(0.6 * len(dataset))
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
        weights = ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,   # ~256 channels
            backbone.layer2,   # ~512 channels
            # backbone.layer3, # removed — was 1024 channels
            # backbone.layer4, # removed — was 2048 channels
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),  # 512 matches layer2 output
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

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
        labels = labels.to(device).float().view(-1,1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        predict = (logits.sigmoid()>0.5).long().view(-1)
        correct += (predict == labels.view(-1)).sum().item()
        samples += images.size(0)
    return correct/samples

# return accuracy of model
def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1,1)
        logits = model(images)

        predict = (logits.sigmoid()>0.5).long().view(-1)
        correct += (predict == labels.view(-1)).sum().item()
        samples += images.size(0)
    print(f"Test_correct: {correct} out of {samples}")
    return correct/samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= 0.001)
    epochs = 25
    train_loader, test_loader = get_med_loaders(128)
    for i in range(epochs):
        train_error = train(model, train_loader, optimizer, criterion, device)
        test_error = evaluate(model, test_loader, criterion, device)
        print(f"Train accuracy for epoch {i}: {train_error}")
        print(f"Test accuracy for epoch {i}: {test_error}")


if __name__ == "__main__":
    main()