import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as T

center_freq = 3e6
fs = 30e6
c = 1540


def preprocess_data(data: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Time gain compensation: multiply each depth row by an exponential gain
    to counteract depth-dependent signal attenuation.
 
    Why exponential? Because ultrasound attenuation is exponential with depth
    in dB scale, so the inverse compensation is also exponential.
 
    Args:
        rf:    (2048, 128) RF array — rows = depth, cols = lateral
        alpha: attention coefficient per sample (tune to your system)
    """
    rf = data[1:, :]
    rf = rf.astype(np.float32)
    depth = np.arange(rf.shape[0], dtype=np.float32)
    gain  = np.exp(alpha * depth)   # shape: (2048,)
                                    # row 0 (shallow) gets gain ~1.0
                                    # row 2047 (deep) gets gain ~e^6.1 ≈ 450
                                    # — compensates for signal loss with depth
    return rf * gain[:, None]

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