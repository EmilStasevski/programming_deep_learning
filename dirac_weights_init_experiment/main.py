import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.shallow_cnn import ShallowCNN
from models.deep_plain_cnn import DeepPlainCNN
from utils.initializers import apply_kaiming_init, apply_dirac_init
from utils.plotting import plot_experiment_results
from train import train_one_epoch


def run_experiment(model_instance, init_func, trainloader, device, epochs=4):
    # Apply selected weight initialization
    model_instance.apply(init_func)
    model = model_instance.to(device)

    criterion = nn.CrossEntropyLoss()
    # Keeping learning rate stable across configurations for a fair benchmark
    # Change this inside main.py -> run_experiment
    # Inside run_experiment in main.py
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    loss_history = []
    for epoch in range(epochs):
        print(f"    Epoch {epoch + 1}/{epochs}...")
        train_one_epoch(model, trainloader, criterion, optimizer, device, loss_history)

    return loss_history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    results = {}

    # Configuration 1: Shallow Architecture Benchmark
    print("\n--- 1. Running Shallow CNN (Kaiming Baseline) ---")
    shallow_model = ShallowCNN(num_classes=100)
    results['Shallow CNN (Kaiming)'] = run_experiment(shallow_model, apply_kaiming_init, trainloader, device)

    # Configuration 2: Deep Network with Standard ReLU and Kaiming
    print("\n--- 2. Running Deep CNN (Standard ReLU + Kaiming) ---")
    deep_relu_kaiming = DeepPlainCNN(num_classes=100, use_leaky=False)
    results['Deep Plain + ReLU (Kaiming)'] = run_experiment(deep_relu_kaiming, apply_kaiming_init, trainloader, device)

    # Configuration 3: Deep Network with Standard ReLU and Dirac (The Trap)
    print("\n--- 3. Running Deep CNN (Standard ReLU + Dirac) ---")
    deep_relu_dirac = DeepPlainCNN(num_classes=100, use_leaky=False)
    results['Deep Plain + ReLU (Dirac)'] = run_experiment(deep_relu_dirac, apply_dirac_init, trainloader, device)

    # Configuration 4: Deep Network with LeakyReLU and Dirac (The Optimized Solution)
    print("\n--- 4. Running Deep CNN (LeakyReLU + Dirac) ---")
    deep_leaky_dirac = DeepPlainCNN(num_classes=100, use_leaky=True)
    results['Deep Plain + LeakyReLU (Dirac)'] = run_experiment(deep_leaky_dirac, apply_dirac_init, trainloader, device)

    # Plot everything together
    print("\nGenerating final comparison plots...")
    plot_experiment_results(results, 'Weight Initialization & Activation Comparison on CIFAR-100')


if __name__ == '__main__':
    main()