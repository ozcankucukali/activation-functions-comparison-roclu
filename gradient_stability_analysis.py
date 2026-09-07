import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from activations import get_activation


def build_loaders(batch_size=128, num_workers=2, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return trainloader, testloader


class GradientMonitor:
    def __init__(self, log_frequency=30):
        self.gradients_history = []
        self.log_frequency = log_frequency
        self.batch_count = 0

    def monitor_gradients(self, model):
        if self.batch_count % self.log_frequency == 0:
            gradient_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    norm = param.grad.detach().norm().item()
                    gradient_norms.append(norm)
            if gradient_norms:
                self.gradients_history.append(float(np.mean(gradient_norms)))
        self.batch_count += 1


class DeepModel(nn.Module):
    """Configurable deep CNN used for gradient/numerical stability stress tests."""

    def __init__(self, activation, num_layers):
        super().__init__()
        layers = [
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            get_activation(activation),
            nn.BatchNorm2d(32),
        ]

        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False))
            layers.append(get_activation(activation))
            if i % 5 == 0 and i > 0:
                layers.append(nn.MaxPool2d(2, padding=1))
            layers.append(nn.BatchNorm2d(32))

        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 256),
            get_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            get_activation(activation),
            nn.Linear(128, 10),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    finite = True

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                finite = False
                break
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if total == 0:
        return math.nan, math.nan, finite
    return total_loss / len(loader), 100.0 * correct / total, finite


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    trainloader, testloader = build_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    activation_functions = {
        'RoCLU': 'roclu',
        'ReLU': 'relu',
        'LeakyReLU': 'leaky_relu',
        'ELU': 'elu',
        'GELU': 'gelu',
        'Mish': 'mish',
        'Swish': 'swish',
    }

    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for num_layers in args.layers:
        print(f'\nTesting with {num_layers} layers...')
        results[num_layers] = {}

        for activation_name, activation in activation_functions.items():
            print(f'  Activation: {activation_name}')
            monitor = GradientMonitor(log_frequency=args.log_frequency)
            model = DeepModel(activation, num_layers).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

            unstable = False
            final_train_loss = math.nan
            final_train_acc = math.nan

            for epoch in range(args.epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for inputs, targets in trainloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    if not torch.isfinite(loss):
                        unstable = True
                        print(f'    Numerical instability detected at epoch {epoch + 1}.')
                        break

                    loss.backward()
                    monitor.monitor_gradients(model)

                    # Treat non-finite gradients as instability as well.
                    finite_grads = all(
                        p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                    if not finite_grads:
                        unstable = True
                        print(f'    Non-finite gradient detected at epoch {epoch + 1}.')
                        break

                    optimizer.step()
                    running_loss += loss.item()
                    predicted = outputs.argmax(dim=1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                if total > 0:
                    final_train_loss = running_loss / len(trainloader)
                    final_train_acc = 100.0 * correct / total

                if unstable:
                    break

                if (epoch + 1) % 2 == 0:
                    print(
                        f'    Epoch {epoch + 1}/{args.epochs} - '
                        f'Loss: {final_train_loss:.4f}, Acc: {final_train_acc:.2f}%'
                    )

            final_val_loss, final_val_acc, finite_eval = evaluate(
                model, testloader, criterion, device
            )
            unstable = unstable or not finite_eval

            results[num_layers][activation_name] = {
                'gradient_history': monitor.gradients_history,
                'final_train_accuracy': final_train_acc,
                'final_val_accuracy': final_val_acc,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'numerically_stable': not unstable,
            }

        plt.figure(figsize=(10, 6))
        for activation_name in activation_functions:
            history = results[num_layers][activation_name]['gradient_history']
            if history:
                plt.plot(history, label=activation_name)
        plt.xlabel('Logged batch index')
        plt.ylabel('Mean parameter-gradient norm')
        plt.title(f'Gradient norms for a {num_layers}-layer network')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f'gradient_norms_{num_layers}_layers.png')
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f'  Plot saved to: {plot_path}')

    print('\n==== Results ====')
    for num_layers in args.layers:
        print(f'\n{num_layers}-layer network:')
        for activation_name, values in results[num_layers].items():
            mean_grad = (
                np.mean(values['gradient_history'])
                if values['gradient_history'] else math.nan
            )
            print(
                f"  {activation_name}: stable={values['numerically_stable']}, "
                f"train_acc={values['final_train_accuracy']:.4f}, "
                f"val_acc={values['final_val_accuracy']:.4f}, "
                f"mean_grad_norm={mean_grad:.6g}"
            )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gradient and numerical stability analysis for RoCLU and baseline activations.'
    )
    parser.add_argument('--layers', nargs='+', type=int, default=[50],
                        help='Network depths to test, e.g. --layers 50 100 300 500 700')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--log_frequency', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
