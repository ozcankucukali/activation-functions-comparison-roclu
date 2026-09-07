import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm

from datasets.cifar10_loader import get_cifar10_dataloaders
from datasets.cifar100_loader import get_cifar100_dataloaders
from models.resnet50_custom import ResNet50
from models.senet18_custom import SENet18
from models.googlenet_custom import googlenet
from models.vgg16_custom import VGG16
from models.densenet121_custom import DenseNet121, densenet_cifar
from models.mobilenet_custom import MobileNetV1, MobileNetV2_model
from activations import AVAILABLE_ACTIVATIONS

def get_model(model_name, num_classes, activation):
    """Factory function to get models by name"""
    models = {
        'resnet50': lambda: ResNet50(num_classes=num_classes, activation=activation),
        'senet18': lambda: SENet18(num_classes=num_classes, activation=activation, use_preact=True),
        'googlenet': lambda: googlenet(num_classes=num_classes, activation=activation),
        'vgg16': lambda: VGG16(num_classes=num_classes, activation=activation),
        'densenet121': lambda: DenseNet121(num_classes=num_classes, activation=activation),
        'densenet_cifar': lambda: densenet_cifar(num_classes=num_classes, activation=activation),
        'mobilenet_v1': lambda: MobileNetV1(num_classes=num_classes, activation=activation),
        'mobilenet_v2': lambda: MobileNetV2_model(num_classes=num_classes, activation=activation),
    }
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not supported. "
                        f"Available models: {list(models.keys())}")
    
    return models[model_name]()

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc='Training', leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{running_loss/len(progress_bar):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(trainloader), 100. * correct / total

def evaluate(model, testloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(testloader), 100. * correct / total

def train_model(model, trainloader, testloader, epochs, learning_rate, device, model_name, activation):
    """Train a single model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
    
    print(f"\nTraining {model_name} with {activation} activation...")
    
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Test accuracy'yi de hesapla
            test_loss, test_acc = evaluate(model, testloader, criterion, device)
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Son epoch'tan sonra final test accuracy
    test_loss, final_acc = evaluate(model, testloader, criterion, device)
    
    return final_acc, {
        'final_accuracy': final_acc
    }

def run_experiments(args):
    """Run experiments with multiple trials"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset with appropriate batch sizes
    if args.dataset == 'cifar10':
        # CIFAR-10 için batch size 128
        trainloader, testloader = get_cifar10_dataloaders(batch_size=args.batch_size)
        num_classes = 10
        print(f"CIFAR-10 loaded with batch size: {args.batch_size}")
    elif args.dataset == 'cifar100':
        # The published RoCLU experiments used batch size 128 for all architectures.
        trainloader, testloader = get_cifar100_dataloaders(batch_size=args.batch_size)
        num_classes = 100
        print(f"CIFAR-100 loaded with batch size: {args.batch_size}")
    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported")
    
    # Results storage
    results = []
    
    # Run experiments for each model and activation combination
    for model_name in args.models:
        for activation in args.activations:
            print(f"\n{'='*60}")
            print(f"Experimenting with {model_name.upper()} + {activation.upper()}")
            print(f"{'='*60}")
            
            trial_accuracies = []
            
            for trial in range(args.num_trials):
                print(f"\nTrial {trial + 1}/{args.num_trials}")
                
                # Create model
                model = get_model(model_name, num_classes, activation)
                model = model.to(device)
                
                # Train model
                final_acc, metrics = train_model(
                    model, trainloader, testloader, 
                    args.epochs, args.learning_rate, device,
                    model_name, activation
                )
                
                trial_accuracies.append(final_acc)
                print(f'Trial {trial + 1} Final Accuracy: {final_acc:.2f}%')
            
            # Calculate statistics
            mean_acc = np.mean(trial_accuracies)
            std_acc = np.std(trial_accuracies)

            results.append({
                'Model': model_name,
                'Activation': activation,
                'Mean Accuracy': mean_acc,
                'Std Accuracy': std_acc
                # 'Max Accuracy' ve 'Min Accuracy' kaldırın
            })

            print(f'\n{model_name} + {activation} Results:')
            print(f'Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%')
    
    # Save results
    df_results = pd.DataFrame(results)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save to CSV
    results_file = f'results/results_{args.dataset}_{"-".join(args.models)}_{"-".join(args.activations)}.csv'
    df_results.to_csv(results_file, index=False)
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(df_results.to_string(index=False))
    print(f"\nResults saved to: {results_file}")
    
    return df_results

def main():
    parser = argparse.ArgumentParser(description='CNN Activation Function Comparison')
    
    # Dataset arguments
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10',
                       help='Dataset to use (default: cifar10)')
    
    # Model arguments
    parser.add_argument('--models', nargs='+', 
                       choices=['resnet50', 'senet18', 'googlenet', 'vgg16', 
                               'densenet121', 'densenet_cifar', 'mobilenet_v1', 'mobilenet_v2'],
                       default=['resnet50'],
                       help='Models to train (default: resnet50)')
    
    # Activation arguments
    parser.add_argument('--activations', nargs='+', 
                       choices=AVAILABLE_ACTIVATIONS,
                       default=['relu', 'roclu'],
                       help='Activation functions to test (default: relu, roclu)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--num_trials', type=int, default=5,
                       help='Number of trials per experiment (default: 5)')
    
    args = parser.parse_args()
    
    print("Configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models}")
    print(f"Activations: {args.activations}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Number of Trials: {args.num_trials}")
    
    # Run experiments
    results_df = run_experiments(args)

if __name__ == '__main__':
    main()
