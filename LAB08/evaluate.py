import torch
import matplotlib.pyplot as plt
import os

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def plot_and_save_results(history_a, history_b):
    os.makedirs("output", exist_ok=True)
    
    epochs = range(1, len(history_a['train_acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # กราฟ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_a['train_acc'], 'b-', label='Train Acc (Config A)')
    plt.plot(epochs, history_a['test_acc'], 'b--', label='Test Acc (Config A)')
    plt.plot(epochs, history_b['train_acc'], 'r-', label='Train Acc (Config B)')
    plt.plot(epochs, history_b['test_acc'], 'r--', label='Test Acc (Config B)')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # กราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_a['train_loss'], 'b-', label='Train Loss (Config A)')
    plt.plot(epochs, history_a['test_loss'], 'b--', label='Test Loss (Config A)')
    plt.plot(epochs, history_b['train_loss'], 'r-', label='Train Loss (Config B)')
    plt.plot(epochs, history_b['test_loss'], 'r--', label='Test Loss (Config B)')
    plt.title('Model Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/comparison_results.png')
    print("Saved comparison graph to output/comparison_results.png successfully!")
    plt.show()