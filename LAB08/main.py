import torch
import torch.nn as nn
import torch.optim as optim

from data_prep import get_data_loaders
from model import FashionDCNN_ConfigA, FashionDCNN_ConfigB
from evaluate import evaluate_model, plot_and_save_results

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(batch_size=64)

    def train_config(model, epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.to(device)

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / total
            train_acc = 100. * correct / total

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            test_loss = val_loss / val_total
            test_acc = 100. * val_correct / val_total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% || "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

        return history

    print("\n--- Training Config A (2 Conv Blocks) ---")
    model_a = FashionDCNN_ConfigA()
    history_a = train_config(model_a, epochs=10)

    print("\n--- Training Config B (3 Conv Blocks) ---")
    model_b = FashionDCNN_ConfigB()
    history_b = train_config(model_b, epochs=10)

    print("\nEvaluating final models...")
    acc_a = evaluate_model(model_a, test_loader, device)
    acc_b = evaluate_model(model_b, test_loader, device)
    print(f"Config A Final Test Accuracy: {acc_a:.2f}%")
    print(f"Config B Final Test Accuracy: {acc_b:.2f}%")

    plot_and_save_results(history_a, history_b)

if __name__ == '__main__':
    run_training()