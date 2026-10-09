# evaluate.py
import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, title_suffix=""):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # กราฟ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation Acc')
    plt.title(f'Training and Validation Accuracy {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # กราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, class_names, config_name=""):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n================ {config_name} ================")
    print(f"Test Accuracy: {accuracy*100:.2f}% | Test Loss: {loss:.4f}")
    
    # ส่วนการทำนายผล (Predictions)
    predictions = model.predict(X_test[:5], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    print("\n--- Sample Predictions ---")
    for i in range(5):
        actual = class_names[y_test[i]]
        predicted = class_names[pred_classes[i]]
        print(f"Sample {i+1}: Actual = {actual:<15} | Predicted = {predicted}")
    print("================================================\n")
    
    return accuracy