import matplotlib.pyplot as plt

def evaluate_and_plot(model, X_train, y_train, X_test, y_test, epochs=10, config_title="Config_A"):
    """
    เทรนโมเดล วัดผล และสร้างกราฟบันทึกผลลัพธ์
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=64
    )

    # ดึงค่าผลลัพธ์การเทรน
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    # พล็อตและบันทึกกราฟ
    plt.figure(figsize=(12, 5))

    # กราฟ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Accuracy ({config_title})')

    # กราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Loss ({config_title})')

    filename = f"{config_title.lower()}_results.png"
    plt.savefig(filename)
    plt.show()
    print(f"บันทึกกราฟผลการทดลองเรียบร้อยแล้ว: {filename}")

    return history