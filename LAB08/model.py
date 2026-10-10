from tensorflow.keras import layers, models

def build_dcnn_model(config_name="Config_A", input_shape=(28, 28, 1), num_classes=10):
    """
    สร้างโมเดล DCNN ตาม Configurations เพื่อเทียบความลึกของเลเยอร์
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    if config_name == "Config_A":
        # Config A: 2 Convolutional Blocks (Shallow DCNN)
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

    elif config_name == "Config_B":
        # Config B: 3 Convolutional Blocks (Deeper DCNN)
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax')) # Multi-class classification (10 คลาส)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model