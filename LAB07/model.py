# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes=3, filters=16):
    model = Sequential()
    
    # Conv1D Layer
    model.add(Conv1D(filters=filters, kernel_size=2, activation='relu', input_shape=input_shape))
    
    # --- เอา MaxPooling1D ออก ---
    
    # Flatten
    model.add(Flatten())
    
    # Dense Layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # ปรับ Learning Rate ให้เรียนรู้ไวขึ้นนิดหน่อย (เช่น 0.005 หรือ 0.01)
    model.compile(optimizer=Adam(learning_rate=0.005), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model