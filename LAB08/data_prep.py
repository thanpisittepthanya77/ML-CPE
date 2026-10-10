import tensorflow as tf

def load_data():
    """
    โหลดชุดข้อมูล Fashion-MNIST จาก Keras และเตรียมข้อมูลสำหรับ DCNN
    """
    # โหลดชุดข้อมูล Train และ Test
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalization ค่าพิกเซลให้อยู่ในช่วง [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # ปรับมิติข้อมูลให้เป็น (Samples, Height, Width, Channels) -> (28, 28, 1) สำหรับ Conv2D
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return (X_train, y_train), (X_test, y_test)