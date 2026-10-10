from data_prep import load_data
from model import build_dcnn_model
from evaluate import evaluate_and_plot

def main():
    print("==> กำลังโหลดชุดข้อมูล Fashion-MNIST...")
    (X_train, y_train), (X_test, y_test) = load_data()

    print("\n================== การทดลองที่ 1: Config A ==================")
    model_a = build_dcnn_model(config_name="Config_A")
    evaluate_and_plot(model_a, X_train, y_train, X_test, y_test, epochs=10, config_title="Config_A")

    print("\n================== การทดลองที่ 2: Config B ==================")
    model_b = build_dcnn_model(config_name="Config_B")
    evaluate_and_plot(model_b, X_train, y_train, X_test, y_test, epochs=15, config_title="Config_B")

if __name__ == "__main__":
    main()