from sklearn.svm import SVC

def train_svm_kernels(X_train, y_train):
    models = {}
    # กำหนด Kernel ทั้ง 3 แบบ
    kernels = ['linear', 'poly', 'rbf']
    
    for kernel in kernels:
        # สร้างและเทรนโมเดล SVM
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train, y_train)
        models[kernel] = svm
        
    return models