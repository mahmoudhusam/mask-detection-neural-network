import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_hog_svm_model(training_paths):
    features = []
    labels = []
    hog1 = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16, 16),
                            _blockStride=(8, 8), _cellSize=(8, 8),
                            _nbins=9)

    for path in training_paths:
        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.resize(image_gray, (64, 128))
        hog1_features = hog1.compute(image_gray)
        hog1_features = hog1_features.flatten()
        features.append(hog1_features)

        if "WithMask" in path:
            labels.append("With Mask")
        elif "WithoutMask" in path:
            labels.append("Without Mask")

    X1 = np.array(features)
    scaler = StandardScaler()
    X_scaled1 = scaler.fit_transform(X1)
    svm_model = SVC()
    svm_model.fit(X_scaled1, labels)
    return svm_model, scaler

def evaluate_hog_svm_model(svm_model, scaler, test_paths):
    test_features = []
    test_labels = []
    hog1 = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16, 16),
                            _blockStride=(8, 8), _cellSize=(8, 8),
                            _nbins=9)

    for path in test_paths:
        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.resize(image_gray, (64, 128))
        hog1_features = hog1.compute(image_gray)
        hog1_features = hog1_features.flatten()
        test_features.append(hog1_features)

        if "WithMask" in path:
            test_labels.append("With Mask")
        elif "WithoutMask" in path:
            test_labels.append("Without Mask")

    X_test_Face = np.array(test_features)
    X_test_scaled_Face = scaler.transform(X_test_Face)
    accuracy = svm_model.score(X_test_scaled_Face, test_labels)
    print("Model accuracy on test set: %.2f%%" % (accuracy * 100))

    y_pred = svm_model.predict(X_test_scaled_Face)
    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['With Mask', 'Without Mask'], yticklabels=['With Mask', 'Without Mask'])
    plt.title('Confusion Matrix - Face Mask Images Dataset')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
