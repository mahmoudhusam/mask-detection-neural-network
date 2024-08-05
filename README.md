# Mask Detection Using Various Models

This project aims to classify images of faces with and without masks using different machine learning and deep learning models. The models implemented include:

- HOG + SVM
- MobileNet with various fine-tuning strategies
- InceptionV3 with various fine-tuning strategies

## Project Structure

mask-detection
├── datasets
│ ├── masks_dataset
│ │ ├── Train
│ │ ├── Test
│ │ └── Validation
├── models
│ ├── mask_detection_model_1m.h5
│ ├── mask_detection_model_2m.h5
│ ├── mask_detection_model_3m.h5
│ ├── mask_detection_model_4m.h5
│ ├── mask_detection_modelv_1m.h5
│ ├── mask_detection_modelv_2m.h5
│ └── mask_detection_modelv_3m.h5
├── notebooks
│ ├── hog_svm_model.ipynb
│ ├── mobilenet_model.ipynb
│ └── inceptionv3_model.ipynb
├── scripts
│ ├── data_loader.py
│ ├── hog_svm_model.py
│ ├── mobilenet_model.py
│ └── inceptionv3_model.py
└── README.md


## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mask-detection

2. **Download the dataset from Kaggle and place it in the datasets/masks_dataset directory:**

Copy code
datasets
├── masks_dataset
│   ├── Train
│   ├── Test
│   └── Validation

3. **Install the required libraries:**

pip install tensorflow scikit-learn opencv-python matplotlib seaborn

4. **Run the models:**

For HOG + SVM:
python scripts/hog_svm_model.py

For MobileNet:
python scripts/mobilenet_model.py

For InceptionV3:
python scripts/inceptionv3_model.py

