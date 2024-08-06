# Mask Detection Using Various Models

This project aims to classify images of faces with and without masks using different machine learning and deep learning models. The models implemented include:

- HOG + SVM
- MobileNet with various fine-tuning strategies
- InceptionV3 with various fine-tuning strategies

## Project Structure

```
mask-detection
├── datasets
│ ├── masks_dataset
│ │ ├── Train
│ │ ├── Test
│ │ └── Validation
├── scripts
│ ├── data_loader.py
│ ├── hog_svm_model.py
│ ├── mobilenet_model.py
│ └── inceptionv3_model.py
└── README.md
```


## Getting Started

1. **Clone the repository:**
   ```bash
   git clone git@github.com:mahmoudhusam/mask-detection-neural-network.git
   cd mask-detection

2. **Download the dataset from Kaggle and place it in the datasets/masks_dataset directory:**
```

Copy code
datasets
├── masks_dataset
│   ├── Train
│   ├── Test
│   └── Validation
```

3. **Install the required libraries:**
   ```bash
      pip install tensorflow scikit-learn opencv-python matplotlib seaborn

4. **Run the models:**

- For HOG + SVM:
   ```bash
      python scripts/hog_svm_model.py


- For MobileNet:
   ```bash
      python scripts/mobilenet_model.py


- For InceptionV3:
   ```bash
      python scripts/inceptionv3_model.py

