import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    image_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
    training_images = image_generator.flow_from_directory("./datasets/masks_dataset/Train", target_size=(128, 128))
    validation_images = image_generator.flow_from_directory("./datasets/masks_dataset/Validation", target_size=(128, 128))
    testing_images = image_generator.flow_from_directory("./datasets/masks_dataset/Test", target_size=(128, 128))

    return training_images, validation_images, testing_images
