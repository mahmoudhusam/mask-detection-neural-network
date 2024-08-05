import time
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def create_inceptionv3_model(training_images, validation_images, freeze_layers, epochs=5):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        Flatten(),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(training_images, epochs=epochs, validation_data=validation_images)
    end_time = time.time()
    training_time = end_time - start_time

    return model, history.history['accuracy'], training_time
