import numpy as np
import tensorflow as tf
from tensorflow import keras
Sequential = keras.models.Sequential
to_categorical = keras.utils.to_categorical  
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
from Model_Training.dataset_loader import load_dataset
import streamlit as st
import time

def train_main_model(epochs=5):
    # Load dataset
    (x_train, y_train), (x_test, y_test) = load_dataset()

    print("Unique labels in y_train:", np.unique(y_train))
    print("Unique labels in y_test:", np.unique(y_test))


    # One-hot Encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Build Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Streamlit Progress Bar
    progress_bar = st.progress(0)
    text_area = st.empty()

    history = {'accuracy': []}

    for epoch in range(epochs):
        hist = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=1)
        progress_bar.progress((epoch + 1) / epochs)
        time.sleep(0.5)  
        history['accuracy'].append(hist.history['accuracy'][0])
        text_area.text(f"Epoch {epoch + 1}/{epochs} - Accuracy: {history['accuracy'][-1]:.2%}")

      
      # Step 4: Predict and check label distribution
    
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)  # Only if y_test is one-hot encoded
    
    # Print distribution of predicted and true labels
    print("Predicted class distribution:", np.unique(y_pred_classes, return_counts=True))
    print("True label distribution:", np.unique(y_true, return_counts=True))




    # Save trained model
    model.save("model/handwritten.h5")
    return history['accuracy'][-1], len(x_train)  # Return final accuracy and dataset size
