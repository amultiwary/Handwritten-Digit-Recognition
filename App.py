import tensorflow as tf 
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Store accuracy results globally
if "dataset_sizes" not in st.session_state:
    st.session_state.dataset_sizes = []
if "accuracies" not in st.session_state:
    st.session_state.accuracies = []

# Function to load model
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

# Function to preprocess image
def predictDigit(image):
    model = load_model()
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    confidence = np.max(pred[0])

    # Reject non-digit predictions
    if confidence < 0.90:
        return "Not a Digit"

    return result

# Streamlit UI
st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')
st.title(' Handwritten Digit Recognition')
st.subheader("Draw the digit on canvas and click on 'Predict Now'")

# Canvas for Drawing
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Predict Button
if st.button('Predict Now'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array[:, :, :3].astype('uint8'), 'RGB')  
        input_image = input_image.convert('L')  
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('Predicted Digit: ' + str(res))
    else:
        st.header('Please draw a digit on the canvas.')

# Train Model Function
def train_model():
    from Model_Training.Main_Model import train_main_model  
    accuracy, dataset_size = train_main_model(epochs=5)

    # Store values for graph
    st.session_state.dataset_sizes.append(dataset_size)
    st.session_state.accuracies.append(accuracy)
    st.success(f" Training Completed! Final Accuracy: {accuracy:.2%}")                                                                                                                              
# Test Model Function
def test_model():
    try:
        model = load_model()

        from Model_Training.dataset_loader import load_dataset
        _, (x_test, y_test) = load_dataset()

       

        #  One-hot encode the test labels
        y_test = to_categorical(y_test,num_classes=10)

        loss, acc = model.evaluate(x_test, y_test)
        st.success(f"Test Accuracy: {acc:.2%}")
    except Exception as e:
        st.error("Model not trained yet! Click 'Train Model' first.")
        print("Error:", e)


# Test Model with Confusion Matrix Function
def test_model_with_confusion_matrix():
    try:
        model = load_model()
        from Model_Training.dataset_loader import load_dataset
        _, (x_test, y_test) = load_dataset()

        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Make predictions
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Get true labels if one-hot encoded
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        # Filter for only digit classes (0–9)
        digit_indices = list(range(10))
        mask = np.isin(y_test, digit_indices)
        filtered_y_test = y_test[mask]
        filtered_y_pred = y_pred_classes[mask]

        # Calculate accuracy just for digits
        digit_accuracy = np.mean(filtered_y_test == filtered_y_pred)
        st.success(f"Digit Accuracy (0-9): {digit_accuracy:.2%}")

        # Generate confusion matrix for digits
        cm = confusion_matrix(filtered_y_test, filtered_y_pred, labels=digit_indices)

        # Plot confusion matrix
        st.subheader("Confusion Matrix for Digits (0-9)")
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digit_indices)
        disp.plot(cmap='Blues', values_format='d', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error("Model not trained yet! Click 'Train Model' first.")
        st.error(str(e))

# Train , Test , Test with Confusion Matrix Buttons
col1, col2,col3 = st.columns(3)

with col1:
    if st.button(' Train Model'):
        train_model()  

with col2:
    if st.button(' Test Model'):
        test_model()

with col3:
    if st.button('Test with Confusion Matrix'):
        test_model_with_confusion_matrix()

# Accuracy vs Dataset Size Graph
if st.session_state.dataset_sizes:
    fig, ax = plt.subplots()
    ax.plot(st.session_state.dataset_sizes, st.session_state.accuracies, marker='o', linestyle='-', color='b')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Dataset Size")
    st.pyplot(fig)

