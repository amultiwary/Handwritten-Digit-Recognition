import os
import idx2numpy
import numpy as np

def load_dataset():
    # Move one directory up from Model_Training to Handwritten_Digit_Recognition
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

    print(f"Looking for dataset in: {dataset_path}")  # Debugging output

    # Required dataset files
    required_files = [
        "emnist-byclass-train-images-idx3-ubyte",
        "emnist-byclass-train-labels-idx1-ubyte",
        "emnist-byclass-test-images-idx3-ubyte",
        "emnist-byclass-test-labels-idx1-ubyte"
    ]

    # Verify dataset files exist
    for file in required_files:
        full_path = os.path.join(dataset_path, file)
        print(f"Checking file: {full_path}")  # Debugging output
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset file missing: {full_path}. Make sure the files are in the correct location.")

    # Load dataset
    train_images = idx2numpy.convert_from_file(os.path.join(dataset_path, "emnist-byclass-train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(os.path.join(dataset_path, "emnist-byclass-train-labels-idx1-ubyte"))
    test_images = idx2numpy.convert_from_file(os.path.join(dataset_path, "emnist-byclass-test-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(os.path.join(dataset_path, "emnist-byclass-test-labels-idx1-ubyte"))  



    # Filter to only include digits (0–9)
    digit_classes = np.arange(10)
    train_mask = np.isin(train_labels, digit_classes)
    test_mask = np.isin(test_labels, digit_classes)

    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask]
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]

    # Normalize images  
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Reshape images to (28, 28, 1)
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)



    print("Filtered for digits only.")
    print("Training Data Shape:", train_images.shape, train_labels.shape)
    print("Testing Data Shape:", test_images.shape, test_labels.shape)

    return (train_images, train_labels), (test_images, test_labels)

# Run the function to check if files are found
if __name__ == "__main__":
    load_dataset()
