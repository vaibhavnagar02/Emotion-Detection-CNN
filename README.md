**Facial Emotion Recognition using Convolutional Neural Network (CNN)**

This repository contains code to train a Convolutional Neural Network (CNN) for facial emotion recognition. The model is trained on a dataset of facial images labeled with different emotions. The CNN processes the images to classify them into one of seven emotion categories: Surprise, Fear, Angry, Neutral, Sad, Disgust, and Happy.

**Project Structure**

train/: Directory containing training images categorized into subfolders by emotion.
validation/: Directory containing validation images categorized into subfolders by emotion.
main.py: Main script to prepare data, define the CNN model, and train the model.
Prerequisites

Python 3.6+
Required libraries: OpenCV, NumPy, TensorFlow, Keras
You can install the required libraries using the following command:

bash
Copy code
pip install opencv-python numpy tensorflow keras
How to Use

**Data Preparation:**

Ensure your training and validation datasets are organized in subfolders by emotion within the train and validation directories, respectively.
The script main.py loads and preprocesses these images, resizing them to 96x96 pixels and normalizing the pixel values.
Model Definition:

The CNN model consists of three convolutional layers followed by max pooling layers, and two fully connected layers with dropout to reduce overfitting.
Training the Model:

The script compiles the model using the Adam optimizer and categorical cross-entropy loss.
The model is trained for 20 epochs with a batch size of 64.
Training and validation data are fed into the model, and accuracy is tracked during training.
Steps to Run the Code

**Clone the repository:**

bash
Copy code
git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition
Ensure the dataset is correctly placed in the train and validation directories.

Run the training script:

bash
Copy code
python main.py
Detailed Description of main.py

**Data Preparation**
The script lists all files in the train and validation directories, excluding any .DS_Store files. It loads and resizes images, normalizes pixel values, and stores them in NumPy arrays.

**Label Encoding**
Labels for the training and validation datasets are created based on the number of images in each subfolder. These labels are one-hot encoded for categorical classification.

**Model Definition**
The CNN model is defined using Keras:

Convolutional Layers: Extract features from images.
Max Pooling Layers: Reduce spatial dimensions.
Fully Connected Layers: Classify the images based on extracted features.
Dropout Layers: Prevent overfitting.
Training
The model is compiled and trained using the prepared data, with accuracy tracked on both training and validation datasets.

Results

Training history, including loss and accuracy, is stored in the history object, which can be used for further analysis and visualization.

Acknowledgements

This project uses the following libraries:

OpenCV: For image processing
NumPy: For numerical operations
TensorFlow/Keras: For building and training the neural network
