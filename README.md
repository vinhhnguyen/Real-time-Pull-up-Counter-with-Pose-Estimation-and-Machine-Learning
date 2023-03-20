# Real-time Pull-up Counter with Pose Estimation and Machine Learning
This project uses Mediapipe and machine learning to detect body landmarks in real-time from a video stream. The project includes a Python script that captures video frames from a camera, processes the frames using Mediapipe to extract pose landmarks, and uses a trained machine learning model to classify the body landmarks as "up" or "down". The classification results are displayed in a GUI window that shows the current stage position ("up" or "down"), the number of times the position has changed from "down" to "up," and the probability of the current classification.

## Getting Started
To run this project, you will need to install the following dependencies:

Python 3
Mediapipe
OpenCV
Pillow
Pandas
Numpy
Scikit-learn
Tkinter

Once you have installed the dependencies, download or clone the repository to your local machine.

## Usage
Run the following command to start the application:

```bash
python main.py
```


The application will launch a GUI window that shows the live video stream with pose landmarks and the classification results. To reset the counter, click the "RESET" button.
