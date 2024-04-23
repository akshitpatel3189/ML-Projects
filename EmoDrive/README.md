
# EmoDrive: Emotional State Detection for Safer Driving

This project dectect facial emotion with two Deep learning models Convolutional Neural Network (CNN) and a VGG16, which is a 16-
layer model.

These two models are trained on  two datasets, the Static Facial
Expressions in the Wild (SFEW) dataset and the Face Expression Recognition (FER) from Kaggle plateform.

Additionally, to test the actual accuracy of models, trained models are loaded into realTimeDetection.py Pyton script and predict emotions in real-time videos fed through webcam.

## Notes
To run the Python script you need to Run the CNN and VGG ipynb file to download the JSON and H5 file. Due to size issue I could not upload the files.

Furthermore, due to copyright issue I can't upload SFEW dataset.\
here is link to obtain SFEW dataset: https://cs.anu.edu.au/few/AFEW.html \
Here is link of FER Kaggle dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data

# Tools and Libraries

Keras\
Pandas\
OS\
Matplotlib\
TensorFlow