import cv2 # type: ignore
from keras.models import model_from_json # type: ignore
import numpy as np # type: ignore
import time

# load json file
json_file = open("CNN_SFEW.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

#load model weights from HDF5 file
model.load_weights("CNN_SFEW.h5")

# Haar Cascade classifier for detecting faces in the webcam frames.
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

# resize frams with input model.
def extract_features(image):
    feature = np.array(image)
    feature = cv2.resize(feature, (720, 576))
    feature = feature.reshape(1,720,576,3)
    return feature/255.0

webcam=cv2.VideoCapture(0)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

while True:
    ## Capture frame from the webcam
    ret, frame = webcam.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Extract face region and resize it to (48, 48)
        #image = gray[y:y+h, x:x+w]
        image = frame[y:y+h, x:x+w]
        image_resized = cv2.resize(image, (720,576))
        
        # Preprocess face image for prediction
        #img = extract_features(image_resized)
        img = extract_features(image)
        
        # Start time
        start_time = time.time()

        # Predict emotion label using the model
        pred = model.predict(img)

        # End time
        end_time = time.time()

        #Calculate time taken for prediction
        time_taken = end_time - start_time
        prediction_label = labels[pred.argmax()]
        
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display predicted emotion label near the face
        text = f"{prediction_label} ({time_taken:.2f} sec)"
        cv2.putText(frame, text, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
    
    # Display the frame with detected faces and emotion labels
    cv2.imshow("Output", frame)
    
    # Exit loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
