import cv2
import numpy as np
import dlib
import os

# Set the working directory for the pre-trained models
os.chdir(r'C:\Project\skinton\Age-and-Gender-Recognition\models')

# Load pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained upper body detection cascade
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Load pre-trained eye detection cascade (for glasses)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load pre-trained gender detection model
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderList = ['Male', 'Female']

# Load pre-trained age detection model
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
ageNet = cv2.dnn.readNet(ageModel, ageProto)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load facial landmark predictor
predictor = dlib.shape_predictor(r"C:\Project\skinton\shape_predictor_68_face_landmarks.dat")

# Load pretrained clothing classifier (replace with your actual clothing classifier)
# Placeholder code:
def classify_clothing(image):
    # Placeholder code to simulate clothing classification
    clothing_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return np.random.choice(clothing_classes)

# Load pretrained accessory detector (replace with your actual accessory detector)
# Placeholder code:
def detect_accessories(image):
    # Placeholder code to simulate accessory detection
    accessories = ['Watch', 'Glasses', 'Necklace', 'Hat', 'Scarf', 'Earrings']
    return [accessory for accessory in accessories if np.random.rand() > 0.5]

# Create a window to display the camera feed
cv2.namedWindow('Camera Output')

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Open a text file for writing
output_file = open(r'C:\Project\skinton\gt.txt', 'w')

# Process the video frames
keyPressed = -1  # -1 indicates no key pressed
padding = 20

while keyPressed < 0:  # any key pressed has a value >= 0
    # Grab video frame, decode it and return next video frame
    readSuccess, sourceImage = videoFrame.read()

    if not readSuccess:
        print("Error: Unable to read video frame")
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw yellow bounding box around faces and mention the color, gender, and skin color
    for (x, y, w, h) in faces:
        cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Draw yellow bounding box

        # Detect facial landmarks
        landmarks = predictor(gray, dlib.rectangle(x, y, x + w, y + h))

        # Get RGB color value of the detected face region
        face_roi = sourceImage[y:y + h, x:x + w]
        avg_face_color_per_row = np.mean(face_roi, axis=0)
        avg_face_color = np.mean(avg_face_color_per_row, axis=0)
        avg_face_color = np.round(avg_face_color[::-1])  # Convert from BGR to RGB

        # Predict gender based on facial features
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict age based on facial features
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Display color values for face on the bounding box
        cv2.putText(sourceImage, f'Face Color (RGB): {avg_face_color}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(sourceImage, f'Gender: {gender}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(sourceImage, f'Age: {age}', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Detect eyes (for glasses)
        eyes = eye_cascade.detectMultiScale(face_roi)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (            ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            # You can classify glasses here based on the eye detection

        # Placeholder code to classify clothing
        clothing_class = classify_clothing(face_roi)
        cv2.putText(sourceImage, f'Clothing: {clothing_class}', (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Placeholder code to detect accessories
        detected_accessories = detect_accessories(face_roi)
        accessory_str = ', '.join(detected_accessories)
        cv2.putText(sourceImage, f'Accessories: {accessory_str}', (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Write detected information to the text file
        output_file.write(f'Face at ({x}, {y}) - Gender: {gender}, Age: {age}, Face Color (RGB): {avg_face_color}, '
                          f'Clothing: {clothing_class}, Accessories: {accessory_str}\n')

    # Detect upper bodies (clothing)
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw yellow bounding box around upper bodies and mention the color
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Draw yellow bounding box

        # Get RGB color value of the detected upper body region
        body_roi = sourceImage[y:y + h, x:x + w]
        avg_body_color_per_row = np.mean(body_roi, axis=0)
        avg_body_color = np.mean(avg_body_color_per_row, axis=0)
        avg_body_color = np.round(avg_body_color[::-1])  # Convert from BGR to RGB

        # Display color value for clothing on the bounding box
        cv2.putText(sourceImage, f'Clothing Color (RGB): {avg_body_color}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    # Write the frame into the file 'output.avi'
    if out.isOpened():
        out.write(sourceImage)
    else:
        print("Error: Video writer is not opened.")

    # Display the source image
    cv2.imshow('Camera Output', sourceImage)

    # Check for user input to close program
    keyPressed = cv2.waitKey(1)  # wait 1 milisecond in each iteration of while loop

    # Check if the close button (cross) on the window is pressed
    if cv2.getWindowProperty('Camera Output', cv2.WND_PROP_VISIBLE) < 1:
        keyPressed = 1  # Exit the loop if window is closed

# Close the output file
output_file.close()

# Release the VideoCapture and VideoWriter objects
videoFrame.release()
out.release()

# Close window
cv2.destroyAllWindows()

