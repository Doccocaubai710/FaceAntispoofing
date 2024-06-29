import cv2
import torch
from torchvision import transforms
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = torch.load("Model/modelVIT.pth", map_location=device)
model.eval()

face_net = cv2.dnn.readNetFromCaffe(
    "Infer/deploy.prototxt",
    "Infer/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)


# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def inference_on_frame(face_image, model):
    # Preprocess the face image
    image_tensor = preprocess(face_image)
    image_tensor = image_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor.to(device))
        pred = output.argmax(dim=1, keepdim=True).item()

    # Process the output
    label = "fake" if pred == 0 else "real"
    return label

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the camera
import cv2
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Resize the frame for better performance (optional)
        frame = cv2.resize(frame, (300, 300))

        # Convert the frame to blob format
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Set the input to the face detection model
        face_net.setInput(blob)

        # Perform face detection
        detections = face_net.forward()

        # Process the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:
                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face region
                face = frame[startY:endY, startX:endX]

                # Perform inference on the face
                label = inference_on_frame(face, model)

                if label=="real":
                    # Draw the bounding box around the face
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    # Display the label on the bounding box
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0, 255), 2)

                    # Display the label on the bounding box
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # Display the frame
        cv2.imshow('Frame', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break