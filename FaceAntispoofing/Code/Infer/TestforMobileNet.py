import cv2
import torch
from torchvision import transforms
import cvzone
import torch.nn as nn
from cvzone.FaceDetectionModule import FaceDetector
from torchvision.models import mobilenet_v2
# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SpoofNet(nn.Module):
    def __init__(self):
        super(SpoofNet,self).__init__()
        self.pretrained_net = mobilenet_v2(pretrained=True)
        self.features = self.pretrained_net.features
        self.conv2d = nn.Conv2d(1280, 32, kernel_size=(3, 3), padding=1)  # Adjust input channels if needed
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.features(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
# Load the pre-trained model
model=SpoofNet()
model.to(device)
checkpoint = torch.load("Model/mobilenetv2-best.pt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
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
        pred = (output > 0.5).int()

    # Process the output
    label = "fake" if pred == 0 else "real"
    return label

# Initialize the camera
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
# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
