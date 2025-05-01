from flask import Flask, Response, render_template, request
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import base64
import os

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and set to evaluation mode

model_path = os.path.join("AI", "model", "gender_detector.pth")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_classes = 2
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Real-time video stream processing
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((224, 224))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Predict gender
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                label = "Male" if predicted.item() == 1 else "Female"

            # Draw rectangle and label on frame
            color = (0, 255, 0) if label == "Female" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded.")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    try:
        # Open the image
        image = Image.open(file.stream).convert("RGB")
        image_np = np.array(image)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return render_template("index.html", error="No face detected in the image.", uploaded_image=None)

        male_count = 0
        female_count = 0

        # Analyze each face
        for (x, y, w, h) in faces:
            face = image_np[y:y + h, x:x + w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((224, 224))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Predict gender
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                if predicted.item() == 1:
                    male_count += 1
                    color = (0, 255, 0)
                    cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)
                else:
                    female_count += 1
                    color = (0, 0, 255)
                    cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)

        total_count = male_count + female_count
        annotated_image = image_np.copy()
        # Encode image as base64 to display in HTML
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        img_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return render_template("index.html",
                               prediction=f"{male_count} Male(s) [green box], {female_count} Female(s) [blue box]",
                               total_count=total_count,
                               uploaded_image=img_data)

    except Exception as e:
        return render_template("index.html", error=f"Error processing the image: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
