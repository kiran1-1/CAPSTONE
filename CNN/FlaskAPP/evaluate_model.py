import torch
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_module import MakeDataset  # Import your custom Dataset class
from torchvision.models import ResNet50_Weights
import cv2
from PIL import Image
import numpy as np


def load_model(model_path, num_classes, device):
    # Load the ResNet-50 model structure
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {100 * test_accuracy}%')

    return test_accuracy
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    return slots

def empty_or_not(spot_bgr):
    if spot_bgr is None or spot_bgr.size == 0:
        return 1

    try:
        # Convert BGR image to RGB and PIL Image
        spot_rgb = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(spot_rgb)

        # Apply transformation
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Get model prediction
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return prediction

    except Exception as e:
        print(f"Error processing image: {e}")
        return 1

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and data paths
    model_path = "resnet50_model.pth"
    test_data_dir ='/Users/kiran_g/Documents/Capstone/final project/Parking_CNN/FlaskAPP/clf-data'
    num_classes = 3
    batch_size = 32

    # Load the test dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = MakeDataset(test_data_dir, transform=transform)
    _, _, test_dataset = torch.utils.data.random_split(dataset, [
        int(0.8 * len(dataset)),
        int(0.1 * len(dataset)),
        len(dataset) - int(0.8 * len(dataset)) - int(0.1 * len(dataset))
    ])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = load_model(model_path, num_classes, device)

    # Evaluate the model
    test_accuracy = evaluate_model(model, test_loader, device)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ####testing video
    mask_path = '/Users/kiran_g/Documents/Capstone/final project/Parking_CNN/FlaskAPP/cropped_mask.png'
    video_path = '/Users/kiran_g/Documents/Capstone/final project/Parking_CNN/FlaskAPP/cropped_video.mp4'
    # output_video_path = 'working/cropped_output_video.mp4'
    mask = cv2.imread(mask_path, 0)
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    print(spots)
    
    # Define video paths
    output_video_path = "/Users/kiran_g/Documents/Capstone/final project/Parking_CNN/output_video.mp4"

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened properly
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        exit()

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        exit()

    # Define the codec and initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))



    spots_status = [None for _ in spots]
    diffs = [None for _ in spots]
    previous_frame = None

    ret = True
    step = 50
    frame_nmr = 0

    while ret:
        disabled_parking_count = 0
        empty_parking_count = 0
        non_empty_parking_count = 0
        ret, frame = cap.read()
        if not ret:
            break

        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

            for spot_indx in arr_:
                spot = spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_indx] = spot_status

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]
            if [134, 77, 64, 34] == spots[spot_indx] and spot_status != 2:
                spot_status = 0
                spots_status[spot_indx] = spot_status
            if spot_status == 0:
                color = (255, 255, 0)
            elif spot_status == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        for status in spots_status:
            if status == 0:
                disabled_parking_count += 1
            elif status == 1:
                empty_parking_count += 1
        non_empty_parking_count = len(spots) - (disabled_parking_count + empty_parking_count)
        total_slots = len(spots)

        status_lines = [
            f'Empty: {empty_parking_count}',
            f'Disabled Parking: {disabled_parking_count}',
            f'Not Empty: {non_empty_parking_count}',
            f'Total Slots: {total_slots}'
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        line_height = 20
        padding = 10

        text_start_x = 50
        frame_height = frame.shape[0]
        text_start_y = frame_height - (len(status_lines) * line_height + padding * 2)

        y_offset = text_start_y + padding
        for line in status_lines:
            cv2.putText(frame, line, (text_start_x, y_offset), font, font_scale, (255, 255, 255), font_thickness)
            y_offset += line_height

        # Write the frame to the output video
        out.write(frame)
        frame_nmr += 1

    cap.release()
    out.release()  