from dotenv import load_dotenv
from pprint import pprint
import os
from IPython import display
from ultralytics import YOLO
import os
import glob
import cv2
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_prediction_m1(image_file):
    # Initialize the model
    model_path = 'models/best_m1_v2.pt'  # Update with your actual path
    model = YOLO(model_path)

    print(f"Processing {image_file}")

    # Start timing the inference
    start_time = time.time()

    # Run inference
    results = model(image_file, conf=0.5, iou=0.6, imgsz=640)

    # Calculate the inference speed
    inference_time = time.time() - start_time

    # Process results list
    for result in results:
        im_array = result.plot()  # plot a BGR numpy array of predictions

        # Extract class probabilities if result.boxes is not None and contains detections
        if result.boxes and len(result.boxes) > 0:
            class_probs = [f"{result.names[int(cls)]}: {prob:.2f}" for cls, prob in
                           zip(result.boxes.cls, result.boxes.conf)]
            class_probs_text = "\n".join(class_probs)
        else:
            class_probs_text = ""

        # Convert BGR to RGB for matplotlib
        im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        img_height, img_width, _ = im_array_rgb.shape

        # Calculate proportional font scale and thickness
        font_scale = img_width / 800.0
        thickness = int(img_width / 400.0)

        # Annotate image with inference speed
        text = f"Prediction Speed: {inference_time:.2f} s"

        if class_probs_text:
            text += f"\n{class_probs_text}"

        # Determine the position to put the text (bottom-left corner)
        position = (10, img_height - 20 * len(text.split('\n')))

        # Set font
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)  # Red color for the text

        # Put the text on the image
        for i, line in enumerate(text.split('\n')):
            y = position[1] + i * 20
            cv2.putText(im_array_rgb, line, (position[0], y), font, font_scale, color, thickness)

        output_image_path_m1 = os.path.join('static/uploads/m1', 'output_image_m1.jpg')
        cv2.imwrite(output_image_path_m1, cv2.cvtColor(im_array_rgb, cv2.COLOR_RGB2BGR))

        return output_image_path_m1
    
def get_prediction_m2(image_file):
    # Initialize the model
    model_path = 'models/best_m2_v1.pt'  # Update with your actual path
    model = YOLO(model_path)

    print(f"Processing {image_file}")

    # Start timing the inference
    start_time = time.time()

    # Run inference
    results = model(image_file, conf=0.5, iou=0.6, imgsz=640)

    # Calculate the inference speed
    inference_time = time.time() - start_time

    # Process results list
    for result in results:
        im_array = result.plot()  # plot a BGR numpy array of predictions

        # Extract class probabilities if result.boxes is not None and contains detections
        if result.boxes and len(result.boxes) > 0:
            class_probs = [f"{result.names[int(cls)]}: {prob:.2f}" for cls, prob in
                           zip(result.boxes.cls, result.boxes.conf)]
            class_probs_text = "\n".join(class_probs)
        else:
            class_probs_text = ""

        # Convert BGR to RGB for matplotlib
        im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        img_height, img_width, _ = im_array_rgb.shape

        # Calculate proportional font scale and thickness
        font_scale = img_width / 800.0
        thickness = int(img_width / 400.0)

        # Annotate image with inference speed
        text = f"Prediction Speed: {inference_time:.2f} s"

        if class_probs_text:
            text += f"\n{class_probs_text}"

        # Determine the position to put the text (bottom-left corner)
        position = (10, img_height - 20 * len(text.split('\n')))

        # Set font
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)  # Red color for the text

        # Put the text on the image
        for i, line in enumerate(text.split('\n')):
            y = position[1] + i * 20
            cv2.putText(im_array_rgb, line, (position[0], y), font, font_scale, color, thickness)

        output_image_path_m2 = os.path.join('static/uploads/m2', 'output_image_m2.jpg')
        cv2.imwrite(output_image_path_m2, cv2.cvtColor(im_array_rgb, cv2.COLOR_RGB2BGR))

        return output_image_path_m2


# if __name__ == "__main--":
#     print('\n*** Get Image Prediction ***\n')
    
#     img = input("\nPlease input an image: ")

#     pre = get_prediction_m1(img)

#     pprint(pre)



