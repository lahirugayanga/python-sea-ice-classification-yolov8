from flask import Flask, render_template, request, redirect, url_for
from prediction import get_prediction_m1, get_prediction_m2
from waitress import serve
from werkzeug.utils import secure_filename
import os
import shutil

app = Flask(__name__)

# Create subdirectories for m1 and m2 in the 'uploads' directory
UPLOAD_FOLDER_M1 = 'static/uploads/m1'
UPLOAD_FOLDER_M2 = 'static/uploads/m2'
os.makedirs(UPLOAD_FOLDER_M1, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_M2, exist_ok=True)

# Consistent filenames for uploaded images
FILENAME_M1 = 'uploaded_image_m1.jpg'
FILENAME_M2 = 'uploaded_image_m2.jpg'

@app.route('/')
@app.route('/index')
def index():
    # Initially, no images are shown
    return render_template('index.html', image_url_m1=None, image_url_m2=None)

@app.route('/prediction_m1', methods=['POST'])
def get_curr_prediction_m1():
    # Get the uploaded file for model m1
    if 'filename_m1' not in request.files:
        return "No file part"
    
    file = request.files['filename_m1']

    if file.filename == '':
        return "No selected file"
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER_M1, FILENAME_M1)

        # Save the new file in the m1 folder, replacing the existing one
        file.save(file_path)

        # Pass the file path to the get_prediction_m1 function
        output_image_path_m1 = get_prediction_m1(file_path)
        
        # Check if there's already a prediction result for model m2
        output_image_path_m2 = os.path.join(UPLOAD_FOLDER_M2, 'output_image_m2.jpg')
        if not os.path.exists(output_image_path_m2):
            output_image_path_m2 = None

        return render_template("index.html", image_url_m1=output_image_path_m1, image_url_m2=output_image_path_m2)

    return redirect(url_for('index'))

@app.route('/prediction_m2', methods=['POST'])
def get_curr_prediction_m2():
    # Get the uploaded file for model m2
    if 'filename_m2' not in request.files:
        return "No file part"
    
    file = request.files['filename_m2']

    if file.filename == '':
        return "No selected file"
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER_M2, FILENAME_M2)

        # Save the new file in the m2 folder, replacing the existing one
        file.save(file_path)

        # Pass the file path to the get_prediction_m2 function
        output_image_path_m2 = get_prediction_m2(file_path)

        # Check if there's already a prediction result for model m1
        output_image_path_m1 = os.path.join(UPLOAD_FOLDER_M1, 'output_image_m1.jpg')
        if not os.path.exists(output_image_path_m1):
            output_image_path_m1 = None

        return render_template("index.html", image_url_m1=output_image_path_m1, image_url_m2=output_image_path_m2)

    return redirect(url_for('index'))

@app.route('/reset_m1', methods=['POST'])
def reset_m1():
    # Remove all files in the m1 folder
    for filename in os.listdir(UPLOAD_FOLDER_M1):
        file_path = os.path.join(UPLOAD_FOLDER_M1, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    # Define the path for the m2 image output before checking existence
    output_image_path_m2 = os.path.join(UPLOAD_FOLDER_M2, 'output_image_m2.jpg')
    
    # Check if the m2 image output file exists
    if not os.path.exists(output_image_path_m2):
        output_image_path_m2 = None
    
    return render_template("index.html", image_url_m2=output_image_path_m2, image_url_m1=None)

@app.route('/reset_m2', methods=['POST'])
def reset_m2():
    # Remove all files in the m2 folder
    for filename in os.listdir(UPLOAD_FOLDER_M2):
        file_path = os.path.join(UPLOAD_FOLDER_M2, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    output_image_path_m1 = os.path.join(UPLOAD_FOLDER_M1, 'output_image_m1.jpg')
    
    if not os.path.exists(output_image_path_m1):
        output_image_path_m1 = None

    return render_template("index.html", image_url_m1=output_image_path_m1, image_url_m2=None)

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
