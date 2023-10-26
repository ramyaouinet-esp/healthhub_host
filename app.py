from flask import Flask, render_template, request, send_file
import requests
import io
import PIL
from PIL import Image
from PIL import UnidentifiedImageError
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
import sys
import pickle
import torch
from typing import Tuple
from io import BytesIO
import base64
import plotly
import plotly.graph_objects as go
import numpy as np
import random
import nibabel as nib
import albumentations as A
import numpy as np
import plotly.graph_objects as go

from flask import Flask

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/yahyasmt/brain-tumor-3"
headers = {"Authorization": "Bearer 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"}



def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'gz', 'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', segment='index')

@app.route('/dashboard', methods=['GET', 'POST'])
def Dashboard():
    return render_template('/dashboard/index.html')

# List of prompts
prompts = [
    "brain tumor mri",
    "tumor mri",
    "brain tumor mri scan",
    "scan of a brain tumor mri",
    "scan of a brain tumor",
    "brain tumor",
    # Add more prompts as needed
]

@app.route('/MRI_Generation', methods=['GET', 'POST'])
def indeximage():
    if request.method == "POST":
        # Select a random prompt from the list
        prompt = random.choice(prompts)
        print(prompt)
        image_bytes = query({
            "inputs": prompt,
            "options": {"wait_for_model": True }
        })
        with open("static/assets_old/mdl/aa.jpeg", "wb") as image_file:
            image_file.write(image_bytes)

    return render_template('/dashboard/MRI_Generation.html')

# Process the uploaded image and display the 3D plot
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load your image (color or grayscale)
        image = plt.imread(filename)

        # Check if it's a color image
        is_color_image = len(image.shape) == 3 and image.shape[2] in [3, 4]

        # Convert color image to grayscale if needed
        if is_color_image:
            gray_image = np.mean(image, axis=2)  # Convert to grayscale
        else:
            gray_image = image

        # Normalize the image to the range [0, 1]
        gray_image = gray_image / np.max(gray_image)

        # Apply a Gaussian filter to smooth the image (adjust sigma for smoothing effect)
        sigma = 2.0
        smoothed_image = gaussian_filter(gray_image, sigma=sigma)

        # Scale the smoothed image to control the elevation magnitude
        elevation_scale = 50.0  # Adjust this value as needed
        elevated_image = smoothed_image * elevation_scale

        # Create a grid of X and Y coordinates
        x, y = np.meshgrid(np.arange(gray_image.shape[1]), np.arange(gray_image.shape[0]))

        # Create the Z coordinates using the elevated image
        z = elevated_image

        # Create a 3D surface plot of the inflated image using Plotly
        fig = go.Figure(data=[go.Surface(z=z, colorscale='Viridis')])
        fig.update_layout(scene=dict(aspectmode="data"))
        plot_div = fig.to_html(full_html=False)

        return render_template('dashboard/result.html', plot_div=plot_div)

    return redirect(request.url)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/3D')
def ThreeD():
    return render_template('dashboard/3D.html')


API_URL2 = "https://api-inference.huggingface.co/models/amjadfqs/swin-base-patch4-window7-224-in22k-finetuned-brain-tumor-final_13"
headers2 = {"Authorization": "Bearer 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"}

API_URL3 = "https://api-inference.huggingface.co/models/Locutusque/gpt2-large-medical"
headers3 = {"Authorization": "Bearer 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"}

def query3(payload):
	response = requests.post(API_URL3, headers=headers3, json=payload)
	return response.json()
	
def query2(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL2, headers=headers2, data=data)
    return response.json()

# output = query2("cats.jpg")

# Your Flask route to handle file upload and JSON response
import time
@app.route('/decttumor', methods=['GET', 'POST'])
def decttumor():
    tumor = None
    tumor2 = "Here we propose a first piste of reflection about the tumors ..."
    zero = "Name of the tumor"
    chart_image_path = None  # Initialize the chart image path as None

    if request.method == "POST":
        retries = 100
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            while retries > 0:
                response = query2(filename)
                if extract_predicted_tumor(response) == "Error extracting predicted tumor":
                    retries -= 1
                else:
                    zero = extract_predicted_tumor(response)
                    one = "what is brain tumor " + zero
                    tumor = query3({"inputs": extract_predicted_tumor(response), "options": {"wait_for_model": True}} )
                    tumor2 = tumor[0]['generated_text']
                    chart_image = create_chart(response)

        
                    # Save the chart image as a PNG file
                    with open("static/assets_old/mdl/chart.png", "wb") as image_file:
                        image_file.write(chart_image)

                    break
        else:
            return "File format not allowed"

    # Provide the chart image path for downloading
    return render_template('dashboard/dector.html',tumor=tumor2, name=zero)

def extract_predicted_tumor(response):
    try:
        # Assuming that the predicted tumor label is the one with the highest score
        tumor_with_highest_score = max(response, key=lambda x: x["score"])
        predicted_tumor = tumor_with_highest_score["label"]
        return predicted_tumor
    except Exception as e:
        # Handle any exceptions that may occur during extraction
        return "Error extracting predicted tumor"
import seaborn as sns

def create_chart(response):
    # Extract data from the JSON response
    labels = [entry['label'] for entry in response]
    scores = [entry['score'] for entry in response]

    # Create a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores)
    plt.xlabel('Label')
    plt.ylabel('Score')

    # Save the chart as an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 for embedding in HTML
    chart_image = img.read()

    return chart_image

if __name__ == "__main__":
    app.run()
