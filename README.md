# Sea-Ice Visual Perception and Risk Management Using Deep Neural Networks

<div style="text-align:center">
  <img src="demo/Peek 2025-08-02 01-13 (online-video-cutter.com).gif" alt="alt text">
</div>

## Overview

This web-based application provides a two-stage deep learning image classification system to support sea-ice visual perception and risk management in polar navigation scenarios.

- **Module I** classifies the image perspective and conditions:
  - Forward Looking
  - Stern Looking
  - Side Looking
  - Lighting Condition
  - Irrelevant

- **Module II** further classifies valid images into:
  - Ice Images
  - Open Water
  - Objects

This tool was developed using Flask, YOLOv8, and deep neural network models trained on sea-ice image datasets.

---


## 📁 Project Structure

- `project-root/`  
  - `models/` *(Trained model files)*   
  - `static/` *(Static assets - CSS, JS)*  
  - `templates/` *(HTML templates)*  
  - `test_dataset/` *(Sample test images)*  
  - `prediction.py` *(Model prediction logic)*  
  - `requirements.txt` *(Python dependencies)*  
  - `server.py` *(Flask backend)*  
  - `.gitignore` *(Git ignore rules)* 

  ---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sea-ice-classification.git
cd sea-ice-classification
```
### 2. Set Up Environment

Make sure you have Python 3.8+ installed. You can use virtualenv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🖥️ Running the Application

Once dependencies are installed, launch the Flask app:

```bash
python server.py
```

By default, the app will be hosted on:

```bash
http://127.0.0.1:8080
```