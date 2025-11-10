# AgriYield Predictor

A Machine Learning project that predicts crop yield based on environmental, soil, and weather features. 
This system helps farmers and planners estimate crop yield using user inputs through a Flask-based web application.

---

## Objective

The goal of this project is to:
- Predict agricultural crop yield using regression models.
- Analyze parameters such as rainfall, temperature, humidity, soil nutrients (NPK), and soil type.
- Provide yield prediction through a simple web interface.

---

## Features

- Data preprocessing (cleaning, encoding, scaling)
- Regression model training
- Saved model files for fast prediction (.pkl)
- Flask web app for user interaction
- Deployment-ready setup (Heroku/Render/AWS)

---

## Project Structure

AI_AgriYield_Predictor/  
│  
├── app.py  
├── preprocessing.py  
├── Model_training.py  
│  
├── Merged_Crop_Yield_Dataset.csv  
├── yield_model.pkl  
├── scaler_input.pkl  
├── scaler_y.pkl  
│  
├── requirements.txt  
├── Procfile  
├── runtime.txt  
└── README.md  

---

## How to Run Locally

### 1. Install Dependencies

pip install -r requirements.txt


### 2. Run the Flask App


### 3. Open in Browser
http://127.0.0.1:5000/

---

## Tech Stack

- Python  
- pandas, numpy  
- scikit-learn  
- Flask  
- matplotlib / seaborn  
- Deployment: Heroku / Render / AWS  

---

## Dataset Sources

- FAO Crop Production  
  https://www.fao.org/faostat/en/#data/QCL

- Kaggle Agricultural Datasets  
  https://www.kaggle.com

- Indian Government Agriculture Data  
  https://data.gov.in/sector/agriculture

---

## Deployment

Includes:
- requirements.txt  
- Procfile  
- runtime.txt  

### Heroku Deployment Steps

heroku login
heroku create
git push heroku main
heroku open


---
