#AgriYield Predictor

A Machine Learning based system that predicts agricultural crop yield using environmental, soil and weather features.
This project is designed to assist farmers and planners with data-driven yield estimation.

#🌾 Objective

To build an ML model that analyzes rainfall, temperature, humidity, soil nutrients and other environmental factors to forecast crop yield accurately.

#📌 Features

1.Clean and preprocess agricultural datasets

2.Feature scaling and encoding

3.ML model training using regression algorithms

4.Saved model files (.pkl) for fast prediction

5.Flask-based web application

6.Ready for deployment on Heroku/AWS/Render

#📂 Project Structure
AI_AgriYield_Predictor/
│
├── app.py                    # Flask web app for user input & prediction
├── preprocessing.py          # Data preprocessing & feature scaling
├── Model_training.py         # Model building, evaluation & saving
│
├── Merged_Crop_Yield_Dataset.csv
├── yield_model.pkl           # Trained regression model
├── scaler_input.pkl          # Input scaler
├── scaler_y.pkl              # Output scaler
│
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
