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
- Deployment-ready setup for Railway or other cloud platforms

---

## Project Structure

AI_AgriYield_Predictor/

├── app.py                                                                                                                                                  
├── preprocessing.py                                                                                                                                             
├── EDA.py                                                                                                                                                        
├── Model_training.py                                                                                                                                            
│                                                                                                                                                                 
├── Merged_Crop_Yield_Dataset.csv                                                                                                                               
├── yield_model.pkl                                                                                                                                              
├── scaler_input.pkl                                                                                                                                             
├── scaler_y.pkl                                                                                                                                                
│                                                                                                                                                               
├── crop_mapping.json                                                                                                                                       
│                                                                                                                                                   
├── static/                                                                                                                                                 
│ └── blog-yield.jpg                                                                                                                                           
│                                                                                                                                                            
├── requirements.txt                                                                                                                                         
├── Procfile                                                                                                                                                   
├── runtime.txt                                                                                                                                               
└── README.md                                                                                                                                                  

---

## 💻 How to Run Locally

 
```bash
### 1️⃣ Install Dependencies

pip install -r requirements.txt


### 2. Run the Flask App

python app.py



---

## 🧠 Tech Stack

- Python  
- pandas , numpy  
- scikit-learn , xgboost  
- Flask 
- matplotlib , seaborn  
- Deployment: Railway (Cloud Hosting)

---

## 🌾 Dataset Sources

- FAO Crop Production Data(https://www.fao.org/faostat/en/#data/QCL)  
- Kaggle Agricultural Datasets(https://www.kaggle.com)  
- Indian Government Agriculture Data(https://data.gov.in/sector/agriculture)

---

## 🚀 Deployment (Railway)

Includes:
- requirements.txt  
- Procfile 
- runtime.txt  
- app.py  
- Model.pkl files  

## 🚀 Railway Deployment Steps

1️⃣ Go to [https://railway.app](https://railway.app) and log in.  
2️⃣ Click “New Project → Deploy from GitHub Repository.”  
3️⃣ Choose your repo:  
    AI_AgriYield_Predictor-Nidadavolu-Lakshmi-Pravallika
4️⃣ Railway will automatically detect your Procfile and build the app.  
5️⃣ Wait until the logs show ✅ Deployment Successful.  
6️⃣ Open your live app using the generated link below 👇  

🔗 AI AgriYield Predictor - Live App(https://web-production-f8a1.up.railway.app)

---

## ⚙️ Files Required for Deployment

|      File        |               Description               |
|------------------|-----------------------------------------|
| app.py           | Main Flask application                  |
| requirements.txt | Python dependencies                     |
| Procfile         | Start command (`web: gunicorn app:app`) |
| runtime.txt      | Python version (e.g., `python-3.10.14`) |
| .pkl files       | Trained model and scalers               |

---

## 🌐 Live App

✅ The project is successfully deployed and live here:

🔗AI AgriYield Predictor - Live App(https://web-production-f8a1.up.railway.app)


---

## 🧑‍💻 Author

**👩‍💻 Nidadavolu Lakshmi Pravallika**  
🎓 Academic Project — Crop Yield Prediction using ML & Flask

