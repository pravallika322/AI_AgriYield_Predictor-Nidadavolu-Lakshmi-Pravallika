from flask import Flask, request, jsonify, render_template_string, url_for, session, send_file
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
app.secret_key = "my_secret_key"

# Load model and scalers
model = joblib.load("yield_model.pkl")
input_scaler = joblib.load("scaler_input.pkl")
y_scaler = joblib.load("scaler_y.pkl")

# Load crop mapping
with open("crop_mapping.json") as f:
    crop_mapping = json.load(f)

crop_count = len(crop_mapping)


# ---------- MAIN PAGE ----------
HTML_MAIN = '''
<!DOCTYPE html>
<html>
<head>
<title>AgriYield Predictor 🌾</title>

<style>
    body {
        font-family: Arial;
        background: url("{{ url_for('static', filename='blog-yield.jpg') }}") no-repeat center center fixed;
        background-size: cover;
    }
    .banner {
        text-align:center;
        font-size: 42px;
        font-weight:bold;
        color: white;
        margin-top: 25px;
        margin-bottom: 25px;
        text-shadow: 3px 3px 8px black;
    }
    .center-box {
        margin: auto;
        width: 440px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 18px;
        padding: 35px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.35);
        text-align: center;
    }
    label {
        font-weight: bold;
        color: #004d14;
        display: block;
        text-align: left;
    }
    input, select {
        width: 100%;
        padding: 10px;
        margin-bottom: 12px;
        border-radius: 6px;
        border: 1px solid green;
    }
    button {
        width: 100%;
        background: #007f1c;
        padding: 12px;
        color: #fff;
        border: none;
        border-radius: 10px;
        font-size: 18px;
        cursor: pointer;
    }
    button:hover {
        background: #005a14;
        transform: scale(1.04);
    }
    .result-box {
        margin-top: 20px;
        background: #ddffdb;
        padding: 15px;
        font-size: 20px;
        font-weight: bold;
        border-left: 6px solid #049b24;
        color: #026b1f;
        text-align:center;
        border-radius:6px;
    }

    /* Floating Icon Buttons */
    .icon-bar {
        position: fixed;
        top: 20px;
        right: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
        z-index: 1000;
    }
    .icon-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .icon {
        width: 36px;
        height: 36px;
        cursor: pointer;
        background: #007f1c;
        border-radius: 50%;
        padding: 6px;
        box-shadow: 0 0 6px rgba(0,0,0,0.5);
        transition: transform 0.2s, background 0.2s;
    }
    .icon:hover {
        transform: scale(1.12);
        background: #005a14;
    }
    .icon-label {
        color: white;
        font-weight: bold;
        text-shadow: 2px 2px 5px black;
        font-size: 13px;
        margin-top: 3px;
    }
</style>

</head>
<body>

<div class="banner">AgriYield Predictor</div>

<!-- Floating Icons -->
<div class="icon-bar">
    <div class="icon-wrapper">
        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828817.png" class="icon" title="View History" onclick="window.location.href='/history'">
        <div class="icon-label">History</div>
    </div>
    <div class="icon-wrapper">
        <img src="https://cdn-icons-png.flaticon.com/512/724/724933.png" class="icon" title="Download Report" onclick="window.location.href='/download'">
        <div class="icon-label">Download</div>
    </div>
</div>

<div class="center-box">

  <label>Nitrogen (N)</label>
  <input type="number" id="N" required>

  <label>Phosphorus (P)</label>
  <input type="number" id="P" required>

  <label>Potassium (K)</label>
  <input type="number" id="K" required>

  <label>Temperature (°C)</label>
  <input type="number" id="temperature" required>

  <label>Humidity (%)</label>
  <input type="number" id="humidity" required>

  <label>Soil pH</label>
  <input type="number" id="ph" step="any" required>

  <label>Rainfall (mm)</label>
  <input type="number" id="rainfall" required>

  <label>Crop Type</label>
  <select id="CropType">
    {% for code, name in crop_mapping.items() %}
    <option value="{{ code }}">{{ name }}</option>
    {% endfor %}
  </select>

  <button onclick="predictYield()">Predict Yield</button>

  <div id="result" class="result-box"></div>
</div>

<script>
function predictYield() {
  const data = {
    N: parseFloat(document.getElementById('N').value),
    P: parseFloat(document.getElementById('P').value),
    K: parseFloat(document.getElementById('K').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    humidity: parseFloat(document.getElementById('humidity').value),
    ph: parseFloat(document.getElementById('ph').value),
    rainfall: parseFloat(document.getElementById('rainfall').value),
    CropType: parseInt(document.getElementById('CropType').value)
  };

  fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("result").innerHTML =
      "✅ Predicted Yield: " + data.prediction + " kilograms/hectare";
  });
}
</script>

</body>
</html>
'''


# ---------- HISTORY PAGE ----------
HTML_HISTORY = '''
<!DOCTYPE html>
<html>
<head>
<title>Prediction History</title>
<style>
    body {
        font-family: Arial;
        background: url("{{ url_for('static', filename='blog-yield.jpg') }}") no-repeat center center fixed;
        background-size: cover;
        color: #003300;
        text-align: center;
    }
    .container {
        width: 700px;
        margin: auto;
        margin-top: 60px;
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.35);
    }
    h2 {
        color: #007f1c;
        text-shadow: 1px 1px 3px white;
    }
    .history-item {
        background: #e6ffe6;
        border-left: 6px solid #04a124;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        text-align: left;
        font-size: 17px;
        line-height: 1.6;
    }
    .back-btn {
        background: #007f1c;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 15px;
        cursor: pointer;
    }
    .back-btn:hover {
        background: #005a14;
    }
</style>
</head>
<body>
<div class="container">
  <h2>📜 Prediction History</h2>
  {% if history %}
    {% for record in history %}
      <div class="history-item">{{ record|safe }}</div>
    {% endfor %}
  {% else %}
      <p>No history available yet.</p>
  {% endif %}
  <button class="back-btn" onclick="window.location.href='/'">⬅ Back to Predictor</button>
</div>
</body>
</html>
'''


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template_string(HTML_MAIN, crop_mapping=crop_mapping)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    crop_name = crop_mapping[str(data["CropType"])]

    X_num = np.array([[data["N"], data["P"], data["K"],
                       data["temperature"], data["humidity"],
                       data["ph"], data["rainfall"]]])
    num_scaled = input_scaler.transform(X_num)
    crop_onehot = np.zeros((1, crop_count))
    crop_onehot[0][data["CropType"]] = 1
    final_features = np.hstack([num_scaled, crop_onehot])

    scaled_pred = model.predict(final_features)[0]
    real_pred = y_scaler.inverse_transform([[scaled_pred]])[0][0]

    # Save in session
    history = session.get("history", [])
    record = f"<b>🌾 Crop:</b> {crop_name} | <b>N:</b> {data['N']} | <b>P:</b> {data['P']} | <b>K:</b> {data['K']} | <b>Temp:</b> {data['temperature']}°C | <b>Humidity:</b> {data['humidity']}% | <b>pH:</b> {data['ph']} | <b>Rainfall:</b> {data['rainfall']} mm → <b>Yield:</b> {round(float(real_pred), 2)} kg/ha"
    history.insert(0, record)
    session["history"] = history[:10]

    # Save in text file
    with open("prediction_history.txt", "a") as f:
        f.write(f"Crop: {crop_name}\n")
        f.write(f"N: {data['N']}, P: {data['P']}, K: {data['K']}, Temp: {data['temperature']}°C, Humidity: {data['humidity']}%, pH: {data['ph']}, Rainfall: {data['rainfall']} mm\n")
        f.write(f"Predicted Yield: {round(float(real_pred), 2)} kg/ha\n")
        f.write("-" * 60 + "\n")

    return jsonify({"prediction": round(float(real_pred), 2)})


@app.route("/history")
def history_page():
    history = session.get("history", [])
    return render_template_string(HTML_HISTORY, history=history)


@app.route("/download")
def download_history():
    file_path = "prediction_history.txt"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("No predictions yet.")
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
