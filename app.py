from flask import Flask, request, jsonify, render_template_string, url_for, session, send_file
import numpy as np
import joblib
import json
import os
import csv
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "my_secret_key"

# Load model & scalers
model = joblib.load("yield_model.pkl")
input_scaler = joblib.load("scaler_input.pkl")
y_scaler = joblib.load("scaler_y.pkl")

# Load crop mapping
with open("crop_mapping.json") as f:
    crop_mapping = json.load(f)

crop_count = len(crop_mapping)

# -------------------- MAIN PAGE --------------------
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
        width: 460px;
        background: rgba(255, 255, 255, 0.96);
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
    input::placeholder {
        color: #666;
        font-size: 14px;
    }
    button {
        width: 48%;
        background: #007f1c;
        padding: 12px;
        color: #fff;
        border: none;
        border-radius: 10px;
        font-size: 18px;
        cursor: pointer;
        margin: 5px 1%;
    }
    button:hover {
        background: #005a14;
        transform: scale(1.04);
    }
    .reset-btn {
        background: #888;
    }
    .alert {
        color: red;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 10px;
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
    .info-card {
        margin-top: 25px;
        padding: 15px;
        background: #f0fff0;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(0,0,0,0.2);
        font-size: 15px;
        text-align:center;
    }
    .dev-line {
        font-weight: bold;
        color: #006400;
        font-size: 16px;
        margin-top: 8px;
        text-align:center;
    }
    footer {
        text-align:center;
        margin-top:25px;
        color:white;
        text-shadow:1px 1px 3px black;
    }
    /* Floating Icons */
    .icon-bar {
        position: fixed;
        top: 20px;
        right: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
    }
    .icon {
        width: 36px;
        height: 36px;
        cursor: pointer;
        background: #007f1c;
        border-radius: 50%;
        padding: 6px;
    }
</style>

</head>
<body>

<div class="banner">AgriYield Predictor</div>

<div class="icon-bar">
    <img src="https://cdn-icons-png.flaticon.com/512/1828/1828817.png" class="icon" onclick="window.location.href='/history'">
    <img src="https://cdn-icons-png.flaticon.com/512/724/724933.png" class="icon" onclick="window.location.href='/download'">
</div>

<div class="center-box">

  <div id="alert" class="alert"></div>

  <label>Nitrogen (0 - 140)</label>
  <input type="number" id="N" placeholder="0 - 140">

  <label>Phosphorus (5 - 100)</label>
  <input type="number" id="P" placeholder="5 - 100">

  <label>Potassium (20 - 250)</label>
  <input type="number" id="K" placeholder="20 - 250">

  <label>Temperature (10°C - 45°C)</label>
  <input type="number" id="temperature" placeholder="10 - 45">

  <label>Humidity (20% - 90%)</label>
  <input type="number" id="humidity" placeholder="20 - 90">

  <label>Soil pH (4.5 - 9.0)</label>
  <input type="number" step="any" id="ph" placeholder="4.5 - 9.0">

  <label>Rainfall (50 - 400 mm)</label>
  <input type="number" id="rainfall" placeholder="50 - 400">

  <label>Crop Type</label>
  <select id="CropType">
    {% for code, name in crop_mapping.items() %}
    <option value="{{ code }}">{{ name }}</option>
    {% endfor %}
  </select>

  <button onclick="predictYield()">Predict</button>
  <button class="reset-btn" onclick="resetForm()">Reset</button>

  <div id="result" class="result-box"></div>

  <div class="info-card">
      🌾 <b>About:</b> Predicts agricultural crop yield using ML.
  </div>

</div>

<footer>© 2025 AgriYield Predictor | Powered by Flask & Machine Learning</footer>

<script>
function validateInputs() {
  const fields = [
    {id:'N', min:0, max:140},
    {id:'P', min:5, max:100},
    {id:'K', min:20, max:250},
    {id:'temperature', min:10, max:45},
    {id:'humidity', min:20, max:90},
    {id:'ph', min:4.5, max:9},
    {id:'rainfall', min:50, max:400}
  ];

  for (let f of fields) {
    let val = parseFloat(document.getElementById(f.id).value);
    if (isNaN(val) || val < f.min || val > f.max) {
      document.getElementById("alert").innerText =
        "⚠️ " + f.id.toUpperCase() + " must be between " + f.min + " and " + f.max;
      return false;
    }
  }
  document.getElementById("alert").innerText = "";
  return true;
}

function predictYield() {
  if (!validateInputs()) return;

  const data = {
    N: parseFloat(N.value),
    P: parseFloat(P.value),
    K: parseFloat(K.value),
    temperature: parseFloat(temperature.value),
    humidity: parseFloat(humidity.value),
    ph: parseFloat(ph.value),
    rainfall: parseFloat(rainfall.value),
    CropType: parseInt(CropType.value)
  };

  fetch("/predict", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(data)
  })
  .then(r=>r.json())
  .then(d=>{
    result.innerHTML = "✅ Predicted Yield: " + d.prediction + " kg/ha";
  });
}

function resetForm() {
  document.querySelectorAll("input").forEach(i=>i.value="");
  result.innerHTML = "";
  alert.innerText = "";
}
</script>

</body>
</html>
'''

# -------------------- HISTORY PAGE --------------------
HTML_HISTORY = '''
<!DOCTYPE html>
<html>
<head>
<title>History</title>
<style>
    body { background:#f0fff4; font-family:Arial; text-align:center; padding:40px; }
    h2 { color:#007f3a; font-size:32px; margin-bottom:20px; }
    .hcard {
        background:white;
        width:750px;
        margin:auto;
        margin-bottom:15px;
        padding:15px;
        text-align:left;
        border-radius:10px;
        border-left:6px solid #00994d;
        box-shadow:0 0 10px rgba(0,0,0,0.2);
        font-size:17px;
        line-height:1.6;
    }
    .btn { padding:12px 20px; background:#00994d; color:white; border:none;
           border-radius:10px; cursor:pointer; font-size:18px; margin-top:15px; }
    .btn:hover { background:#007f3a; }
</style>
</head>
<body>

<h2>📜 Prediction History</h2>

{% if history %}
  {% for rec in history %}
      {{ rec|safe }}
  {% endfor %}
{% else %}
<p>No predictions yet.</p>
{% endif %}

<button class="btn" onclick="window.location.href='/'">⬅ Back</button>

</body>
</html>
'''

# -------------------- DOWNLOAD PAGE --------------------
HTML_DOWNLOAD = '''
<html>
<head>
<style>
    body { background:#f0fff4; font-family:Arial; text-align:center; padding-top:50px; }
    h2 { color:#007f3a; font-size:32px; }
    .dbtn {
        padding:14px 26px; margin:12px; border-radius:10px;
        background:#00994d; color:white; border:none;
        font-size:18px; cursor:pointer;
        transition:0.2s;
    }
    .dbtn:hover { background:#007f3a; transform:scale(1.05); }
    .back { background:#555; }
    .back:hover { background:#333; }
</style>
</head>

<body>

<h2>📄 Download Your Reports</h2>

<a href="/get_csv"><button class="dbtn">⬇ Download CSV</button></a><br>
<a href="/get_pdf"><button class="dbtn">⬇ Download PDF</button></a><br>

<button onclick="window.location.href='/'" class="dbtn back">⬅ Back</button>

</body>
</html>
'''

# -------------------- ROUTES --------------------

@app.route("/")
def home():
    return render_template_string(HTML_MAIN, crop_mapping=crop_mapping)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    cname = crop_mapping[str(data["CropType"])]

    X = np.array([[data["N"], data["P"], data["K"],
                   data["temperature"], data["humidity"],
                   data["ph"], data["rainfall"]]])

    Xs = input_scaler.transform(X)
    hot = np.zeros((1, crop_count))
    hot[0][data["CropType"]] = 1

    final = np.hstack([Xs, hot])
    scaled = model.predict(final)[0]
    pred = y_scaler.inverse_transform([[scaled]])[0][0]

    # Save history
    record = f"""
    <div class='hcard'>
    <b>🌾 Crop:</b> {cname}<br>
    <b>N:</b> {data['N']} | <b>P:</b> {data['P']} | <b>K:</b> {data['K']}<br>
    <b>Temp:</b> {data['temperature']}°C | <b>Humidity:</b> {data['humidity']}%<br>
    <b>pH:</b> {data['ph']} | <b>Rainfall:</b> {data['rainfall']} mm<br>
    <b>Predicted Yield:</b> {round(pred,2)} kg/ha
    </div>
    """

    history = session.get("history", [])
    history.insert(0, record)
    session["history"] = history[:15]

    # Save CSV
    with open("prediction_history.csv", "a", newline="") as c:
        writer = csv.writer(c)
        writer.writerow([cname, data["N"], data["P"], data["K"],
                         data["temperature"], data["humidity"],
                         data["ph"], data["rainfall"], round(pred,2)])

    # Save TXT for PDF
    with open("prediction_history.txt", "a") as f:
        f.write(f"{cname}, N:{data['N']}, P:{data['P']}, K:{data['K']}, ")
        f.write(f"Temp:{data['temperature']}°C Hum:{data['humidity']}%, pH:{data['ph']}, Rain:{data['rainfall']}\n")
        f.write(f"Yield:{round(pred,2)} kg/ha\n")
        f.write("-"*50 + "\n")

    return jsonify({"prediction": round(pred,2)})

@app.route("/history")
def history():
    return render_template_string(HTML_HISTORY, history=session.get("history", []))

@app.route("/download")
def download():
    return render_template_string(HTML_DOWNLOAD)

@app.route("/get_csv")
def get_csv():
    return send_file("prediction_history.csv", as_attachment=True)

@app.route("/get_pdf")
def get_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    if os.path.exists("prediction_history.txt"):
        with open("prediction_history.txt","r") as f:
            for line in f:
                pdf.multi_cell(0, 8, txt=line.strip())

    pdf.output("prediction_report.pdf")
    return send_file("prediction_report.pdf", as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
