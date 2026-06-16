from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ==========================================
# INISIALISASI FLASK
# ==========================================

app = Flask(__name__)

# ==========================================
# KONFIGURASI
# ==========================================

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/flower_model.h5"
CLASS_NAMES_PATH = "models/class_names.txt"

IMG_SIZE = 128

# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Buat folder uploads otomatis
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Maks 16MB

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model berhasil dimuat.")

# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

print("Class names:", class_names)

# ==========================================
# FUNGSI VALIDASI FILE
# ==========================================

def allowed_file(filename):
    """Cek apakah ekstensi file diperbolehkan."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# FUNGSI PREDIKSI
# ==========================================

def predict_flower(img_path):
    """
    Melakukan prediksi jenis bunga dari gambar.
    Mengembalikan: predicted_class, confidence, all_predictions
    """

    # Load dan resize gambar
    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    # Konversi ke array dan normalisasi
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array, verbose=0)

    # Ambil indeks kelas dengan probabilitas tertinggi
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)

    # Semua probabilitas kelas (untuk ditampilkan di halaman result)
    all_predictions = []
    for i, class_name in enumerate(class_names):
        all_predictions.append({
            "name": class_name.capitalize(),
            "confidence": round(float(prediction[0][i] * 100), 2)
        })

    # Urutkan dari yang tertinggi
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return predicted_class, confidence, all_predictions

# ==========================================
# HALAMAN UTAMA
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# HALAMAN ABOUT
# ==========================================

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================================
# PREDIKSI GAMBAR
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Cek apakah file ada dalam request
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]

    # Cek apakah file dipilih
    if file.filename == "":
        return redirect(url_for("home"))

    # Validasi ekstensi file
    if not allowed_file(file.filename):
        return redirect(url_for("home"))

    # Simpan file
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )
    file.save(filepath)

    # Prediksi
    flower, confidence, all_predictions = predict_flower(filepath)

    return render_template(
        "result.html",
        image_path=filepath,
        flower=flower.capitalize(),
        confidence=round(confidence, 2),
        all_predictions=all_predictions
    )

# ==========================================
# JALANKAN FLASK
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )