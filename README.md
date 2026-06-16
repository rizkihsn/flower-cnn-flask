# 🌸 Flower Classification CNN — Aplikasi Web Flask

Sistem Klasifikasi Jenis Bunga menggunakan **Convolutional Neural Network (CNN)** berbasis web dengan **Flask** dan **TensorFlow/Keras**.

## 📋 Deskripsi

Aplikasi ini memanfaatkan model CNN untuk mengklasifikasikan 5 jenis bunga berdasarkan gambar yang diunggah oleh pengguna. Model dilatih menggunakan dataset **Flowers Recognition** dari Kaggle.

### Jenis Bunga yang Dapat Dideteksi

| No | Nama Bunga | Emoji |
|----|-----------|-------|
| 1  | Daisy     | 🌼    |
| 2  | Dandelion | 🌾    |
| 3  | Rose      | 🌹    |
| 4  | Sunflower | 🌻    |
| 5  | Tulip     | 🌷    |

## 🧠 Arsitektur Model CNN

```
Input (128×128×3)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (128 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (256 neuron, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (5 kelas, Softmax)
```

## 📁 Struktur Projek

```
flower-cnn-flask/
├── app.py                  # Aplikasi Flask (backend)
├── train_model.py          # Script training model CNN
├── requirements.txt        # Dependencies Python
├── Procfile                # Deployment Heroku/Render
├── render.yaml             # Konfigurasi Render.com
├── .gitignore              # Git ignore rules
├── README.md               # Dokumentasi ini
├── Data/
│   └── flowers/            # Dataset gambar bunga
│       ├── daisy/
│       ├── dandelion/
│       ├── rose/
│       ├── sunflower/
│       └── tulip/
├── models/
│   ├── flower_model.h5     # Model CNN yang sudah dilatih
│   ├── class_names.txt     # Nama kelas bunga
│   ├── accuracy.png        # Grafik akurasi & loss
│   ├── confusion_matrix.png # Confusion matrix
│   └── classification_report.txt # Laporan klasifikasi
├── static/
│   ├── css/
│   │   └── style.css       # Stylesheet custom
│   └── uploads/            # Gambar yang diupload user
└── templates/
    ├── index.html           # Halaman utama (upload)
    ├── result.html          # Halaman hasil prediksi
    └── about.html           # Halaman tentang model
```

## 📦 Dataset

- **Sumber**: [Flowers Recognition — Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- **Jumlah Kelas**: 5 (Daisy, Dandelion, Rose, Sunflower, Tulip)
- **Split Data**: 80% Training, 20% Validasi
- **Augmentasi**: Rotasi, Zoom, Shift, Horizontal Flip

## 🛠️ Teknologi yang Digunakan

- **Python 3.10+**
- **Flask** — Web framework
- **TensorFlow / Keras** — Deep learning framework
- **NumPy** — Komputasi numerik
- **Pillow** — Pemrosesan gambar
- **Matplotlib** — Visualisasi grafik
- **Scikit-learn** — Evaluasi model (classification report, confusion matrix)
- **Bootstrap 5** — UI/UX framework
- **Gunicorn** — WSGI server untuk deployment

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/flower-cnn-flask.git
cd flower-cnn-flask
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
python app.py
```

Buka browser: **http://localhost:5000**
atau
Buka link ini : web-production-b525d.up.railway.app 

### 5. (Opsional) Training Ulang Model

```bash
python train_model.ipynb
```

## 🌐 Deployment

### Deploy ke railway (Gratis)

1. Push projek ke GitHub
2. Buka railway dan buat akun
3. Klik **New**
4. Hubungkan repository GitHub
5. Deploy akan otomatis mendeteksi
6. Klik **Deploy**

### Deploy ke Heroku

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create nama-app`
4. Push: `git push heroku main`

## 📄 Lisensi

Projek ini dibuat untuk keperluan tugas mata kuliah.
