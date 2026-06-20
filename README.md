# 🏦 Aplikasi Prediksi Risiko Gagal Bayar Kredit (Streamlit)

Repositori ini berisi aplikasi web interaktif **Loan Default Risk Predictor** yang dibangun menggunakan Streamlit. Aplikasi ini mendeteksi risiko gagal bayar (default) calon nasabah berdasarkan model Machine Learning (LightGBM, CatBoost, Logistic Regression) yang telah dituning.

---

## 📂 Struktur Folder App

```text
streamlit_app/
├── models/                     <-- Taruh file .pkl & .json hasil export di sini
│   ├── models_trained.pkl
│   ├── models_class_weight.pkl
│   ├── imputer.pkl
│   ├── feature_names.pkl
│   └── config.json
├── app.py                      <-- Aplikasi utama Streamlit Anda
├── requirements.txt            <-- Daftar library python yang dibutuhkan
└── .gitignore                  <-- Mengabaikan cache git
```

---

## 🛠️ Persiapan Sebelum Deploy

1. Jalankan sel **"Export Model untuk Deployment"** yang ada di bagian akhir notebook Anda di Google Colab.
2. Google Colab akan secara otomatis mengunduh 5 berkas model (`.pkl` dan `.json`).
3. Buat folder baru bernama `models` di dalam folder `streamlit_app` komputer Anda secara manual jika belum ada.
4. Salin kelima berkas yang diunduh dari Colab ke dalam folder `streamlit_app/models/` tersebut.

---

## 💻 Cara Menjalankan Aplikasi di Lokal

1. Buka CMD/Terminal di dalam folder `streamlit_app` Anda.
2. Pasang semua pustaka yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 🌐 Cara Deploy ke Streamlit Cloud

1. Buat repositori baru di akun **GitHub** Anda (misalnya: `loan-risk-app`) dan atur menjadi **Public**.
2. Push isi dari folder `streamlit_app` ini (bukan folder induk `gcolab`) langsung ke akar repositori GitHub Anda.
   * *Pastikan file `app.py` berada langsung di root repositori GitHub.*
3. Masuk ke [Streamlit Share](https://share.streamlit.io/) menggunakan akun GitHub Anda.
4. Klik **Create app**, lalu pilih repositori `loan-risk-app` yang baru saja Anda buat.
5. Set **Main file path** menjadi `app.py`.
6. Klik **Deploy!** dan tunggu 2-3 menit hingga aplikasi online. 🚀
