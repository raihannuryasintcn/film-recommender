# Rangkuman Notebook: Training Model Rekomendasi Film dengan TensorFlow

Dokumen ini merangkum proses yang dilakukan dalam notebook `tensorflow_recommender.ipynb` untuk membuat, melatih, dan menggunakan model sistem rekomendasi film berbasis TensorFlow/Keras.

## 1. Tujuan
Tujuan utama dari notebook ini adalah membangun sebuah model *collaborative filtering* yang dapat memprediksi rating film untuk seorang pengguna dan memberikan rekomendasi film yang dipersonalisasi.

## 2. Proses Utama

### a. Persiapan Data
- **Load Dataset**: Memuat dua file CSV: `ratings.csv` (berisi `userId`, `movieId`, `rating`) dan `movies.csv` (berisi `movieId`, `title`).
- **Pra-pemrosesan**:
  - `userId` dan `movieId` yang asli diubah menjadi indeks integer yang berurutan (misal: 0, 1, 2, ...). Proses ini wajib dilakukan karena *Embedding Layer* di Keras memerlukan input berupa indeks integer.
  - `LabelEncoder` dari Scikit-learn digunakan untuk melakukan transformasi ini.
- **Split Data**: Dataset dibagi menjadi 80% data training dan 20% data testing untuk melatih dan mengevaluasi model.

### b. Arsitektur dan Algoritma Model
Model ini menggunakan pendekatan **Collaborative Filtering**, di mana rekomendasi dibuat berdasarkan pola rating dari banyak pengguna. Secara spesifik, algoritma yang diimplementasikan adalah **Faktorisasi Matriks (Matrix Factorization)** yang dibangun menggunakan arsitektur Jaringan Saraf Tiruan (Neural Network) di TensorFlow/Keras.

Cara kerjanya adalah sebagai berikut:
1.  **Input**: Model menerima dua input terpisah: indeks pengguna (`user_idx`) dan indeks film (`movie_idx`).
2.  **Embedding Layers (Pembelajaran Fitur Laten)**:
    - Model tidak menggunakan fitur eksplisit seperti genre atau aktor. Sebaliknya, ia mempelajari fitur-fitur tersembunyi (laten) untuk setiap pengguna dan film.
    - Terdapat dua `Embedding` layer. Satu untuk memetakan setiap `user_idx` ke sebuah **vektor laten pengguna** (representasi numerik dari preferensi user).
    - Satu lagi untuk memetakan setiap `movie_idx` ke sebuah **vektor laten film** (representasi numerik dari karakteristik film).
    - Dimensi dari vektor embedding ini diatur ke `50`.
3.  **Prediksi (Dot Product)**: Vektor laten dari pengguna dan film digabungkan menggunakan operasi **dot product**. Hasil dari operasi ini adalah prediksi rating, yang menunjukkan seberapa cocok film tersebut untuk pengguna.

### c. Training dan Evaluasi
- **Kompilasi**: Model dikompilasi menggunakan:
  - **Optimizer**: `adam`.
  - **Loss Function**: `mean_squared_error` (MSE), yang cocok untuk tugas regresi seperti prediksi rating.
- **Training**: Model dilatih selama `5` epoch dengan `batch_size` sebesar `64`. Hyperparameter ini dipilih agar proses training tidak memakan waktu terlalu lama.
- **Evaluasi**: Kinerja model diukur pada data testing menggunakan metrik **Root Mean Squared Error (RMSE)** untuk melihat seberapa besar rata-rata kesalahan prediksi rating.

### d. Penyimpanan Model
Setelah dilatih, model disimpan ke dalam file `models/tf_model.keras` agar dapat dimuat dan digunakan kembali di masa depan tanpa perlu melakukan training ulang.

### e. Fungsi Rekomendasi
- Dibuat sebuah fungsi bernama `recommend_movies_for_user_tf`.
- Fungsi ini menerima `user_id` sebagai input dan melakukan langkah-langkah berikut:
  1.  Mengidentifikasi semua film yang **belum** ditonton oleh pengguna tersebut.
  2.  Menggunakan model yang telah dilatih untuk **memprediksi rating** untuk semua film yang belum ditonton.
  3.  Mengurutkan film berdasarkan prediksi rating dari yang tertinggi ke terendah.
  4.  Mengembalikan **N film teratas** beserta judul dan prediksi ratingnya sebagai hasil rekomendasi.
- Notebook ini juga menyertakan contoh cara menggunakan fungsi ini untuk mendapatkan rekomendasi bagi `userId` 33.
