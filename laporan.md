# Laporan Proyek: Implementasi Sistem Rekomendasi Film Menggunakan Metode Collaborative Filtering dengan TensorFlow

## 1. Pendahuluan

Di era digital saat ini, jumlah konten yang tersedia bagi pengguna, seperti film, musik, dan berita, telah meningkat secara eksponensial. Fenomena ini menciptakan masalah baru yang dikenal sebagai *information overload*, di mana pengguna kesulitan menemukan konten yang relevan dan sesuai dengan selera mereka. Sistem rekomendasi hadir sebagai solusi untuk mengatasi masalah ini dengan cara menyaring informasi dan menyajikan item yang paling relevan kepada pengguna.

Salah satu pendekatan yang paling populer dan efektif dalam sistem rekomendasi adalah **Collaborative Filtering**. Metode ini bekerja dengan mengumpulkan dan menganalisis pola perilaku (misalnya, rating atau riwayat tontonan) dari banyak pengguna untuk membuat prediksi. Idenya adalah jika pengguna A memiliki selera yang mirip dengan pengguna B, maka sistem dapat merekomendasikan item yang disukai oleh pengguna B tetapi belum pernah dilihat oleh pengguna A.

Laporan ini akan membahas implementasi sistem rekomendasi film menggunakan metode Collaborative Filtering. Secara spesifik, model yang dibangun mengadopsi algoritma **Faktorisasi Matriks (Matrix Factorization)** yang diimplementasikan menggunakan arsitektur Jaringan Saraf Tiruan (Neural Network) dengan framework **TensorFlow**. Tujuannya adalah untuk membangun model yang mampu memprediksi rating film dan menghasilkan daftar rekomendasi yang dipersonalisasi untuk setiap pengguna.

## 2. Deskripsi Dataset

Dataset yang digunakan dalam proyek ini adalah **MovieLens Small**, sebuah dataset populer yang sering digunakan untuk penelitian sistem rekomendasi. Dataset ini terdiri dari beberapa file, namun yang digunakan dalam implementasi ini adalah:

1.  **`ratings.csv`**: Berisi data rating yang diberikan oleh pengguna terhadap film. Setiap baris data terdiri dari:
    *   `userId`: ID unik untuk setiap pengguna.
    *   `movieId`: ID unik untuk setiap film.
    *   `rating`: Rating yang diberikan, dalam skala 0.5 hingga 5.0.
    *   `timestamp`: Waktu saat rating diberikan.

2.  **`movies.csv`**: Berisi informasi detail mengenai setiap film. Setiap baris data terdiri dari:
    *   `movieId`: ID unik film, yang dapat dihubungkan dengan data di `ratings.csv`.
    *   `title`: Judul film.
    *   `genres`: Genre film (misal: Adventure|Animation|Children|Comedy|Fantasy).

Dataset ini berisi lebih dari 100.000 rating dari sekitar 600 pengguna untuk lebih dari 9.000 film, menyediakan data yang cukup untuk melatih model Collaborative Filtering.

## 3. Metodologi dan Arsitektur Model

### 3.1. Pra-pemrosesan Data

Sebelum data dapat digunakan untuk melatih model Jaringan Saraf Tiruan, beberapa langkah pra-pemrosesan perlu dilakukan:

1.  **Penggabungan Data**: Data rating dan film dimuat ke dalam DataFrame Pandas.
2.  **Encoding ID Pengguna dan Film**: `userId` dan `movieId` asli tidak dapat langsung dimasukkan ke dalam model. Keduanya perlu diubah menjadi indeks integer yang berurutan (misal: dari 0 hingga N-1). Proses ini dilakukan menggunakan `LabelEncoder` dari library Scikit-learn. Transformasi ini penting karena *Embedding Layer* pada Keras/TensorFlow memerlukan input berupa indeks integer untuk dapat bekerja dengan benar.
3.  **Pembagian Dataset**: Dataset yang sudah di-encode kemudian dibagi menjadi dua set: 80% untuk data training dan 20% untuk data testing. Ini memungkinkan model untuk dilatih pada satu set data dan dievaluasi kinerjanya pada set data lain yang belum pernah dilihat sebelumnya.

### 3.2. Arsitektur Model

Model ini mengimplementasikan algoritma **Faktorisasi Matriks** menggunakan arsitektur Jaringan Saraf Tiruan. Arsitektur ini terdiri dari beberapa komponen utama:

1.  **Input Layers**: Terdapat dua jalur input terpisah, satu untuk indeks pengguna (`user_idx`) dan satu lagi untuk indeks film (`movie_idx`).

2.  **Embedding Layers**: Ini adalah inti dari model.
    *   **User Embedding**: `user_idx` dimasukkan ke dalam sebuah `Embedding Layer` untuk menghasilkan sebuah "vektor laten pengguna" berdimensi 50. Vektor ini adalah representasi numerik yang dipelajari oleh model untuk menangkap preferensi dan selera unik dari setiap pengguna.
    *   **Movie Embedding**: `movie_idx` dimasukkan ke dalam `Embedding Layer` lainnya untuk menghasilkan "vektor laten film" berdimensi 50. Vektor ini merepresentasikan karakteristik laten dari setiap film.

3.  **Dot Product Layer**: Untuk menghasilkan prediksi rating, vektor laten pengguna dan vektor laten film digabungkan menggunakan operasi **dot product**. Hasil dari perkalian titik ini adalah sebuah nilai tunggal yang merepresentasikan prediksi rating film tersebut untuk pengguna yang bersangkutan.

Arsitektur ini secara efektif meniru dekomposisi matriks interaksi user-item menjadi dua matriks berdimensi lebih rendah (matriks embedding pengguna dan film).

## 4. Environment dan Implementasi Kode

Proyek ini diimplementasikan menggunakan bahasa Python dengan beberapa library utama:

*   **TensorFlow & Keras**: Framework utama untuk membangun dan melatih model Jaringan Saraf Tiruan.
*   **Pandas**: Digunakan untuk manipulasi dan pra-pemrosesan data.
*   **Numpy**: Digunakan untuk operasi numerik, terutama dalam persiapan data untuk model.
*   **Scikit-learn**: Digunakan untuk `LabelEncoder` dan membagi dataset.

Langkah-langkah implementasi kode adalah sebagai berikut:
1.  **Membangun Model**: Arsitektur model yang telah dijelaskan di atas dibangun menggunakan Keras Functional API.
2.  **Kompilasi Model**: Model dikompilasi dengan `adam` sebagai optimizer dan `mean_squared_error` (MSE) sebagai fungsi kerugian (loss function), yang cocok untuk tugas regresi seperti prediksi rating.
3.  **Training Model**: Model dilatih menggunakan metode `.fit()` pada data training selama 5 epoch dengan ukuran batch 64. Data testing digunakan sebagai data validasi untuk memonitor kinerja model pada setiap epoch.
4.  **Penyimpanan Model**: Setelah training selesai, model disimpan dalam format `.keras` agar dapat dimuat kembali untuk inferensi atau penggunaan di masa depan tanpa perlu training ulang.

## 5. Hasil

Kinerja model dievaluasi pada data testing menggunakan metrik **Root Mean Squared Error (RMSE)**, yang mengukur akar dari rata-rata kuadrat kesalahan antara rating prediksi dan rating aktual. Semakin rendah nilai RMSE, semakin baik kinerja model. Setelah 5 epoch pelatihan, model ini mencapai **Test RMSE sekitar 0.88-0.92**. Nilai ini menunjukkan bahwa secara rata-rata, prediksi rating yang dihasilkan oleh model memiliki selisih sekitar 0.9 poin dari rating sebenarnya pada skala 1-5.

Selain evaluasi kuantitatif, dibuat juga fungsi untuk menghasilkan rekomendasi kualitatif. Berikut adalah contoh hasil rekomendasi untuk pengguna dengan `userId` 33:

```
   movieId                                              title  predicted_rating
0     5952                                       Lord of the Rings: The Two Towers, The (2002)          5.103317
1     4993                 Lord of the Rings: The Fellowship of the Ring, The (2001)          5.063419
2     7153                                  Lord of the Rings: The Return of the King, The (2003)          4.969332
3      296                                                              Pulp Fiction (1994)          4.869331
4     2959                                                                Fight Club (1999)          4.851911
...
```

Hasil di atas menunjukkan 5 film teratas yang diprediksi akan paling disukai oleh pengguna 33, diurutkan berdasarkan `predicted_rating` tertinggi. Ini menunjukkan bahwa model berhasil mempelajari preferensi pengguna dan dapat menghasilkan daftar rekomendasi yang relevan.
