# Rekomendasi Film

Proyek ini adalah sistem rekomendasi film yang memberikan rekomendasi film kepada pengguna berdasarkan peringkat yang telah mereka berikan sebelumnya. Sistem ini menggunakan pendekatan *collaborative filtering* dengan algoritma SVD. Aplikasi ini dibangun dengan Flask dan dapat diakses melalui antarmuka web atau REST API.

## Fitur

*   **Rekomendasi Berbasis Pengguna:** Dapatkan rekomendasi film untuk pengguna tertentu.
*   **Statistik Pengguna:** Lihat statistik peringkat pengguna dan film dengan peringkat tertinggi dari mereka.
*   **REST API:** Akses mesin rekomendasi secara terprogram.
*   **Antarmuka Web:** Antarmuka web sederhana untuk berinteraksi dengan sistem rekomendasi.

## Teknologi yang Digunakan

*   **Python:** Bahasa pemrograman inti.
*   **Flask:** Kerangka kerja web mikro untuk aplikasi web dan API.
*   **Pandas:** Untuk manipulasi dan analisis data.
*   **scikit-surprise:** Pustaka Python untuk membangun dan menganalisis sistem rekomendasi.
*   **Joblib:** Untuk menyimpan dan memuat model yang telah dilatih.
*   **Conda:** Untuk manajemen lingkungan.

## Pengaturan dan Instalasi

1.  **Clone repositori:**
    ```bash
    git clone https://github.com/username-anda/film-recommender.git
    cd film-recommender
    ```

2.  **Buat dan aktifkan lingkungan Conda:**
    ```bash
    conda env create -f recommender-env.yml
    conda activate recommender-env
    ```

3.  **Jalankan aplikasi Flask:**
    ```bash
    python app.py
    ```
    Aplikasi akan berjalan di `http://127.0.0.1:5000`.

## Penggunaan

### Antarmuka Web

Buka `http://127.0.0.1:5000` di browser web Anda. Anda dapat memilih ID pengguna dari menu dropdown untuk melihat informasi mereka dan mendapatkan rekomendasi film.

### Endpoint API

*   **GET /api/users**
    *   Mengembalikan daftar semua ID pengguna yang tersedia.

*   **GET /api/user/<user_id>/info**
    *   Mengembalikan informasi tentang pengguna tertentu, termasuk jumlah total peringkat, peringkat rata-rata, dan daftar 10 film dengan peringkat teratas dari mereka.

*   **GET /api/recommend/<user_id>**
    *   Mengembalikan daftar 10 rekomendasi film untuk pengguna yang ditentukan.

## Deskripsi File

*   **`app.py`**: File utama aplikasi Flask. Berisi endpoint API dan logika untuk menyajikan antarmuka web.
*   **`recommender.ipynb`**: Jupyter Notebook yang digunakan untuk melatih model SVD.
*   **`recommender-env.yml`**: File lingkungan Conda, yang berisi daftar semua dependensi yang diperlukan untuk menjalankan proyek.
*   **`models/svd_model.joblib`**: Model SVD yang sudah dilatih sebelumnya.
*   **`data/`**: Direktori ini berisi file dataset MovieLens (`movies.csv` dan `ratings.csv`).
*   **`templates/index.html`**: Template HTML untuk antarmuka web.
*   **`.gitignore`**: Menentukan file dan direktori mana yang akan diabaikan di Git.
*   **`.gitattributes`**: File atribut Git.