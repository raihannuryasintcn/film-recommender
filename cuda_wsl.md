# Rangkuman: Menjalankan TensorFlow dengan GPU di WSL

Berikut adalah langkah-langkah dan proses yang telah kita lalui untuk mengkonfigurasi lingkungan Windows Subsystem for Linux (WSL) agar dapat menjalankan TensorFlow dengan dukungan GPU NVIDIA.

### 1. Tujuan Awal
Tujuan utamanya adalah untuk melatih model rekomendasi film dari notebook `recommender_tensorflow.ipynb` menggunakan akselerasi GPU NVIDIA (RTX 2050) di dalam lingkungan WSL, karena TensorFlow memerlukan CUDA yang didukung penuh di lingkungan Linux.

### 2. Instalasi CUDA Toolkit di WSL
Langkah pertama adalah memastikan bahwa WSL dapat berkomunikasi dengan GPU.

- **Pemeriksaan Versi Kernel:** Kita memastikan versi kernel WSL sudah memenuhi syarat (`> 5.10.43.3`).
- **Instalasi NVIDIA CUDA Toolkit:** Kita mengikuti panduan resmi NVIDIA untuk WSL:
  1.  Mengunduh dan menginstal *keyring* NVIDIA.
      ```bash
      # Awalnya menggunakan wget, namun beralih ke curl karena wget tidak tersedia
      curl -O https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
      ```
  2.  Menginstal keyring, memperbarui daftar paket, dan menginstal CUDA Toolkit.
      ```bash
      sudo dpkg -i cuda-keyring_1.1-1_all.deb
      sudo apt-get update
      sudo apt-get install -y cuda
      ```
- **Verifikasi Driver:** Kita memverifikasi bahwa driver NVIDIA berhasil diakses dari dalam WSL menggunakan perintah `nvidia-smi`, yang menunjukkan detail GPU dan versi CUDA yang terpasang di host Windows.

### 3. Menjalankan Notebook dan Identifikasi Masalah
Setelah CUDA terpasang di level sistem, kita mencoba menjalankan notebook TensorFlow.

- **Instalasi Environment:** Anda telah menyiapkan environment Python (`tf_env`) yang berisi library seperti `jupyter`, `pandas`, dan `scikit-learn`.
- **Masalah Terdeteksi:** Saat mencoba menjalankan kode TensorFlow untuk mendeteksi GPU, hasilnya adalah `[]` (kosong). Log error menunjukkan:
  > `Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly... Skipping registering GPU devices...`
- **Akar Masalah:** Meskipun driver utama sudah ada, TensorFlow tidak dapat menemukan library spesifik yang dibutuhkannya, yaitu **cuDNN** dan **cuBLAS**, di dalam path environment Python.

### 4. Solusi: Instalasi TensorFlow Modern dengan `[and-cuda]`
Untuk mengatasi masalah ini, kita menggunakan metode instalasi TensorFlow yang lebih modern dan direkomendasikan.

1.  **Uninstall TensorFlow Lama:** Kita menghapus instalasi TensorFlow yang ada untuk memastikan tidak ada konflik.
    ```bash
    pip uninstall -y tensorflow
    ```
2.  **Install TensorFlow dengan CUDA Bundled:** Kita menginstal ulang TensorFlow menggunakan target `[and-cuda]`. Perintah ini secara otomatis mengunduh dan menginstal versi **cuDNN** dan **library CUDA lain** yang 100% kompatibel dengan versi TensorFlow yang diinstal. Library ini disimpan di dalam environment Python, sehingga TensorFlow pasti bisa menemukannya.
    ```bash
    pip install tensorflow[and-cuda]
    ```

### 5. Verifikasi Akhir dan Keberhasilan
Setelah instalasi ulang, kita melakukan verifikasi akhir di dalam JupyterLab.

- **Restart Kernel:** Langkah krusial untuk memastikan notebook memuat library yang baru diinstal.
- **Pengecekan GPU:** Menjalankan kembali sel kode verifikasi.
  ```python
  import tensorflow as tf
  print(tf.config.list_physical_devices('GPU'))
  ```
- **Hasil Sukses:** Output yang dihasilkan adalah `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`, yang mengkonfirmasi bahwa TensorFlow telah berhasil mendeteksi dan siap menggunakan GPU.
- **Konfirmasi Pelatihan:** Saat `model.fit()` dijalankan, log menunjukkan pesan-pesan positif seperti `XLA service ... initialized for platform CUDA` dan `StreamExecutor device (0): NVIDIA GeForce RTX 2050`, yang menjadi bukti final bahwa pelatihan model benar-benar berjalan di atas GPU.

Perjalanan ini menunjukkan pentingnya tidak hanya menginstal driver di level sistem, tetapi juga memastikan bahwa library spesifik seperti cuDNN tersedia di dalam environment Python yang digunakan oleh TensorFlow. Metode `pip install tensorflow[and-cuda]` adalah cara paling andal untuk mencapai ini.
