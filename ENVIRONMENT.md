# Environment

Main runs (Google Colab): NVIDIA Tesla T4 (15 GB), Intel Xeon 2.00 GHz, 12 GB RAM, Linux kernel 6.6, Python 3.12,
TensorFlow 2.20.0, Keras 3.13.2, CUDA 12.5.1 (cuDNN 9). Sensitivity and robustness runs: Kaggle sessions (NVIDIA T4
GPU or CPU only), Python 3.12–3.13, TensorFlow 2.18–2.20.

`requirements.txt` gives minimum versions. To record the exact versions of your own run, execute
`pip freeze > requirements-lock.txt` and commit the file.
