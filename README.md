# Anomaly Detection with VAE and GAN
This repository contains two implementations for anomaly detection on image data using Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN), specifically tailored for detecting anomalies in computer vision tasks. Each model is trained on normal images to learn their representations and then detects anomalies based on deviations from these learned patterns.

## Repository Contents
- `vae.py`: Implements a VAE model that learns to reconstruct normal images. Anomalies are detected based on high reconstruction error.
- `gan.py`: Implements a DCGAN model with a generator and discriminator for anomaly detection. The discriminator's confidence score serves as an anomaly indicator.

## Usage 
1. Clone the repository:
```
git clone https://github.com/yanagriw/Anomaly_Detection.git
cd anomaly-detection
```
2. Run the Models:
- VAE:
  ```
  python vae.py --train  # Train the VAE
  python vae.py  # Evaluate the VAE on test data
  ```
- GAN:
  ```
  python gan.py --train  # Train the GAN
  python gan.py  # Evaluate the GAN on test data
  ```
## Model Saving
Each model saves the trained weights as:
- VAE: `vae_model/`
- GAN: `gan_model/`
