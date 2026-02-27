# FGSM-ATTACK

A full-stack demo of the Fast Gradient Sign Method (FGSM) adversarial attack on MNIST, with a FastAPI backend and a Next.js frontend.

---

## 📖 FGSM Overview

The Fast Gradient Sign Method (FGSM) is an adversarial attack technique introduced by Goodfellow et al. (2014). It generates adversarial examples by adding a small, carefully crafted perturbation to the input image in the direction of the gradient of the loss with respect to the input. This perturbation is scaled by a parameter $\epsilon$:

$$
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$

where $x$ is the input, $y$ is the true label, $J$ is the loss, and $\theta$ are the model parameters. Increasing $\epsilon$ makes the attack stronger, often causing the model to misclassify the input.

---

## 🚀 How to Run Locally

### Backend (FastAPI + PyTorch)

   ```sh
   git clone https://github.com/Ab-21651/FGSM-ATTACK.git
   cd FGSM-ATTACK
   ```
2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the API:**
   ```sh
   uvicorn app_fgsm:app --host 0.0.0.0 --port 8000
   ```
   The API will be available at [http://localhost:8000](http://localhost:8000)

### Frontend (Next.js)

1. **Install Node.js dependencies:**
   ```sh
   cd frontend
   npm install
   ```
2. **Run the frontend:**
   ```sh
   npm run dev
   ```
   The app will be available at [http://localhost:3000](http://localhost:3000)

---

## 🖥️ API Usage

- **POST /attack**
  - Input: image file (PNG/JPEG), epsilon (float, default 0.1)

---

- **Backend:** AWS EC2 (t2.micro, Free Tier)

> _Note: Due to AWS account issues, deployment is pending. All code is ready for EC2/Amplify._

---

## 📊 Observations

- The attack is effective even with small $\epsilon$ values, demonstrating the vulnerability of neural networks to adversarial perturbations.

---
```
├── app_fgsm.py          # FastAPI backend
├── model.py             # MNISTNet model
├── mnist_cnn.pth        # Trained model weights
├── requirements.txt     # Python dependencies
├── vercel.json          # (for Vercel, ignored on EC2)
├── .gitignore
└── frontend/            # Next.js frontend
    ├── ...
```

---

## ✍️ Author
- [Ab-21651](https://github.com/Ab-21651)

---

## 📑 References
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)
