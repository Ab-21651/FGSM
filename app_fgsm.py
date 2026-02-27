"""
app_fgsm.py — FastAPI Service for FGSM Adversarial Attack

Provides a REST API endpoint to perform FGSM attacks on uploaded images
using a pretrained MNIST CNN model.

Endpoints:
    POST /attack — Upload an image and epsilon to get adversarial results.

Usage:
    uvicorn app_fgsm:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import MNISTNet
from fgsm import Attack

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FGSM Adversarial Attack API",
    description=(
        "REST API for demonstrating the Fast Gradient Sign Method (FGSM) "
        "adversarial attack on a pretrained MNIST CNN model."
    ),
    version="1.0.0",
)

# CORS — allow the Next.js frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model & Attack Setup (loaded once at startup)
# ---------------------------------------------------------------------------

import os

# Resolve paths relative to this file (needed for Vercel serverless)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MNISTNet().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

# Initialize attack instance
attack = Attack(model=model, loss_fn=nn.CrossEntropyLoss(), device=DEVICE)

# Image preprocessing — same transform used during training
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # scales pixel values to [0, 1]
])

# ---------------------------------------------------------------------------
# Response Schema
# ---------------------------------------------------------------------------


class AttackResponse(BaseModel):
    """JSON response schema for the /attack endpoint."""
    clean_prediction: int
    adversarial_prediction: int
    adversarial_image_base64: str
    attack_success: bool
    epsilon: float


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """
    Convert a (1, 1, 28, 28) tensor to a base64-encoded PNG string.

    Args:
        tensor: Image tensor with values in [0, 1].

    Returns:
        Base64-encoded PNG string.
    """
    # Squeeze to (28, 28), move to CPU, convert to numpy
    img_array = tensor.squeeze().cpu().detach().numpy()
    # Scale to 0-255 and convert to PIL Image
    img_pil = Image.fromarray((img_array * 255).astype("uint8"), mode="L")
    # Encode to PNG bytes
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def load_and_preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """
    Load raw image bytes, convert to a preprocessed tensor ready for the model.

    Args:
        file_bytes: Raw bytes of the uploaded image (PNG/JPEG).

    Returns:
        Tensor of shape (1, 1, 28, 28) with values in [0, 1].

    Raises:
        HTTPException: If the image cannot be processed.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
        tensor = preprocess(image).unsqueeze(0)  # (1, 1, 28, 28)
        return tensor
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded image: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Health check / welcome endpoint."""
    return {
        "message": "FGSM Adversarial Attack API is running.",
        "usage": "POST /attack with an image file and optional epsilon value.",
    }


@app.post("/attack", response_model=AttackResponse)
async def perform_attack(
    file: UploadFile = File(..., description="Image file (PNG/JPEG) to attack"),
    epsilon: float = Form(default=0.1, description="Perturbation magnitude (0.0 – 1.0)"),
):
    """
    Perform an FGSM adversarial attack on the uploaded image.

    **Input:**
    - `file`: An uploaded image (PNG or JPEG). Will be resized to 28×28 grayscale.
    - `epsilon`: Perturbation strength (default 0.1). Range: 0.0 to 1.0.

    **Output (JSON):**
    - `clean_prediction`: Model's prediction on the original image.
    - `adversarial_prediction`: Model's prediction on the adversarial image.
    - `adversarial_image_base64`: Base64-encoded PNG of the adversarial image.
    - `attack_success`: Whether the attack changed the prediction.
    - `epsilon`: The epsilon value used.
    """
    # Validate epsilon
    if not (0.0 <= epsilon <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="Epsilon must be between 0.0 and 1.0.",
        )

    # Validate file type
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PNG or JPEG.",
        )

    # Read and preprocess the image
    file_bytes = await file.read()
    image_tensor = load_and_preprocess_image(file_bytes)

    # Get clean prediction to use as the label for the attack
    with torch.no_grad():
        clean_output = model(image_tensor.to(DEVICE))
        clean_pred = clean_output.argmax(dim=1).item()

    # Use the clean prediction as the target label for FGSM
    # (untargeted attack — we try to move the prediction away from clean_pred)
    label_tensor = torch.tensor([clean_pred], dtype=torch.long)

    # Run FGSM attack
    adv_image, clean_prediction, adversarial_prediction, success = attack.generate(
        image_tensor, epsilon, label_tensor
    )

    # Convert adversarial image to base64
    adv_base64 = tensor_to_base64_png(adv_image)

    return AttackResponse(
        clean_prediction=clean_prediction,
        adversarial_prediction=adversarial_prediction,
        adversarial_image_base64=adv_base64,
        attack_success=success,
        epsilon=epsilon,
    )
