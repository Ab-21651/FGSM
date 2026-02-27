"""
fgsm.py — Fast Gradient Sign Method (FGSM) Attack

Implementation of the FGSM adversarial attack introduced by Goodfellow et al. (2014)
in "Explaining and Harnessing Adversarial Examples".

The attack perturbs an input image by adding a small perturbation in the direction
of the gradient of the loss with respect to the input:

    x_adv = x + epsilon * sign(∇_x J(θ, x, y))

This is a single-step attack that is fast to compute and effective at fooling
neural network classifiers.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class Attack:
    """
    Encapsulates the Fast Gradient Sign Method (FGSM) adversarial attack.

    This class is model-agnostic — it accepts any PyTorch nn.Module classifier
    and applies FGSM to craft adversarial examples that attempt to fool the model.

    Attributes:
        model (nn.Module): The target classifier to attack.
        loss_fn (nn.Module): The loss function used to compute gradients.
        device (torch.device): The device to perform computations on (cpu or cuda).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the FGSM Attack.

        Args:
            model: A PyTorch classifier (nn.Module). Must output raw logits
                   or log-probabilities.
            loss_fn: Loss function for computing gradients.
                     Defaults to CrossEntropyLoss.
            device: Device for computation.
                    Defaults to CUDA if available, else CPU.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def fgsm_attack(
        self,
        image: torch.Tensor,
        epsilon: float,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core FGSM attack: computes the adversarial perturbation and
        returns the adversarial image.

        Steps:
            1. Enable gradient tracking on the input image.
            2. Forward pass through the model to get predictions.
            3. Compute loss between predictions and the true label.
            4. Backpropagate to get ∇_x J(θ, x, y).
            5. Compute perturbation: epsilon * sign(gradient).
            6. Add perturbation to original image.
            7. Clamp to valid pixel range [0, 1].

        Args:
            image: Input image tensor of shape (1, C, H, W), values in [0, 1].
            epsilon: Perturbation magnitude (float). Higher = stronger attack.
            label: True label tensor of shape (1,) or scalar.

        Returns:
            Adversarial image tensor of same shape, clamped to [0, 1].
        """
        # Ensure image is on the correct device and requires gradient
        image = image.clone().detach().to(self.device).requires_grad_(True)
        label = label.clone().detach().to(self.device)

        # Forward pass
        output = self.model(image)

        # Compute loss
        loss = self.loss_fn(output, label)

        # Zero any existing gradients on the model
        self.model.zero_grad()

        # Backward pass — compute gradient of loss w.r.t. the input image
        loss.backward()

        # Collect the gradient sign
        gradient_sign = image.grad.data.sign()

        # Create the adversarial image: x_adv = x + epsilon * sign(∇_x J)
        adversarial_image = image.data + epsilon * gradient_sign

        # Clamp to valid pixel range [0, 1]
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

        return adversarial_image

    def generate(
        self,
        image: torch.Tensor,
        epsilon: float,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int, bool]:
        """
        Public API: generates an adversarial example and returns structured results.

        Puts the model in eval mode, runs the FGSM attack, and compares
        clean vs. adversarial predictions.

        Args:
            image: Input image tensor of shape (1, C, H, W), values in [0, 1].
            epsilon: Perturbation magnitude (float, default 0.1 recommended).
            label: True label tensor of shape (1,) or scalar.

        Returns:
            Tuple of:
                - adversarial_image (Tensor): The crafted adversarial image.
                - clean_prediction (int): Model's prediction on the original image.
                - adversarial_prediction (int): Model's prediction on the adversarial image.
                - attack_success (bool): True if adversarial prediction != clean prediction.
        """
        # Set model to evaluation mode
        self.model.eval()

        # Get clean prediction (no gradients needed)
        with torch.no_grad():
            image_device = image.to(self.device)
            clean_output = self.model(image_device)
            clean_prediction = clean_output.argmax(dim=1).item()

        # Generate adversarial example
        adversarial_image = self.fgsm_attack(image, epsilon, label)

        # Get adversarial prediction
        with torch.no_grad():
            adversarial_output = self.model(adversarial_image)
            adversarial_prediction = adversarial_output.argmax(dim=1).item()

        # Determine if the attack was successful
        attack_success = clean_prediction != adversarial_prediction

        return adversarial_image, clean_prediction, adversarial_prediction, attack_success
