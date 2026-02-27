"""
evaluate.py — Evaluate FGSM Attack Robustness on MNIST

Loads the trained MNIST model, applies FGSM at various epsilon values,
records accuracy drops, and saves results (table + plot).

Usage:
    python evaluate.py
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from model import MNISTNet
from fgsm import Attack


def evaluate_attack(model, device, test_loader, epsilon, attack):
    """
    Run FGSM attack on the entire test set at a given epsilon and
    compute accuracy metrics.

    Args:
        model: The target model.
        device: Computation device.
        test_loader: DataLoader for test data.
        epsilon: FGSM perturbation strength.
        attack: Attack instance.

    Returns:
        Tuple of (clean_correct, adversarial_correct, total, examples)
        where examples is a list of (original, adversarial, clean_pred, adv_pred)
        for visualization.
    """
    clean_correct = 0
    adversarial_correct = 0
    total = 0
    examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        for i in range(data.size(0)):
            image = data[i].unsqueeze(0)  # (1, 1, 28, 28)
            label = target[i].unsqueeze(0)  # (1,)

            adv_image, clean_pred, adv_pred, success = attack.generate(
                image, epsilon, label
            )

            total += 1

            if clean_pred == label.item():
                clean_correct += 1

            if adv_pred == label.item():
                adversarial_correct += 1

            # Save a few examples where the attack succeeded
            if success and len(examples) < 5:
                examples.append((
                    image.squeeze().cpu().detach().numpy(),
                    adv_image.squeeze().cpu().detach().numpy(),
                    clean_pred,
                    adv_pred,
                    label.item(),
                ))

    return clean_correct, adversarial_correct, total, examples


def save_examples_plot(examples, epsilon, output_dir):
    """Save a side-by-side plot of original vs adversarial examples."""
    if not examples:
        return

    num = len(examples)
    fig, axes = plt.subplots(2, num, figsize=(num * 3, 6))

    if num == 1:
        axes = axes.reshape(2, 1)

    for i, (orig, adv, clean_pred, adv_pred, true_label) in enumerate(examples):
        # Original image
        axes[0, i].imshow(orig, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Original\nPred: {clean_pred}\nTrue: {true_label}", fontsize=9)
        axes[0, i].axis("off")

        # Adversarial image
        axes[1, i].imshow(adv, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Adversarial\nPred: {adv_pred}", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(f"FGSM Attack Examples (ε = {epsilon})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"fgsm_examples_eps_{epsilon}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved example plot: {filepath}")


def save_accuracy_plot(epsilons, clean_accs, adv_accs, output_dir):
    """Save a line plot of accuracy vs epsilon."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epsilons, clean_accs, "b-o", label="Clean Accuracy", linewidth=2)
    ax.plot(epsilons, adv_accs, "r-s", label="Adversarial Accuracy", linewidth=2)

    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("FGSM Attack: Accuracy vs Perturbation Strength", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    filepath = os.path.join(output_dir, "fgsm_accuracy_vs_epsilon.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved accuracy plot: {filepath}")


def main():
    # Configuration
    MODEL_PATH = "mnist_cnn.pth"
    OUTPUT_DIR = "results"
    EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # Use a subset for faster evaluation (set to None for full test set)
    MAX_SAMPLES = 1000

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # Load test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Optionally limit dataset size for faster evaluation
    if MAX_SAMPLES and MAX_SAMPLES < len(test_dataset):
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(MAX_SAMPLES)
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # Initialize attack
    loss_fn = nn.CrossEntropyLoss()
    attack = Attack(model=model, loss_fn=loss_fn, device=device)

    # Evaluate at each epsilon
    print("\n" + "=" * 65)
    print(f"{'Epsilon':>10} | {'Clean Acc':>12} | {'Adv Acc':>12} | {'Acc Drop':>10}")
    print("=" * 65)

    clean_accs = []
    adv_accs = []
    results_lines = []

    results_lines.append("FGSM Attack Evaluation Results")
    results_lines.append("=" * 50)
    results_lines.append(f"Model: MNISTNet")
    results_lines.append(f"Test Samples: {len(test_dataset)}")
    results_lines.append(f"Device: {device}")
    results_lines.append("")
    results_lines.append(f"{'Epsilon':>10} | {'Clean Acc':>12} | {'Adv Acc':>12} | {'Acc Drop':>10}")
    results_lines.append("-" * 50)

    for eps in EPSILONS:
        print(f"\nEvaluating epsilon = {eps}...")
        clean_correct, adv_correct, total, examples = evaluate_attack(
            model, device, test_loader, eps, attack
        )

        clean_acc = 100.0 * clean_correct / total
        adv_acc = 100.0 * adv_correct / total
        acc_drop = clean_acc - adv_acc

        clean_accs.append(clean_acc)
        adv_accs.append(adv_acc)

        print(f"{'ε=' + str(eps):>10} | {clean_acc:>11.2f}% | {adv_acc:>11.2f}% | {acc_drop:>9.2f}%")
        results_lines.append(
            f"{'ε=' + str(eps):>10} | {clean_acc:>11.2f}% | {adv_acc:>11.2f}% | {acc_drop:>9.2f}%"
        )

        # Save example visualizations for non-zero epsilons
        if eps > 0:
            save_examples_plot(examples, eps, OUTPUT_DIR)

    print("=" * 65)

    # Save accuracy plot
    save_accuracy_plot(EPSILONS, clean_accs, adv_accs, OUTPUT_DIR)

    # Save results to text file
    results_path = os.path.join(OUTPUT_DIR, "fgsm_evaluation_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))
    print(f"Saved results to: {results_path}")

    print("\nDone! Check the 'results/' directory for outputs.")


if __name__ == "__main__":
    main()
