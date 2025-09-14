import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from pathlib import Path
import cv2

from huetuber import ReinhardNormalizer
from torchstain.numpy.normalizers.reinhard import NumpyReinhardNormalizer


def load_image(path):
    """Load image and convert to RGB format."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize_reinhard_normalization():
    """Visualize Reinhard stain normalization results."""
    
    # Data directory
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load target image
    target_path = data_dir / "target.png"
    target_img = load_image(target_path)
    
    # Load test images
    test_paths = [data_dir / f"test_{i}.png" for i in range(1, 6)]
    test_images = [load_image(path) for path in test_paths if path.exists()]
    
    print(f"Loaded target image: {target_img.shape}")
    print(f"Loaded {len(test_images)} test images")
    
    # Convert to CuPy arrays with channel_axis=1 (NCHW format)
    target_cp = cp.array(target_img).transpose(2, 0, 1)[None, ...]  # Add batch dimension
    test_images_cp = [cp.array(img).transpose(2, 0, 1)[None, ...] for img in test_images]
    
    # Initialize and fit Reinhard normalizer (huetuber)
    normalizer = ReinhardNormalizer(channel_axis=1)
    normalizer.fit(target_cp)
    
    # Initialize and fit torchstain numpy normalizer
    torchstain_normalizer = NumpyReinhardNormalizer()
    torchstain_normalizer.fit(target_img)  # torchstain expects HWC format
    
    # Normalize test images with huetuber
    normalized_images = []
    for test_img in test_images_cp:
        normalized = normalizer.normalize(test_img)
        normalized_images.append(normalized)
    
    # Normalize test images with torchstain
    torchstain_normalized = []
    for test_img in test_images:
        normalized = torchstain_normalizer.normalize(test_img)
        torchstain_normalized.append(normalized)
    
    # Convert back to numpy for visualization (NHWC format)
    target_np = target_cp[0].transpose(1, 2, 0).get()
    test_images_np = [img[0].transpose(1, 2, 0).get() for img in test_images_cp]
    normalized_np = [img[0].transpose(1, 2, 0).get() for img in normalized_images]
    torchstain_normalized_np = torchstain_normalized  # Already in correct format
    
    # Create visualization with 4 rows
    num_tests = len(test_images)
    fig, axes = plt.subplots(4, num_tests, figsize=(3 * num_tests, 12))
    
    # Ensure axes is 2D
    if num_tests == 1:
        axes = axes.reshape(4, 1)
    
    # First row: Target image (show in first column, hide others)
    axes[0, 0].imshow(target_np)
    axes[0, 0].set_title("Target Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Hide other columns in first row
    for j in range(1, num_tests):
        axes[0, j].axis('off')
    
    # Second row: Original test images
    for i, original in enumerate(test_images_np):
        axes[1, i].imshow(original)
        axes[1, i].set_title(f"Test {i+1} - Original", fontsize=12)
        axes[1, i].axis('off')
    
    # Third row: Normalized test images (huetuber)
    for i, normalized in enumerate(normalized_np):
        axes[2, i].imshow(normalized)
        axes[2, i].set_title(f"Test {i+1} - HueTuber", fontsize=12)
        axes[2, i].axis('off')
    
    # Fourth row: Normalized test images (torchstain)
    for i, normalized in enumerate(torchstain_normalized_np):
        axes[3, i].imshow(normalized)
        axes[3, i].set_title(f"Test {i+1} - TorchStain", fontsize=12)
        axes[3, i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.suptitle("Reinhard Stain Normalization Comparison: HueTuber vs TorchStain", fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    save_path = Path(__file__).parent / "results"
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / "reinhard_visualization.png", dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved to: {save_path / 'reinhard_visualization.png'}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    visualize_reinhard_normalization()