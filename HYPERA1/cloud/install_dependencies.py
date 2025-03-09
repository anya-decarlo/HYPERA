#!/usr/bin/env python3
"""
Script to install missing dependencies for the HYPERA project.
"""
import subprocess
import sys

def install_package(package):
    """Install a Python package using pip."""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages."""
    # Essential packages that must be installed
    essential_packages = [
        "ruptures==1.1.9",  # Change point detection
        "monai",            # Medical image processing
        "torch",            # Deep learning
        "torchvision",      # Computer vision for PyTorch
        "numpy",            # Numerical computing
        "pandas",           # Data analysis
        "matplotlib",       # Plotting
        "scikit-learn",     # Machine learning
        "scikit-image",     # Image processing
        "scipy",            # Scientific computing
        "tqdm",             # Progress bars
        "tensorboard",      # Visualization
        "nibabel",          # Medical image I/O
    ]
    
    # Optional packages that are nice to have
    optional_packages = [
        "h5py",             # HDF5 file format
        "SimpleITK",        # Medical image processing
        "opencv-python",    # Computer vision
        "networkx",         # Network analysis
        "pillow",           # Image processing
    ]
    
    # Install essential packages first
    print("Installing essential packages...")
    for package in essential_packages:
        install_package(package)
    
    # Install optional packages
    print("\nInstalling optional packages...")
    for package in optional_packages:
        install_package(package)
    
    print("\nAll dependencies installed!")

if __name__ == "__main__":
    main()
