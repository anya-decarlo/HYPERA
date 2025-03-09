#!/usr/bin/env python3
"""
Script to generate a comprehensive requirements.txt file for the HYPERA project.
This script scans Python files for import statements and creates a list of required packages.
"""
import os
import re
import subprocess
import sys
from collections import defaultdict

def find_python_files(start_dir):
    """Find all Python files in the directory tree."""
    python_files = []
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path):
    """Extract import statements from a Python file."""
    imports = set()
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all import statements
    import_patterns = [
        r'^\s*import\s+([a-zA-Z0-9_.]+)',  # import package
        r'^\s*from\s+([a-zA-Z0-9_.]+)\s+import',  # from package import
    ]
    
    for pattern in import_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            # Get the base package name (first part before any dots)
            package = match.group(1).split('.')[0]
            if package not in ('__future__', 'legacy'):  # Skip built-ins and local modules
                imports.add(package)
    
    return imports

def get_installed_packages():
    """Get a list of installed packages and their versions."""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                               capture_output=True, text=True, check=True)
        packages = {}
        for line in result.stdout.splitlines():
            if '==' in line:
                name, version = line.split('==', 1)
                packages[name.lower()] = version
        return packages
    except subprocess.SubprocessError:
        return {}

def generate_requirements(project_dir):
    """Generate a requirements.txt file for the project."""
    python_files = find_python_files(project_dir)
    
    all_imports = set()
    for py_file in python_files:
        imports = extract_imports(py_file)
        all_imports.update(imports)
    
    # Add known dependencies that might not be directly imported
    additional_deps = {
        'monai': 'MONAI for medical image processing',
        'torch': 'PyTorch for deep learning',
        'torchvision': 'Computer vision utilities for PyTorch',
        'numpy': 'Numerical computing',
        'pandas': 'Data analysis and manipulation',
        'matplotlib': 'Plotting and visualization',
        'scikit-learn': 'Machine learning utilities',
        'scikit-image': 'Image processing',
        'scipy': 'Scientific computing',
        'tqdm': 'Progress bars',
        'pillow': 'Image processing (PIL)',
        'tensorboard': 'Visualization for TensorFlow/PyTorch',
        'ruptures': 'Change point detection',
        'nibabel': 'Medical image I/O',
        'h5py': 'HDF5 file format support',
        'SimpleITK': 'Medical image processing',
        'opencv-python': 'Computer vision',
        'networkx': 'Network analysis',
    }
    
    # Get installed packages and versions
    installed_packages = get_installed_packages()
    
    # Prepare requirements with versions when available
    requirements = []
    for package in sorted(all_imports):
        package_lower = package.lower()
        
        # Map some package names to their pip install names
        pip_name = {
            'sklearn': 'scikit-learn',
            'PIL': 'pillow',
            'cv2': 'opencv-python',
            'yaml': 'pyyaml',
        }.get(package, package)
        
        # Check if we have version info
        if pip_name.lower() in installed_packages:
            requirements.append(f"{pip_name}=={installed_packages[pip_name.lower()]}")
        else:
            requirements.append(pip_name)
    
    # Add additional dependencies that might not be directly imported
    for dep, comment in additional_deps.items():
        if dep not in [r.split('==')[0].lower() for r in requirements]:
            if dep.lower() in installed_packages:
                requirements.append(f"{dep}=={installed_packages[dep.lower()]} # {comment}")
            else:
                requirements.append(f"{dep} # {comment}")
    
    return requirements

if __name__ == "__main__":
    # Use the parent directory of the script's location as the project root
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    requirements = generate_requirements(project_dir)
    
    # Write to requirements.txt
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write("# HYPERA Project Requirements\n")
        f.write("# Generated automatically\n\n")
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"Generated requirements.txt with {len(requirements)} packages")
    print(f"File saved to: {requirements_path}")
