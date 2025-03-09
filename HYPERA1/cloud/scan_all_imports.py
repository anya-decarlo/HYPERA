#!/usr/bin/env python3
"""
Script to scan all Python files in the HYPERA project for imports
and create a comprehensive list of required packages.
"""
import os
import re
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
    imports = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all import statements
    import_patterns = [
        r'^\s*import\s+([a-zA-Z0-9_.,\s]+)',  # import package, package2
        r'^\s*from\s+([a-zA-Z0-9_.]+)\s+import',  # from package import
    ]
    
    for pattern in import_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            # Handle multiple imports on one line (import x, y, z)
            if ',' in match.group(1):
                for pkg in match.group(1).split(','):
                    pkg = pkg.strip()
                    if pkg:
                        # Get the base package name (first part before any dots)
                        base_pkg = pkg.split('.')[0]
                        if base_pkg not in ('__future__'):  # Skip built-ins
                            imports.append((base_pkg, file_path))
            else:
                # Get the base package name (first part before any dots)
                base_pkg = match.group(1).split('.')[0]
                if base_pkg not in ('__future__'):  # Skip built-ins
                    imports.append((base_pkg, file_path))
    
    return imports

def is_standard_library(module_name):
    """Check if a module is part of the Python standard library."""
    standard_libs = {
        'abc', 'argparse', 'array', 'ast', 'asyncio', 'base64', 'bisect', 'calendar',
        'collections', 'concurrent', 'contextlib', 'copy', 'csv', 'ctypes', 'datetime',
        'decimal', 'difflib', 'dis', 'email', 'enum', 'errno', 'fnmatch', 'functools',
        'gc', 'getopt', 'getpass', 'gettext', 'glob', 'gzip', 'hashlib', 'heapq',
        'hmac', 'html', 'http', 'importlib', 'inspect', 'io', 'itertools', 'json',
        'keyword', 'linecache', 'locale', 'logging', 'math', 'mimetypes', 'multiprocessing',
        'netrc', 'numbers', 'operator', 'os', 'pathlib', 'pickle', 'pkgutil', 'platform',
        'pprint', 'pwd', 'queue', 'random', 're', 'reprlib', 'select', 'shlex', 'shutil',
        'signal', 'socket', 'socketserver', 'sqlite3', 'ssl', 'stat', 'string', 'struct',
        'subprocess', 'sys', 'tarfile', 'tempfile', 'textwrap', 'threading', 'time',
        'timeit', 'token', 'tokenize', 'traceback', 'types', 'typing', 'unicodedata',
        'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'xmlrpc', 'zipfile',
        'zipimport', 'zlib'
    }
    return module_name in standard_libs

def map_to_pip_package(module_name):
    """Map module names to their pip package names."""
    mapping = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4',
        'wx': 'wxpython',
        'tk': 'tk',
        'IPython': 'ipython',
        'cairo': 'pycairo',
    }
    return mapping.get(module_name, module_name)

def scan_imports(project_dir):
    """Scan all Python files for imports and categorize them."""
    python_files = find_python_files(project_dir)
    
    all_imports = []
    for py_file in python_files:
        imports = extract_imports(py_file)
        all_imports.extend(imports)
    
    # Group imports by module
    grouped_imports = defaultdict(list)
    for module, file_path in all_imports:
        grouped_imports[module].append(file_path)
    
    # Categorize imports
    std_lib = []
    third_party = []
    local_modules = []
    
    for module, files in grouped_imports.items():
        if is_standard_library(module):
            std_lib.append((module, files))
        elif os.path.exists(os.path.join(project_dir, module)) or module.startswith('.'):
            local_modules.append((module, files))
        else:
            third_party.append((module, files))
    
    return {
        'standard_library': std_lib,
        'third_party': third_party,
        'local_modules': local_modules
    }

if __name__ == "__main__":
    # Use the parent directory of the script's location as the project root
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    imports = scan_imports(project_dir)
    
    print("=== THIRD-PARTY PACKAGES ===")
    for module, files in sorted(imports['third_party']):
        pip_package = map_to_pip_package(module)
        print(f"{pip_package} (imported as '{module}' in {len(files)} files)")
        # Print a sample of files using this import
        for file in files[:3]:  # Show up to 3 examples
            rel_path = os.path.relpath(file, project_dir)
            print(f"  - {rel_path}")
        if len(files) > 3:
            print(f"  - ... and {len(files) - 3} more files")
    
    print("\n=== LOCAL MODULES ===")
    for module, files in sorted(imports['local_modules']):
        print(f"{module} (used in {len(files)} files)")
    
    print("\n=== STANDARD LIBRARY MODULES ===")
    for module, files in sorted(imports['standard_library']):
        print(f"{module} (used in {len(files)} files)")
