
import os
import sys
import subprocess
import tarfile
import tempfile

def main():
    # Extract the code
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "code.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(CODE)
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=tmp_dir)
        
        # Run the training script
        subprocess.run([sys.executable, os.path.join(tmp_dir, "HYPERA1", "legacy", "train_bbbc039_with_agents.py")] + sys.argv[1:], check=True)

if __name__ == "__main__":
    main()
