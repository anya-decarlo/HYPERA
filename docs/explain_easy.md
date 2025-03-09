# HYPERA Explained Simply

## What is HYPERA?

HYPERA is an AI system that makes medical image segmentation better and easier. It uses smart "agents" (like little AI assistants) to automatically figure out the best settings for training neural networks and to improve the quality of image segmentation.

## How HYPERA Works: The Simple Version

HYPERA has two main parts that work together:

### Part 1: Training Optimization

Imagine you're baking a cake, but instead of following a recipe with exact measurements, you have AI helpers that figure out the perfect amount of each ingredient as you go:

1. **Smart Helpers**: HYPERA has different AI helpers (we call them "agents") that each focus on one aspect of training:
   - One helper adjusts how quickly the AI learns
   - Another helper decides which parts of the training data are most important
   - Other helpers adjust different settings to make training better

2. **How They Work**:
   - They watch what's happening during training
   - They make small adjustments to improve results
   - They learn from what works and what doesn't
   - They communicate with each other to coordinate their actions

3. **The Result**: Instead of a human having to guess the best settings (which could take weeks or months of trial and error), HYPERA's helpers figure it out automatically during training.

### Part 2: Segmentation Improvement

After training a neural network to segment medical images (like finding tumors in CT scans or cells in microscope images), HYPERA has another set of helpers that make the segmentation results even better:

1. **Specialized Helpers**:
   - One helper focuses on getting the overall shape right
   - Another helper makes sure the boundaries between different regions are accurate
   - Other helpers focus on different aspects of image quality

2. **How They Work**:
   - They look at the initial segmentation from the neural network
   - They make small adjustments to improve different aspects of the result
   - They learn which adjustments work best for different types of images
   - They work together to balance different quality goals

3. **The Result**: Better segmentation that's more accurate and reliable than what you'd get from a standard neural network alone.

## The HYPERA Workflow

Here's how the whole system works from start to finish:

1. **Setup**: Load your medical images and choose a basic neural network architecture (like U-Net)

2. **Training Phase**: 
   - The training helpers automatically find the best learning settings
   - The neural network learns to segment images with these optimized settings
   - The system saves checkpoints of the best models along the way

3. **Segmentation Phase**:
   - The trained neural network makes initial segmentations
   - The segmentation helpers refine these results to make them more accurate
   - The system learns which refinement strategies work best

4. **Deployment**:
   - The final system can be used on new images
   - It continues to adapt to new data
   - It produces more accurate segmentations than traditional methods

## Why HYPERA is Special

1. **It's Automatic**: No need for experts to spend weeks tuning parameters by hand

2. **It's Adaptive**: It adjusts to different types of medical images and segmentation tasks

3. **It's Comprehensive**: It optimizes both the training process AND the segmentation quality

4. **It's Practical**: It works with existing neural network frameworks and can be integrated into clinical workflows

Think of HYPERA as having a team of AI experts working 24/7 to make your medical image segmentation as good as possible, automatically adjusting thousands of settings that would be impossible for a human to optimize manually.



## Frequently Asked Questions

### 1. Do all images need ground truth labels?
During training, yes - we need labeled images to teach the system. But once HYPERA is fully trained, it can work on new images without ground truth labels.

### 2. What is the loss function in HYPERA?
HYPERA uses a combination of different loss functions that work together. The main ones include Dice loss (which measures overlap between predicted and actual regions), boundary loss (which focuses on getting the edges right), and focal loss (which helps with imbalanced data). The system automatically adjusts how much weight to give each type of loss during training.

### 3. How do the agents communicate with each other?
Agents communicate through a shared state system, kind of like a shared whiteboard. Each agent can read information that other agents have written (like current settings and results) and can write its own decisions. A coordinator component makes sure agents don't make conflicting decisions and helps them work together effectively.

### 4. How will HYPERA use Google Cloud?

We're planning to use Google Cloud to make HYPERA even more powerful:

1. **Super-Powered Computers**: We'll use Google's special computers with powerful graphics cards (GPUs) that can train AI much faster than regular computers.

2. **Big Data Storage**: We'll store all our medical images in Google's cloud storage, which is like a super-secure online hard drive that we can access from anywhere.

3. **Working with Full-Size Images**: We're updating HYPERA to work with the largest, highest-quality medical images without having to shrink them down, which will improve accuracy.

4. **Team Training**: We'll set up HYPERA to train across multiple computers at once, like having a team of computers working together to solve a problem faster.

5. **Ready-to-Go Packages**: We'll create special containers (think of them like pre-packed suitcases with everything HYPERA needs) so the system can be easily set up on any Google Cloud computer.

This cloud approach means HYPERA can process more images, train faster, and work with even larger datasets than before.

### 5. What is the ultimate goal of HYPERA with segmentation masks?

One of our main goals is to create what we call "Region Adjacency Graphs" (RAGs) from the segmentation masks that HYPERA produces:

1. **What is a Region Adjacency Graph?** Think of it like a map showing which cells or regions are neighbors. Each cell becomes a dot on the map, and lines connect dots that touch each other.

2. **How we create it:**
   - First, HYPERA's U-Net model (running in MONAI) creates a detailed segmentation mask where each cell has its own unique ID
   - Then, we analyze which cells are touching each other
   - Finally, we create a graph structure where each cell is a node and connections show which cells are neighbors

3. **Why this is valuable:**
   - It helps us understand the relationships between different regions
   - We can analyze patterns of how cells connect to each other
   - It makes it easier to track changes over time (like cell division or movement)
   - We can extract important measurements about cell neighborhoods

4. **What we can do with it:**
   - Analyze tissue organization
   - Detect abnormal cell arrangements
   - Study how cells interact with their neighbors
   - Track changes in cell connections over time

This approach combines the power of deep learning (for accurate segmentation) with graph theory (for understanding relationships), giving us much richer information than just the segmentation alone.

### 6. What is the reward function in HYPERA?

The reward function is like a scoring system that tells our AI agents how well they're doing. In HYPERA, we use a special multi-objective reward system:

1. **What it measures:**
   - How well the predicted regions match the actual regions (Dice Score)
   - How accurate the boundaries are between different regions
   - Whether we're detecting the right number of objects (not too many, not too few)
   - If the shapes look realistic and biologically plausible
   - Whether we have a good balance between finding too much or too little

2. **How it works:**
   - Each aspect of performance gets its own score
   - These scores are combined with different weights depending on what's most important
   - The weights automatically adjust during training based on what needs improvement
   - The agents use these rewards to learn which actions lead to better results

Think of it like training a dog - when the dog does something good, you give it a treat. Our reward function gives the AI agents "treats" when they make good decisions that improve segmentation quality.

### 7. What is SAC and how does HYPERA use it?

SAC stands for "Soft Actor-Critic," which is a type of reinforcement learning algorithm that HYPERA uses to train its agents:

1. **What it is in simple terms:**
   - It's like a learning system that balances trying new things (exploration) with using what already works (exploitation)
   - "Soft" means it doesn't just pick the single best action but considers multiple good options
   - "Actor" is the part that decides what actions to take
   - "Critic" is the part that evaluates how good those actions are

2. **How HYPERA uses it:**
   - Each agent in HYPERA (both for training optimization and segmentation) uses SAC to learn
   - The agent tries different actions and gets rewards based on the results
   - Over time, it learns which actions lead to better segmentations
   - It stores its experiences in a "memory bank" and learns from past successes and failures

3. **Why it's better than simpler approaches:**
   - It's more stable during learning
   - It can handle complex decision spaces with many possible actions
   - It balances exploration and exploitation automatically
   - It works well even when the best action isn't immediately obvious

4. **What results SAC is based on:**
   - For hyperparameter agents: Improvements in validation metrics like accuracy and loss
   - For segmentation agents: Improvements in segmentation quality metrics (Dice score, boundary accuracy, etc.)
   - For object detection: Better object count, size distribution, and instance separation
   - The system tracks both immediate improvements and long-term progress

SAC helps HYPERA's agents become smarter over time through experience, similar to how humans learn from trial and error, but much faster.



# Fix the paths to use the correct nested directories without spaces
image_data_dir = os.path.join(bbbc039_path, "BBBC039", "images", "images")  # Fixed path
label_data_dir = os.path.join(bbbc039_path, "BBBC039", "masks", "masks")    # Fixed path

print(f"Image data directory: {image_data_dir}")
print(f"Label data directory: {label_data_dir}")


# Install required dependencies with stable versions
# Run this cell first to ensure all necessary packages are installed

# First, uninstall the conflicting packages
!pip uninstall -y scikit-image stable-baselines3

# Then install compatible versions in the correct order
!pip install pillow==9.5.0
!pip install torch==2.1.2
!pip install scikit-image==0.19.3  # This version is compatible with pillow 9.5.0
!pip install tifffile==2023.7.10
!pip install imagecodecs

# Essential packages
!pip install ruptures==1.1.9  # Change point detection
!pip install monai==1.3.1  # Medical image processing (stable release)
!pip install torch==2.1.2 torchvision==0.16.2  # Deep learning (stable releases)
!pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.0  # Data analysis and visualization
!pip install scikit-learn scikit-image==0.25.2 scipy==1.13.1  # Scientific computing
!pip install tqdm tensorboard==2.19.0  # Progress and visualization
!pip install nibabel==5.3.2  # Medical image I/O
!pip install statsmodels==0.14.4  # Time series analysis and statistical models
!pip install torchmetrics==1.2.1  # Additional PyTorch metrics

# Image processing libraries for TIFF support
!pip install tifffile==2023.7.10
!pip install imagecodecs
!pip install SimpleITK==2.4.1
!pip install pillow==10.2.0
!pip install opencv-python networkx

# Optional packages
!pip install h5py==3.13.0

# Verify TIFF support
import sys
print(f"Python version: {sys.version}")
try:
    import tifffile
    print(f"tifffile version: {tifffile.__version__}")
except ImportError:
    print("tifffile not installed")


# Check installed versions
import sys
print(f"Python version: {sys.version}")
try:
    import PIL
    print(f"Pillow version: {PIL.__version__}")
except ImportError:
    print("Pillow not installed")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not installed")

try:
    import skimage
    print(f"scikit-image version: {skimage.__version__}")
except ImportError:
    print("scikit-image not installed")

try:
    import tifffile
    print(f"tifffile version: {tifffile.__version__}")
except ImportError:
    print("tifffile not installed")
