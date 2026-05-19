# 🍃 Pollution Vision

A modern, AI-driven Streamlit application designed to estimate particulate matter density (air pollution) from images of leaves. 

## 1. The Simple Architecture (The Data Flow)
Think of the architecture as a funnel that takes a high-resolution image and boils it down to a single number representing pollution density.

* **Input Image**: A picture of a leaf (e.g., Ficus or Mango) is fed into the system.
* **OpenCV Pre-Processing (The Filter)**: The image is converted into an HSV (Hue, Saturation, Value) color space. A digital "mask" logic is applied to manipulate and isolate the green leaf properties and highlight pixels that match the color profile of urban dust (grays and browns).
* **Deep Learning Backbone (The Brain)**: The highlighted image passes through layers of convolutions. For this, we use **MobileNetV2** (a highly efficient Deep Convolutional Neural Network). The network looks for patterns: How thick is the texture? How much does the dust dull the natural shine (reflectance) of the leaf?
* **Regression Head (The Calculator)**: The neural network condenses these patterns into a final sequence of mathematical weights, outputting a single continuous number (e.g., 14.5 g/m²).

## 2. The Core Calculations
There is no single "magic formula" in deep learning, but the calculation relies on these mathematical pillars working together:

### A. The OpenCV Pixel Ratio (The Baseline Proxy)
Before the deep learning model makes its prediction, the OpenCV algorithm evaluates deterministic color ratios in the HSV space to estimate coverage. It maps the percentage of "dust-colored" pixels against the total size of the image, giving a coverage percentage that perfectly aligns the training labels to guide the neural network.

### B. The Deep Learning Regression (The Prediction)
Unlike the binary classifier which uses probabilities to pick a category, the dust calculator uses a **Linear Regression Algorithm** at the very end of its network. It takes the hundreds of features extracted by MobileNetV2 ($x_1, x_2, ... x_n$), multiplies them by learned weights ($w_1, w_2, ... w_n$), and adds a bias ($b$).

$$y = (w_1x_1 + w_2x_2 + ... + w_nx_n) + b$$

$y$ is the final predicted dust amount. This mirrors the real-world spectral prediction models scientists use, such as predicting dust based on the Simple Ratio index of leaf light reflection.

### C. Error Correction (How it Corrects Itself)
To ensure the regression calculation is accurate, the model evaluates its Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) during training. It compares its predicted dust level against the actual known dust level from the dataset.

By measuring these errors, the algorithm heavily penalizes predictions that are far off from the real dust levels. The Adam optimizer then adjusts the weights ($w$) in the regression formula to make the error as close to zero as possible.

## 3. The 4 Core Components (What We Actually Used)

To summarize the entire project pipeline into its absolute core components, here are the four most important models and algorithms that make this specific framework function:

### 1. HSV Color Thresholding Algorithm (Computer Vision)
* **What it is**: A deterministic image-processing algorithm from the OpenCV library.
* **Why it's critical**: Before the AI even "thinks," this algorithm mathematically processes the green color of the leaf to simulate and isolate the gray and brown particulates.
* **How it works**: It converts the image from standard RGB (Red, Green, Blue) to HSV (Hue, Saturation, Value). By adjusting the numerical limits on the Saturation and Value, it creates a profile that tells the deep learning model exactly what physical dust looks like on the leaf.

### 2. MobileNetV2 (Deep Convolutional Neural Network)
* **What it is**: The core "brain" or backbone model of your project. It is a highly efficient deep neural network pre-trained on millions of images, utilized here in place of heavier architectures like ResNet-50.
* **Why it's critical**: It replaces manual feature extraction. Instead of a human trying to measure the reflectance or texture of the leaf, MobileNetV2 automatically learns what physical dust deposition looks like at a microscopic pixel level.
* **How it works**: It uses "Inverted Residual Blocks" (skip connections) to pass visual information through dozens of mathematical filters without losing the original image data. It shrinks the high-resolution leaf image down into a dense list of mathematical features.

### 3. Two-Stage Inference Pipeline (Optimization & Strategy)
* **What it is**: Our custom system design acting as the "steering wheel" during inference.
* **Why it's critical**: Rather than a single Multi-Task Weighted Loss model, we utilized a highly robust **Two-Stage Pipeline**. This ensures the system explicitly prioritizes classifying the presence of disease/dust *before* it calculates the density, preventing skewed predictions on clean leaves.
* **How it works**: Stage 1 uses Binary Cross-Entropy to mathematically classify the image as 'Clean' or 'Dusty'. Only if the leaf is dusty, Stage 2 activates the Adam optimizer-trained Regression model to strictly focus processing power on calculating the exact dust density.

### 4. Linear Regression Prediction Model (Scientific Output)
* **What it is**: The mathematical model sitting at the very end of our network's "Dust Head."
* **Why it's critical**: It translates the abstract patterns found by MobileNetV2 into a real-world, continuous numerical value (like density of dust). This perfectly mirrors real-world botanical monitoring studies, which establish forecast models of dust deposition using linear regression equations based on spectral reflectance indices.
* **How it works**: It takes the final features extracted by the network and applies a simple mathematical formula: $y = (wx) + b$. The output ($y$) represents the predicted physical dust deposition, replicating the logic used by environmental scientists.

---

## 🚀 How to Run

1. **Prepare Python Environment**:
   ```powershell
   Set-Location -Path 'F:\Mini-Project\pollution_vision'
   .\.venv312\Scripts\activate
   python -m pip install -r requirements.txt
   ```

2. **Train the Models**:
   ```powershell
   python scripts/01_download_data.py
   python scripts/02_augment_data.py
   python scripts/03_train_classifier.py
   python scripts/04_train_regressor.py
   ```

3. **Launch the Application**:
   ```powershell
   streamlit run app.py
   ```
