# ⚡ Deep Learning Optimizer Visualizer: From SGD to AdamW

An interactive, from-scratch optimization playground and Multi-Layer Perceptron (MLP) benchmarking dashboard built with pure **NumPy**, **Plotly**, and **Streamlit**.

This project implements and visualizes 7 core optimization algorithms from first mathematical principles without relying on high-level autodiff or optimizer libraries (such as PyTorch or TensorFlow).

---

## 🚀 Live Demo & Repository
* **GitHub Repository:** https://github.com/SHASHI434/optimizer-visualizer-from-scratch
* **Live App:** https://optimizer-from-scratch.streamlit.app/
---

## 📌 Key Features

* **7 Stateful Optimizers Built from Scratch:**
  1. **Plain SGD:** Basic first-order gradient descent.
  2. **SGD with Momentum:** Velocity accumulation to dampen oscillations along high-curvature ravines.
  3. **NAG (Nesterov Accelerated Gradient):** Look-ahead gradient evaluation that acts as an adaptive brake against overshooting.
  4. **AdaGrad:** Coordinate-wise adaptive scaling using historical squared gradient accumulation.
  5. **RMSProp:** Exponential moving average of squared gradients to resolve AdaGrad's premature learning rate decay.
  6. **Adam:** Combined first-moment direction and second-moment scaling with early-step bias correction.
  7. **AdamW:** Decoupled weight decay regularization to ensure uniform weight shrinkage across adaptive updates.

* **Part A: 2D Loss Surface Playground:**
  * Evaluates curvature conditioning across anisotropic quadratic bowls: L1 to L4.
  * Synchronized contour trajectory mapping and logarithmic loss vs. iteration curves.
  * Step-by-step live path animation with speed controls and divergence safeguards.

* **Part B: Neural Network Training Dashboard:**
  * From-scratch 3-layer MLP architecture: Input(30) -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid.
  * Analytical manual backpropagation engine with Binary Cross-Entropy (BCE) loss.
  * Real-world binary classification benchmark on the Breast Cancer Wisconsin Dataset.
  * Support for both Full-Batch and Mini-Batch gradient descent modes.
  * Live monitoring of Training Loss, Test Loss, Test Accuracy, and effective learning rate scaling per epoch.

* **Automated Comparison Table:**
  * Computes final training/test loss, accuracy metrics, and automatic convergence epoch detection.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Core Mathematics & Neural Network:** NumPy (Matrix algebra, vector calculus, custom backpropagation)
* **Data Preprocessing:** Scikit-Learn (Dataset loading & standard feature scaling only)
* **Interactive Visualization:** Plotly Subplots & Graph Objects
* **Frontend Web Framework:** Streamlit
* **Tabular Summaries:** Pandas

---

## 💻 Installation & Local Setup

1. **Clone the repository:**
   git clone https://github.com/SHASHI434/optimizer-visualizer-from-scratch.git
   cd optimizer-visualizer-from-scratch

2. **Create and activate a virtual environment (optional):**
   python -m venv env
   # Windows:
   env\Scripts\activate
   # macOS/Linux:
   source env/bin/activate

3. **Install dependencies:**
   pip install -r requirements.txt

4. **Run the Streamlit application:**
   streamlit run app.py

---

## 📁 Repository Structure

```text
optimizer-visualizer-from-scratch/
├── app.py                # Complete Streamlit application (Optimizers, NN, UI)
├── requirements.txt      # Project dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation

👤 Author
Shashi — GitHub: @SHASHI434