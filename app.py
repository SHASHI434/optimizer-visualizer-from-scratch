import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Optimizer Visualizer: SGD to AdamW", layout="wide")

# =========================================================
# MODULE 1: FROM-SCRATCH OPTIMIZERS (Stateful Classes)
# =========================================================

class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = max(1e-6, lr)  # Input validation
    def step(self, params, grads):
        raise NotImplementedError

# 1. Plain SGD
class SGD(Optimizer):
    def step(self, params, grads):
        return {k: params[k] - self.lr * grads[k] for k in params}

# 2. SGD with Momentum
class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = {}

    def step(self, params, grads):
        if not self.v:
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        new_params = {}
        for k in params:
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * grads[k]
            new_params[k] = params[k] - self.lr * self.v[k]
        return new_params

# 3. NAG (Nesterov Accelerated Gradient)
class NAG(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = {}

    def get_lookahead(self, params):
        if not self.v:
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        return {k: params[k] - self.beta * self.v[k] for k in params}

    def step(self, params, lookahead_grads):
        if not self.v:
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        new_params = {}
        for k in params:
            self.v[k] = self.beta * self.v[k] + self.lr * lookahead_grads[k]
            new_params[k] = params[k] - self.v[k]
        return new_params

# 4. AdaGrad
class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.G = {}

    def step(self, params, grads):
        if not self.G:
            self.G = {k: np.zeros_like(v) for k, v in params.items()}
        new_params = {}
        for k in params:
            self.G[k] += grads[k] ** 2
            new_params[k] = params[k] - (self.lr * grads[k]) / (np.sqrt(self.G[k]) + self.eps)
        return new_params

# 5. RMSProp
class RMSProp(Optimizer):
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.v = {}

    def step(self, params, grads):
        if not self.v:
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        new_params = {}
        for k in params:
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * (grads[k] ** 2)
            new_params[k] = params[k] - (self.lr * grads[k]) / (np.sqrt(self.v[k]) + self.eps)
        return new_params

# 6. Adam
class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        if not self.m:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t += 1
        new_params = {}
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            new_params[k] = params[k] - (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)
        return new_params

# 7. AdamW
class AdamW(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-3):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        if not self.m:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t += 1
        new_params = {}
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            step_update = (m_hat / (np.sqrt(v_hat) + self.eps)) + (self.weight_decay * params[k])
            new_params[k] = params[k] - self.lr * step_update
        return new_params

COLORS = {
    "SGD": "#636EFA", "SGD + Momentum": "#EF553B", "NAG": "#00CC96",
    "AdaGrad": "#AB63FA", "RMSProp": "#FFA15A", "Adam": "#19D3F3", "AdamW": "#FF6692"
}

def get_optimizer_instance(name, lr, beta, beta1, beta2, wd):
    if name == "SGD": return SGD(lr=lr)
    if name == "SGD + Momentum": return SGDMomentum(lr=lr, beta=beta)
    if name == "NAG": return NAG(lr=lr, beta=beta)
    if name == "AdaGrad": return AdaGrad(lr=lr)
    if name == "RMSProp": return RMSProp(lr=lr, beta=beta)
    if name == "Adam": return Adam(lr=lr, beta1=beta1, beta2=beta2)
    if name == "AdamW": return AdamW(lr=lr, beta1=beta1, beta2=beta2, weight_decay=wd)

# =========================================================
# UI SETUP & SIDEBAR
# =========================================================

st.title("⚡ Optimizer Visualizer: SGD to AdamW")
tab1, tab2, tab3 = st.tabs(["Part A: 2D Loss Surface", "Part B: Neural Network Training", "Theory & Explanations"])

with st.sidebar:
    st.header("⚙️ Global Hyperparameters")
    lr = st.number_input("Learning Rate (η)", min_value=0.0001, max_value=1.0, value=0.01, step=0.005, format="%.4f")
    beta = st.slider("Momentum / RMSProp β", 0.0, 0.99, 0.9, 0.01)
    beta1 = st.slider("Adam/AdamW β₁", 0.0, 0.99, 0.9, 0.01)
    beta2 = st.slider("Adam/AdamW β₂", 0.0, 0.999, 0.999, 0.001)
    weight_decay = st.number_input("AdamW λ (Weight Decay)", min_value=0.0, max_value=0.1, value=0.001, format="%.4f")
    
    st.markdown("---")
    st.markdown("**Defaults:** $\eta=0.01$, $\\beta=0.9$, $\\beta_1=0.9$, $\\beta_2=0.999$, $\\lambda=0.001$")

# =========================================================
# MODULE 2: PART A - 2D LOSS SURFACE PLAYGROUND
# =========================================================

with tab1:
    col_ctrl, col_view = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Surface Controls")
        loss_choice = st.selectbox(
            "Loss Surface",
            ["L1: x² + 10y²", "L2: x² + 50y² (Default)", "L3: x² + 100y²", "L4: x² + 1000y²"],
            index=1
        )
        k_val = {"L1: x² + 10y²": 10, "L2: x² + 50y² (Default)": 50, "L3: x² + 100y²": 100, "L4: x² + 1000y²": 1000}[loss_choice]
        
        selected_optims = st.multiselect(
            "Optimizers to Compare",
            list(COLORS.keys()),
            default=["SGD", "SGD + Momentum", "Adam"]
        )
        
        init_x = st.number_input("Initial x₀", value=8.0)
        init_y = st.number_input("Initial y₀", value=8.0)
        iterations = st.slider("Iterations", min_value=10, max_value=500, value=150, step=10)
        
        animate = st.checkbox("▶ Live Animation Mode", value=False)
        anim_speed = st.slider("Anim Speed (fps delay)", 0.01, 0.2, 0.03) if animate else 0

    # Compute Trajectories
    trajectories = {}
    loss_histories = {}

    for opt_name in selected_optims:
        opt = get_optimizer_instance(opt_name, lr, beta, beta1, beta2, weight_decay)
        params = {'x': float(init_x), 'y': float(init_y)}
        traj = [(params['x'], params['y'])]
        losses = [params['x']**2 + k_val * (params['y']**2)]

        for _ in range(iterations):
            if opt_name == "NAG":
                la = opt.get_lookahead(params)
                grads = {'x': 2.0 * la['x'], 'y': 2.0 * k_val * la['y']}
            else:
                grads = {'x': 2.0 * params['x'], 'y': 2.0 * k_val * params['y']}
            
            params = opt.step(params, grads)
            
            if abs(params['x']) > 1e4 or abs(params['y']) > 1e4 or np.isnan(params['x']):
                break
                
            traj.append((params['x'], params['y']))
            losses.append(params['x']**2 + k_val * (params['y']**2))

        trajectories[opt_name] = traj
        loss_histories[opt_name] = losses

    with col_view:
        plot_placeholder = st.empty()
        
        def render_plot(step_limit):
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Contour Map & Trajectory", "Loss vs. Iteration"))
            x_max = max(10.0, abs(init_x) + 2)
            y_max = max(10.0, abs(init_y) + 2)
            x_grid = np.linspace(-x_max, x_max, 100)
            y_grid = np.linspace(-y_max, y_max, 100)
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
            Z_mesh = X_mesh**2 + k_val * (Y_mesh**2)

            fig.add_trace(go.Contour(z=Z_mesh, x=x_grid, y=y_grid, colorscale='Blues', opacity=0.4, showscale=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(symbol='star', size=14, color='gold', line=dict(color='black', width=1)), name='Minimum (0,0)'), row=1, col=1)

            for name in selected_optims:
                pts = np.array(trajectories[name][:step_limit])
                l_pts = loss_histories[name][:step_limit]
                c = COLORS[name]
                fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='lines+markers', marker=dict(size=4), line=dict(color=c, width=2), name=name), row=1, col=1)
                fig.add_trace(go.Scatter(y=l_pts, mode='lines', line=dict(color=c, width=2), name=name, showlegend=False), row=1, col=2)

            fig.update_xaxes(title_text="x", row=1, col=1)
            fig.update_yaxes(title_text="y", row=1, col=1)
            fig.update_xaxes(title_text="Iteration", row=1, col=2)
            fig.update_yaxes(title_text="Loss L(θ)", type="log", row=1, col=2)
            fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
            return fig

        if animate:
            step_stride = max(1, iterations // 30)
            for step in range(2, iterations + 1, step_stride):
                plot_placeholder.plotly_chart(render_plot(step), use_container_width=True)
                time.sleep(anim_speed)
            plot_placeholder.plotly_chart(render_plot(iterations + 1), use_container_width=True)
        else:
            plot_placeholder.plotly_chart(render_plot(iterations + 1), use_container_width=True)

# =========================================================
# MODULE 3: PART B - NEURAL NETWORK TRAINING FROM SCRATCH
# =========================================================

class ScratchMLP:
    def __init__(self, input_dim=30):
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(input_dim, 16) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros((1, 16)),
            'W2': np.random.randn(16, 8) * np.sqrt(2.0 / 16),
            'b2': np.zeros((1, 8)),
            'W3': np.random.randn(8, 1) * np.sqrt(2.0 / 8),
            'b3': np.zeros((1, 1))
        }

    def relu(self, z): 
        return np.maximum(0, z)

    def sigmoid(self, z): 
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    def forward(self, X, params=None):
        p = params if params is not None else self.params
        z1 = X @ p['W1'] + p['b1']
        a1 = self.relu(z1)
        z2 = a1 @ p['W2'] + p['b2']
        a2 = self.relu(z2)
        z3 = a2 @ p['W3'] + p['b3']
        a3 = self.sigmoid(z3)
        cache = (X, z1, a1, z2, a2, z3, a3)
        return a3, cache

    def compute_loss(self, y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, cache, y_true, params=None):
        p = params if params is not None else self.params
        X, z1, a1, z2, a2, z3, a3 = cache
        m = X.shape[0]

        dz3 = (a3 - y_true) / m
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ p['W3'].T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ p['W2'].T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

with tab2:
    st.subheader("Train MLP on Breast Cancer Wisconsin Dataset")
    
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target.reshape(-1, 1), test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", X_train.shape[1])
    c2.metric("Train Samples", X_train.shape[0])
    c3.metric("Test Samples", X_test.shape[0])
    epochs = c4.slider("Epochs", 20, 200, 80, 10)

    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        batch_mode = st.radio("Batch Type", ["Full Batch", "Mini-Batch"], horizontal=True)
    with col_batch2:
        if batch_mode == "Mini-Batch":
            batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
        else:
            batch_size = X_train.shape[0]

    selected_opts_nn = st.multiselect("Select Optimizers for NN", list(COLORS.keys()), default=["SGD", "SGD + Momentum", "Adam", "AdamW"])

    if st.button("Train Models", type="primary"):
        history = {}
        summary_rows = []
        train_bar = st.progress(0)
        status_text = st.empty()

        for idx, opt_name in enumerate(selected_opts_nn):
            status_text.text(f"Training with {opt_name}...")
            model = ScratchMLP(input_dim=30)
            opt = get_optimizer_instance(opt_name, lr, beta, beta1, beta2, weight_decay)
            
            tr_losses, te_losses, tr_accs, te_accs = [], [], [], []
            eff_lrs = []
            n_samples = X_train.shape[0]

            for ep in range(epochs):
                indices = np.random.permutation(n_samples)
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]

                for i in range(0, n_samples, batch_size):
                    X_b = X_train_shuffled[i:i + batch_size]
                    y_b = y_train_shuffled[i:i + batch_size]

                    if opt_name == "NAG":
                        la_params = opt.get_lookahead(model.params)
                        _, cache = model.forward(X_b, params=la_params)
                        grads = model.backward(cache, y_b, params=la_params)
                    else:
                        _, cache = model.forward(X_b)
                        grads = model.backward(cache, y_b)

                    model.params = opt.step(model.params, grads)

                pred_train, _ = model.forward(X_train)
                pred_test, _ = model.forward(X_test)
                
                loss_tr = model.compute_loss(pred_train, y_train)
                loss_te = model.compute_loss(pred_test, y_test)
                acc_tr = np.mean((pred_train >= 0.5) == y_train)
                acc_te = np.mean((pred_test >= 0.5) == y_test)

                tr_losses.append(loss_tr)
                te_losses.append(loss_te)
                tr_accs.append(acc_tr)
                te_accs.append(acc_te)

                # Track effective LR for W1[0, 0]
                if opt_name == "AdaGrad":
                    eff = lr / (np.sqrt(opt.G['W1'][0, 0]) + 1e-8)
                elif opt_name == "RMSProp":
                    eff = lr / (np.sqrt(opt.v['W1'][0, 0]) + 1e-8)
                elif opt_name in ["Adam", "AdamW"]:
                    v_hat = opt.v['W1'][0, 0] / (1 - beta2 ** opt.t)
                    eff = lr / (np.sqrt(v_hat) + 1e-8)
                else:
                    eff = lr
                eff_lrs.append(eff)

            train_bar.progress((idx + 1) / len(selected_opts_nn))

            history[opt_name] = {
                'tr_loss': tr_losses, 'te_loss': te_losses,
                'tr_acc': tr_accs, 'te_acc': te_accs, 'eff_lr': eff_lrs
            }

            final_loss = te_losses[-1]
            conv_ep = next((i + 1 for i, l in enumerate(te_losses) if abs(l - final_loss) <= 0.01 * final_loss), epochs)
            current_mode = f"Mini-Batch ({batch_size})" if batch_mode == "Mini-Batch" else "Full Batch"

            summary_rows.append({
                "Optimizer": opt_name,
                "Mode": current_mode,
                "Final Train Loss": f"{tr_losses[-1]:.4f}",
                "Final Test Loss": f"{te_losses[-1]:.4f}",
                "Train Acc.": f"{tr_accs[-1]*100:.2f}%",
                "Test Acc.": f"{te_accs[-1]*100:.2f}%",
                "Convergence Epoch": conv_ep
            })

        status_text.success("Training complete!")

        # Dashboard Visuals
        fig_nn = make_subplots(
            rows=2, cols=2, 
            subplot_titles=(
                "<b>Train Loss vs Epoch</b>", 
                "<b>Test Loss vs Epoch</b>", 
                "<b>Test Accuracy vs Epoch</b>", 
                "<b>Effective LR (W1[0,0]) vs Epoch</b>"
            ),
            vertical_spacing=0.18,
            horizontal_spacing=0.12
        )
        
        for opt_name in selected_opts_nn:
            c = COLORS[opt_name]
            ep_axis = list(range(1, epochs + 1))
            fig_nn.add_trace(go.Scatter(x=ep_axis, y=history[opt_name]['tr_loss'], mode='lines', line=dict(color=c, width=2), name=opt_name), row=1, col=1)
            fig_nn.add_trace(go.Scatter(x=ep_axis, y=history[opt_name]['te_loss'], mode='lines', line=dict(color=c, width=2), showlegend=False), row=1, col=2)
            fig_nn.add_trace(go.Scatter(x=ep_axis, y=history[opt_name]['te_acc'], mode='lines', line=dict(color=c, width=2), showlegend=False), row=2, col=1)
            fig_nn.add_trace(go.Scatter(x=ep_axis, y=history[opt_name]['eff_lr'], mode='lines', line=dict(color=c, width=2), showlegend=False), row=2, col=2)

        for r in [1, 2]:
            for col_idx in [1, 2]:
                fig_nn.update_xaxes(title=dict(text="<b>Epoch</b>", font=dict(size=12, color="#ffffff")), showgrid=True, gridcolor='rgba(255,255,255,0.2)', row=r, col=col_idx)

        fig_nn.update_yaxes(title=dict(text="<b>Training Loss (BCE)</b>", font=dict(size=12, color="#ffffff")), showgrid=True, gridcolor='rgba(255,255,255,0.2)', row=1, col=1)
        fig_nn.update_yaxes(title=dict(text="<b>Test Loss (BCE)</b>", font=dict(size=12, color="#ffffff")), showgrid=True, gridcolor='rgba(255,255,255,0.2)', row=1, col=2)
        fig_nn.update_yaxes(title=dict(text="<b>Test Accuracy (0 to 1)</b>", font=dict(size=12, color="#ffffff")), showgrid=True, gridcolor='rgba(255,255,255,0.2)', row=2, col=1)
        fig_nn.update_yaxes(title=dict(text="<b>Effective LR (Scaling)</b>", font=dict(size=12, color="#ffffff")), showgrid=True, gridcolor='rgba(255,255,255,0.2)', row=2, col=2)

        fig_nn.update_layout(
            height=780,
            margin=dict(l=80, r=40, t=60, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_nn, use_container_width=True)

        st.subheader("Comparison Table (B3)")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# =========================================================
# MODULE 4: THEORY & HOW-TO-USE GUIDE (A4 & Section 4)
# =========================================================
with tab3:
    st.subheader("📖 How to Use This Tool")
    st.markdown(r"""
    1. **Part A (2D Loss Surface):**
       - Choose an anisotropic loss surface ($L_1$ to $L_4$) to test different Hessian condition numbers.
       - Select multiple optimizers to compare their trajectory curves on the contour plot.
       - Toggle **Live Animation Mode** to see them iterate step-by-step.
    2. **Part B (Neural Network Training):**
       - Train a from-scratch 3-layer MLP on the Breast Cancer dataset.
       - Select between **Full Batch** and **Mini-Batch** gradient descent.
       - Inspect BCE loss curves, accuracy, and effective learning rate scaling.
    """)
    
    st.markdown("---")
    st.subheader("🧠 Mathematical Formulations & Explain-as-you-go Notes")
    st.markdown(r"""
    * **NAG vs. Momentum:** NAG evaluates the gradient at a projected "look-ahead" point ($\theta_t - \beta v_{t-1}$). If momentum pushes parameters towards a steep slope, the look-ahead gradient detects it early and slows down update velocity, dampening overshoot.
    * **AdaGrad Scaling:** Parameters with historically large or frequent gradient updates accumulate higher $G_t$, drastically scaling down their effective learning rate.
    * **RMSProp vs. AdaGrad:** AdaGrad monotonically accumulates all historical squared gradients ($G_t = G_{t-1} + g_t^2$), halting learning early. RMSProp replaces this with an exponential moving average ($v_t = \beta v_{t-1} + (1-\beta) g_t^2$), allowing learning to continue dynamically.
    * **AdamW vs. Adam (Decoupled Weight Decay):** Standard L2 regularization adds $\lambda \theta$ directly to $g_t$, which gets scaled down by $\sqrt{\hat{v}_t}$ in Adam. AdamW decouples weight decay by subtracting $\eta \lambda \theta$ directly, regularizing all weights uniformly.
    """)