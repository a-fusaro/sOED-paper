import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import logsumexp

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm to see the progress bar: pip install tqdm")
    def tqdm(iterable, **kwargs): return iterable

# ============================================================================
# SURROGATE MODEL LOADING
# ============================================================================
class ForwardPDESurrogate(nn.Module):
    def __init__(self, input_dim=3):
        super(ForwardPDESurrogate, self).__init__()
        self.register_buffer('mu', torch.zeros(input_dim))
        self.register_buffer('sigma', torch.ones(input_dim))
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.GELU(),
            nn.Linear(40, 80),
            nn.GELU(),
            nn.Linear(80, 40),
            nn.GELU(),
            nn.Linear(40, 20),
            nn.GELU(),
            nn.Linear(20, 10),
            nn.GELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        sigma_safe = torch.clamp(self.sigma, min=1e-8)
        x_scaled = (x - self.mu) / sigma_safe
        return self.network(x_scaled)

surrogate_model = ForwardPDESurrogate(input_dim=3)
model_path = "semiconductor_gaussian_pde_surrogate.pt"
if os.path.exists(model_path):
    surrogate_model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Successfully loaded surrogate model from {model_path}")
else:
    print(f"Warning: {model_path} not found. Ensure you run the 3-input training script first.")

surrogate_model.eval()

# ============================================================================
# EXPECTED INFORMATION GAIN (EIG) EVALUATION VIA NESTED MONTE CARLO
# ============================================================================
# Grid of applied bias voltages to evaluate
n_grid_points = 50
U_grid = np.linspace(-6.0, 6.0, n_grid_points)

# Number of Monte Carlo samples
N_out = 20000
N_in = 20000
batch_size = 1000  # Inner calculation chunk size to prevent RAM overflow

# Sample from the defined uniform priors
theta_N_y_out = np.random.uniform(0.2, 0.8, N_out)
theta_P_y_out = np.random.uniform(-0.8, -0.2, N_out)
theta_N_y_in = np.random.uniform(0.2, 0.8, N_in)
theta_P_y_in = np.random.uniform(-0.8, -0.2, N_in)

# Define noise scales 
noise_scales = [0.02, 0.05, 0.1]
trace_colors = {0.02: "forestgreen", 0.05: "navy", 0.1: "firebrick"}
all_EIG_vals = {}

# ============================================================================
# NESTED MONTE CARLO CALCULATION (BATCHED)
# ============================================================================
with torch.no_grad():
    for noise_val in noise_scales:
        print(f"\nCalculating EIG landscape over U in [-6.0, 6.0] for noise scale sigma = {noise_val}...")
        EIG_vals = np.zeros(n_grid_points)
        
        for k, U_val in enumerate(tqdm(U_grid, desc=f"Computing EIG (\u03C3={noise_val})", unit="step")):
            
            # Evaluate surrogate for all outer samples once 
            X_out = torch.zeros(N_out, 3, dtype=torch.float32)
            X_out[:, 0] = torch.tensor(theta_N_y_out)
            X_out[:, 1] = torch.tensor(theta_P_y_out)
            X_out[:, 2] = U_val
            G_out = surrogate_model(X_out).numpy().flatten()
            
            # Simulate observations with proportional noise 
            sigma_out = noise_val * (1.0 + np.abs(G_out))
            y_out = G_out + np.random.normal(0, sigma_out)
            
            # Evaluate surrogate for all inner samples once 
            X_in = torch.zeros(N_in, 3, dtype=torch.float32)
            X_in[:, 0] = torch.tensor(theta_N_y_in)
            X_in[:, 1] = torch.tensor(theta_P_y_in)
            X_in[:, 2] = U_val
            G_in = surrogate_model(X_in).numpy().flatten()
            sigma_in = noise_val * (1.0 + np.abs(G_in))
            
            # Compute outer log likelihoods 
            log_like_out = -0.5 * np.log(2 * np.pi * sigma_out**2) - 0.5 * ((y_out - G_out) / sigma_out)**2
            
            # Batched inner evidence calculation 
            log_evidence = np.zeros(N_out)
            
            # Prepare inner row matrices 
            G_in_row = G_in[None, :]          
            sigma_in_row = sigma_in[None, :]  
            
            # Iterate over outer samples in chunks
            for i in range(0, N_out, batch_size):
                i_end = min(i + batch_size, N_out)
                y_out_batch = y_out[i:i_end, None]  
                
                # Compute inner likelihood matrix for the batch
                log_like_in_batch = (
                    -0.5 * np.log(2 * np.pi * sigma_in_row**2)
                    - 0.5 * ((y_out_batch - G_in_row) / sigma_in_row)**2
                )
                
                # Marginal log-evidence for this batch
                log_evidence[i:i_end] = logsumexp(log_like_in_batch, axis=1) - np.log(N_in)
            
            # Compute EIG for this U 
            EIG_vals[k] = np.mean(log_like_out - log_evidence)
            
        # Store results for this noise level
        all_EIG_vals[noise_val] = EIG_vals

print("\nEIG calculations complete.")

# ============================================================================
# PLOTTING THE EIG LANDSCAPE
# ============================================================================
plt.figure(figsize=(8, 5))

# Plot the computed EIG curves for each noise scale
for noise_val in noise_scales:
    plt.plot(
        U_grid, 
        all_EIG_vals[noise_val], 
        color=trace_colors[noise_val], 
        linewidth=2.5, 
        label=f'$\\sigma={noise_val}$'
    )

plt.xlabel("Applied Voltage $U$", fontsize=12)
plt.ylabel("Expected Information Gain (EIG)", fontsize=12)
plt.title("EIG landscapes for a single static experiment across noise scales", fontsize=14)
plt.xlim(-6.0, 6.0)
plt.ylim(bottom=0) 
plt.grid(True, ls=':', alpha=0.7)
plt.legend(loc="upper left", fontsize=11)

plt.tight_layout()
plt.savefig("semiconductor_combined_EIG_landscapes.png", dpi=300)
plt.show()
