import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from mpi4py import MPI
import ufl

# Set seed
def set_seed(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# DNN architecture definition
# ==========================================
class ForwardPDESurrogate(nn.Module):
    def __init__(self, input_dim=3):  
        super(ForwardPDESurrogate, self).__init__()
        
        self.register_buffer('mu', torch.zeros(input_dim))
        self.register_buffer('sigma', torch.ones(input_dim))
        
        # 5 hidden layers: 40, 80, 40, 20, 10
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
            nn.Linear(10, 1) # Scalar output G
        )

    # Calculates and sets the mean and std deviation from the training data
    def fit_scalers(self, X_train):
        self.mu = torch.mean(X_train, dim=0)
        self.sigma = torch.std(X_train, dim=0) + 1e-8

    # Forward pass for training
    def forward(self, x):
        x_scaled = (x - self.mu) / self.sigma
        return self.network(x_scaled)

    @torch.no_grad()
    # API for the sOED framework
    def predict_observation(self, U, theta_N_y, theta_P_y):
        U_t = torch.atleast_1d(torch.as_tensor(U, dtype=torch.float32))
        t_Ny_t = torch.atleast_2d(torch.as_tensor(theta_N_y, dtype=torch.float32))
        t_Py_t = torch.atleast_2d(torch.as_tensor(theta_P_y, dtype=torch.float32))

        x = torch.cat([t_Ny_t, t_Py_t, U_t.unsqueeze(1) if U_t.ndim == 1 else U_t], dim=1)
        
        self.eval()
        y_pred = self.forward(x)
        return y_pred.squeeze(-1).cpu().numpy()


# ==========================================
# Adapted DOLFINx FEM Solver
# ==========================================
def solve_pdes_and_get_observation(t_Ny, t_Py, U_val, V_bi_val=1.0):
    # Fixed x-coordinates of cluster centers
    t_Nx = -0.7
    t_Px = 0.8
    
    comm = MPI.COMM_WORLD
    domain = dolfinx.mesh.create_rectangle(
        comm, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])], [32, 32],
        cell_type=dolfinx.mesh.CellType.triangle
    )
    
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    scalar_type = dolfinx.default_scalar_type

    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    def bottom_boundary(x):
        return np.isclose(x[1], -1.0)

    top_facets = dolfinx.mesh.locate_entities_boundary(domain, tdim - 1, top_boundary)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(domain, tdim - 1, bottom_boundary)

    facet_indices = np.hstack([top_facets, bottom_facets])
    facet_markers = np.hstack([np.full_like(top_facets, 1, dtype=np.int32),
                               np.full_like(bottom_facets, 2, dtype=np.int32)])
    sorted_facets = np.argsort(facet_indices)

    dofs_top = dolfinx.fem.locate_dofs_topological(V, tdim - 1, top_facets)
    dofs_bottom = dolfinx.fem.locate_dofs_topological(V, tdim - 1, bottom_facets)

    bcs_Ve = [
        dolfinx.fem.dirichletbc(scalar_type(V_bi_val + U_val), dofs_top, V),
        dolfinx.fem.dirichletbc(scalar_type(V_bi_val), dofs_bottom, V)
    ]

    C_func = dolfinx.fem.Function(V)
    # Fixed other parameters
    theta_s = 5.0 
    theta_h = 0.3
    coeff = theta_s / (2.0 * np.pi * theta_h**2)
    denom = 2.0 * theta_h**2
    
    def c_expr(x):
        S_N = coeff * np.exp(-((x[0] - t_Nx)**2 + (x[1] - t_Ny)**2) / denom)
        S_P = coeff * np.exp(-((x[0] - t_Px)**2 + (x[1] - t_Py)**2) / denom)
        return S_N - S_P
        
    C_func.interpolate(c_expr)

    # Solve PDE 1: Non-linear Poisson-Boltzmann
    V_e = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    
    F_nl = ufl.dot(ufl.grad(V_e), ufl.grad(v)) * ufl.dx + (ufl.exp(V_e) - ufl.exp(-V_e) - C_func) * v * ufl.dx
    J_nl = ufl.derivative(F_nl, V_e, du)
    
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F_nl, V_e, bcs=bcs_Ve, J=J_nl, petsc_options_prefix="semiconductor_snes_"
    )
    problem.solve()

    # Solve PDE 2: Linear Drift-Diffusion
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    a_lin = ufl.exp(V_e) * ufl.dot(ufl.grad(u), ufl.grad(w)) * ufl.dx
    L_lin = dolfinx.fem.Constant(domain, scalar_type(0.0)) * w * ufl.dx

    bcs_u = [
        dolfinx.fem.dirichletbc(scalar_type(U_val), dofs_top, V),
        dolfinx.fem.dirichletbc(scalar_type(0.0), dofs_bottom, V)
    ]

    linear_problem = dolfinx.fem.petsc.LinearProblem(
        a_lin, L_lin, bcs=bcs_u,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="pde2_"
    )
    u_sol = linear_problem.solve()

    n = ufl.FacetNormal(domain)
    facets_tag = dolfinx.mesh.meshtags(domain, tdim - 1, facet_indices[sorted_facets], facet_markers[sorted_facets])
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facets_tag)
    
    flux_form = dolfinx.fem.form(ufl.exp(scalar_type(V_bi_val)) * ufl.dot(ufl.grad(u_sol), n) * ds(2))
    observation = domain.comm.allreduce(dolfinx.fem.assemble_scalar(flux_form), op=MPI.SUM)
    
    return float(observation)

def load_fem_data(num_samples=8000):
    t_Ny_samples = np.random.uniform(0.1, 0.9, num_samples)
    t_Py_samples = np.random.uniform(-0.9, -0.1, num_samples)
    U_samples = np.random.uniform(-10.0, 10.0, num_samples)
    
    X_train = np.vstack((t_Ny_samples, t_Py_samples, U_samples)).T
    y_train = np.zeros(num_samples)
    
    print(f"Starting FEM solves for {num_samples} samples")
    for i in range(num_samples):
        try:
            y_train[i] = solve_pdes_and_get_observation(
                t_Ny_samples[i], t_Py_samples[i], U_samples[i]
            )
        except RuntimeError:
            y_train[i] = np.nan
              
        if (i + 1) % 100 == 0:
            print(f"Solved {i + 1}/{num_samples} PDEs...")
            
    valid_indices = ~np.isnan(y_train)
    X_clean = X_train[valid_indices]
    y_clean = y_train[valid_indices]
    
    return torch.tensor(X_clean, dtype=torch.float32), torch.tensor(y_clean, dtype=torch.float32).view(-1, 1)

# ==========================================
# Training Function
# ==========================================
def train_surrogate(X, y, epochs=500, batch_size=32, lr=1e-3, val_split=0.2):
    dataset_size = len(X)
    test_size = int(val_split * dataset_size)
    train_size = dataset_size - test_size
    
    dataset = TensorDataset(X, y)
    
    generator = torch.Generator().manual_seed(1111)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ForwardPDESurrogate(input_dim=3)
    model.fit_scalers(X)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training on {train_size} samples, testing on {test_size} samples...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train MSE: {train_loss/len(train_loader):.6e}")
            
    model.eval()
    final_test_mse = 0.0
    l2_num = 0.0
    l2_den = 0.0
    
    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_preds = model(test_X)
            batch_mse = criterion(test_preds, test_y).item()
            final_test_mse += batch_mse * test_X.size(0)
            
            l2_num += torch.sum((test_y - test_preds) ** 2).item()
            l2_den += torch.sum(test_y ** 2).item()
            
    final_test_mse /= test_size
    final_rel_l2 = np.sqrt(l2_num) / np.sqrt(l2_den) if l2_den > 0 else 0.0
    
    print("-" * 40)
    print(f"Optimization Complete!")
    print(f"Final Test MSE: {final_test_mse:.6e}")
    print(f"Final Relative L2 Error: {final_rel_l2 * 100:.2f}%")
    print("-" * 40)
          
    return model, final_test_mse

# ==========================================
# Main Execution & Saving
# ==========================================
if __name__ == "__main__":
    set_seed(1111)
    
    X_train, y_train = load_fem_data(num_samples=8000)
    trained_model, test_mse = train_surrogate(X_train, y_train, epochs=500, val_split=0.2)
    
    save_path = "semiconductor_gaussian_pde_surrogate.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"\nSuccess! The trained model weights have been securely saved locally to: {os.path.abspath(save_path)}")