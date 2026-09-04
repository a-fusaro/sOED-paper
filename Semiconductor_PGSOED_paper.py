import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.typeDict = np.sctypeDict

from oed_package.pg_soed import PGsOED

# ============================================================================
# LOAD TRAINED GAUSSIAN SURROGATE DNN MODEL (3 Inputs)
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
# PARAMETERS & PROBLEM CONFIGURATION 
# ============================================================================
n_stage = 8        
n_param = 2        
n_design = 1       
n_obs = 1          
n_phys_state = 1   
n_grid = 50        


prior_info = [
    ("uniform", 0.2, 0.6),   # Corresponds to U([0.2, 0.8])
    ("uniform", -0.8, 0.6),  # Corresponds to U([-0.8, -0.2])
]

design_bounds = [(-0.75, 0.75)]  

noise_loc = 0.0

init_phys_state = (0.0,)        
post_rvs_method = "Rejection"        

random_state = 2026
np.random.seed(random_state)
torch.manual_seed(random_state)

def semiconductor_surrogate_model(stage, theta, d, xp=None):
    n_sample = max(len(theta), len(d), len(xp) if xp is not None else 0)
    t_Ny = theta[:, 0]
    t_Py = theta[:, 1]
    U = xp.flatten() if (xp is not None and len(xp) > 0) else np.zeros(n_sample)

    X_input = torch.zeros(n_sample, 3, dtype=torch.float32)
    X_input[:, 0] = torch.tensor(t_Ny, dtype=torch.float32)
    X_input[:, 1] = torch.tensor(t_Py, dtype=torch.float32)
    X_input[:, 2] = torch.tensor(U, dtype=torch.float32)

    with torch.no_grad():
        preds = surrogate_model(X_input)

    return preds.detach().numpy()

def reward_fun(stage, xb, xp, d, y):
    if d is None:
        return 0.0
    movement_penalty = 0.15  
    return -movement_penalty * float(np.sum(np.square(d)))

def phys_state_fun(xp, stage, d, y):
    new_xp = np.array(xp) + np.array(d)
    return np.clip(new_xp, -4.0, 4.0)  

phys_state_info = (n_phys_state, init_phys_state, phys_state_fun)

def plot_semiconductor_trajectories(
    soed_instance, 
    title, 
    file_suffix, 
    theta_val, 
    agent_instance=None,
    x_lims=(0.2, 0.8),  
    y_lims=(-0.8, -0.2), 
    noise_base_scale=0.05
):
    if agent_instance is None:
        agent_instance = soed_instance

    np.random.seed(112)

    d_hist = np.zeros((soed_instance.n_stage, soed_instance.n_design))
    y_hist = np.zeros((soed_instance.n_stage, soed_instance.n_obs))
    xp = np.array(soed_instance.init_xp)

    xb_list = []

    for i in range(soed_instance.n_stage):
        if hasattr(agent_instance, 'get_design'):
            try:
                d_hist[i] = agent_instance.get_design(i, d_hist[:i], y_hist[:i])
            except TypeError:
                d_hist[i] = agent_instance.get_design(i, d_hist=d_hist[:i], y_hist=y_hist[:i])
        else:
            d_hist[i] = soed_instance.get_design(i, d_hist[:i], y_hist[:i])

        G = soed_instance.m_f(
            i,
            theta_val.reshape(1, -1),
            d_hist[i].reshape(1, -1),
            xp.reshape(1, -1),
        ).flatten()

        noise_std = noise_base_scale * (1.0 + np.abs(G))
        y_hist[i] = np.random.normal(loc=G, scale=noise_std)

        xp = soed_instance.xp_f(xp, i, d_hist[i], y_hist[i])

        xb_stage = soed_instance.get_xb(d_hist=d_hist[:i+1], y_hist=y_hist[:i+1])
        xb_list.append(xb_stage)

    all_densities = np.concatenate([xb[:, -1] for xb in xb_list])
    vmin, vmax = np.min(all_densities), np.max(all_densities)
    shared_levels = np.linspace(vmin, vmax, 16)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, xb in enumerate(xb_list):
        n_grid = int(np.sqrt(xb.shape[0]))
        ax = axes[i]
        ax.set_aspect("auto")

        cf = ax.contourf(
            xb[:, 0].reshape(n_grid, n_grid),
            xb[:, 1].reshape(n_grid, n_grid),
            xb[:, -1].reshape(n_grid, n_grid),
            cmap="viridis",
            levels=shared_levels,
            vmin=vmin,
            vmax=vmax
        )

        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax.scatter(
            theta_val[0],
            theta_val[1],
            marker="*",
            s=150,
            c="magenta",
            edgecolors="black",
            zorder=5,
            label="True Parameters"
        )

        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("$\\theta_{N,y}$", fontsize=10)
        ax.set_ylabel("$\\theta_{P,y}$", fontsize=10)
        ax.set_title(f"$p(\\theta|I_{{{i}}})$", fontsize=11)
        ax.grid(True, ls="--", alpha=0.5)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{title}, True $\\theta = ({theta_val[0]}, {theta_val[1]})$", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"semiconductor_trajectory_{file_suffix}.png", dpi=300)
    plt.show()

    # Calculate MSE for posterior means against assumed true theta
    mses = []
    for xb in xb_list:
        weights = xb[:, -1]
        if np.sum(weights) == 0:
            mean_t1 = np.mean(xb[:, 0])
            mean_t2 = np.mean(xb[:, 1])
        else:
            mean_t1 = np.average(xb[:, 0], weights=weights)
            mean_t2 = np.average(xb[:, 1], weights=weights)
        
        mse = ((mean_t1 - theta_val[0])**2 + (mean_t2 - theta_val[1])**2) / 2.0
        mses.append(mse)
    
    return mses

# ============================================================================
# LOOP OVER NOISE SCALES: INITIALIZE, TRAIN, PLOT POSTERIORS & EVALUATE
# ============================================================================

noise_scales = [0.02, 0.05, 0.1]
all_U_states = {}
all_rewards = {}
all_mses = {}

for noise_val in noise_scales:
    print(f"\n{'='*55}")
    print(f"RUNNING EXPERIMENT FOR NOISE SCALE = {noise_val}")
    print(f"{'='*55}")

    noise_info = [(noise_loc, noise_val, noise_val)]

    soed = PGsOED(
        model_fun=semiconductor_surrogate_model,
        n_stage=n_stage,
        n_param=n_param,
        n_design=n_design,
        n_obs=n_obs,
        prior_info=prior_info,
        design_bounds=design_bounds,
        noise_info=noise_info,
        reward_fun=reward_fun,
        phys_state_info=phys_state_info,
        n_grid=n_grid,
        post_rvs_method=post_rvs_method,
        random_state=random_state,
        actor_dimns=[80, 80],
        critic_dimns=[80, 80],
    )

    soed.initialize()

    actor_optimizer = optim.Adam(soed.actor_net.parameters(), lr=3e-3)
    actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.98) 

    n_critic_update = 100
    critic_optimizer = optim.Adam(soed.critic_net.parameters(), lr=5e-3)
    critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.98) 

    soed.soed(
        n_update=100, 
        n_traj=1000,
        actor_optimizer=actor_optimizer,
        actor_lr_scheduler=actor_lr_scheduler,
        n_critic_update=n_critic_update,
        critic_optimizer=critic_optimizer,
        critic_lr_scheduler=critic_lr_scheduler,
        design_noise_scale=0.3,
        design_noise_decay=0.98, 
    )

    true_theta = np.array([0.65, -0.3]) 

    stage_mses = plot_semiconductor_trajectories(
        agent_instance=soed,            
        soed_instance=soed,              
        title=f"PG-sOED posterior contours ($\\sigma={noise_val}$)", 
        file_suffix=f"pg_soed_gaussian_8stage_noise_{noise_val}",  
        theta_val=true_theta, 
        x_lims=(0.2, 0.8),  
        y_lims=(-0.8, -0.2),
        noise_base_scale=noise_val
    )
    all_mses[noise_val] = stage_mses

    print("Running built-in assessment across 10,000 trajectories...")
    soed.asses(n_traj=10000)

    final_rewards = soed.rewards_hist[:, -1]
    design_history = soed.dcs_hist  

    all_rewards[noise_val] = final_rewards

    mean_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)
    min_reward = np.min(final_rewards)
    max_reward = np.max(final_rewards)

    print("\n+" + "-"*35 + "+")
    print(f"| {'EVALUATION METRICS (10,000 TRAJ)':^33} |")
    print("+" + "-"*19 + "+" + "-"*15 + "+")
    print(f"| {'Metric':<17} | {'Value':<13} |")
    print("+" + "-"*19 + "+" + "-"*15 + "+")
    print(f"| {'Mean Reward':<17} | {mean_reward:<13.6f} |")
    print(f"| {'Std Dev':<17} | {std_reward:<13.6f} |")
    print(f"| {'Min Reward':<17} | {min_reward:<13.6f} |")
    print(f"| {'Max Reward':<17} | {max_reward:<13.6f} |")
    print("+" + "-"*19 + "+" + "-"*15 + "+\n")

    n_traj_eval = design_history.shape[0]
    U_states = np.zeros((n_traj_eval, n_stage + 1)) 
    current_U = np.full(n_traj_eval, init_phys_state[0])

    for k in range(n_stage):
        U_states[:, k] = current_U
        current_U = np.clip(current_U + design_history[:, k, 0], -4.0, 4.0)
    U_states[:, n_stage] = current_U

    all_U_states[noise_val] = U_states


# ============================================================================
# 6. PHYSICAL STATE (BIAS VOLTAGE U_k) FOR 10,000 TRAJECTORIES
# ============================================================================
plt.figure(figsize=(12, 6))
stages = np.arange(n_stage + 1)
trace_colors = {0.02: "forestgreen", 0.05: "navy", 0.1: "firebrick"}

for noise_val in noise_scales:
    plt.plot(
        stages, 
        all_U_states[noise_val].T,  
        color=trace_colors[noise_val], 
        linewidth=1.0, 
        alpha=0.015
    )

# Baseline reference line at U_0 = 0
plt.axhline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

plt.xlim(0, n_stage)
plt.ylim(-4.1, 4.1)
plt.xticks(stages, [f"Stage {k}" for k in stages], fontsize=11)
plt.xlabel("Stage $k$", fontsize=12)
plt.ylabel("Physical State (Voltage $U_k$)", fontsize=12)
plt.title("Physical state trajectories ($U_k$) for different noise levels", fontsize=14)
plt.grid(True, ls=":", alpha=0.5)

handles = [mlines.Line2D([], [], color=trace_colors[nv], label=f"$\\sigma={nv}$", linewidth=2.0) for nv in noise_scales]
plt.legend(handles=handles, loc="upper right", fontsize=11)

plt.tight_layout()
plt.savefig("semiconductor_design_trajectories_all_10k_combined.png", dpi=300)
plt.show()

# ============================================================================
# REWARD HISTOGRAM PLOT FOR 10000 TRAJECTORIES
# ============================================================================
plt.figure(figsize=(6, 4))
bins_reward = np.linspace(0, 6, 80)
hist_colors = {0.02: 'forestgreen', 0.05: 'navy', 0.1: 'firebrick'}

for noise_val in noise_scales:
    plt.hist(
        all_rewards[noise_val], 
        alpha=0.6, 
        bins=bins_reward, 
        color=hist_colors[noise_val], 
        label=f'$\\sigma={noise_val}$', 
        edgecolor='none'
    )

plt.xlim(0, 3)
plt.xlabel('Reward', fontsize=11)
plt.ylabel('Counts', fontsize=11)
plt.title('Histogram of rewards', fontsize=11)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, ls=':', alpha=0.5)

plt.tight_layout()
plt.savefig("figure_rewards_histograms_combined.png", dpi=300)
plt.show()


# ============================================================================
# POSTERIOR MSE TABLE
# ============================================================================
print("\n+" + "-"*92 + "+")
print(f"| {'POSTERIOR MEAN SQUARED ERROR (MSE) VS TRUE THETA':^90} |")
print("+" + "-"*15 + "+" + "-"*76 + "+")
header = f"| {'Noise (sigma)':<13} |"
for s in range(1, n_stage + 1):
    header += f" {'Stage ' + str(s):<7} |"
print(header)
print("+" + "-"*15 + "+" + "-"*76 + "+")

for noise_val in noise_scales:
    row = f"| {noise_val:<13.3f} |"
    for mse in all_mses[noise_val]:
        row += f" {mse:<7.4f} |"
    print(row)
print("+" + "-"*15 + "+" + "-"*76 + "+\n")
