import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
np.typeDict = np.sctypeDict
from oed_package.pg_soed import PGsOED

# ============================================================================
# LOAD TRAINED GAUSSIAN SURROGATE DNN MODEL 
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
# PARAMETERS & PROBLEM CONFIGURATION (8 STAGES)
# ============================================================================
n_stage = 8        
n_param = 2        
n_design = 1       
n_obs = 1          
n_phys_state = 1   
n_grid = 50        

noise_val = 0.05
noise_loc = 0.0
noise_info = [(noise_loc, noise_val, noise_val)]

prior_info = [
    ("uniform", 0.2, 0.6),   # Corresponds to U([0.2,0.8]) 
    ("uniform", -0.8, 0.6),  # Corresponds to U([-0.8,-0.2])
]

design_bounds = [(-0.75, 0.75)]  # dk design bound
init_phys_state = (0.0,)        
post_rvs_method = "Rejection"        

random_state = 2026
np.random.seed(random_state)
torch.manual_seed(random_state)

def semiconductor_surrogate_model(stage, theta, d, xp=None):
    n_sample = max(
        theta.shape[0] if theta is not None else 0,
        d.shape[0] if d is not None else 0,
        xp.shape[0] if xp is not None else 0
    )

    X_input = torch.zeros(n_sample, 3, dtype=torch.float32)
    X_input[:, 0] = torch.tensor(np.broadcast_to(theta[:, 0], n_sample), dtype=torch.float32)
    X_input[:, 1] = torch.tensor(np.broadcast_to(theta[:, 1], n_sample), dtype=torch.float32)

    if xp is not None and xp.size > 0:
        X_input[:, 2] = torch.tensor(np.broadcast_to(xp.flatten(), n_sample), dtype=torch.float32)

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
    return np.clip(new_xp, -4.0, 4.0)    # Uk total voltage bounds 

phys_state_info = (n_phys_state, init_phys_state, phys_state_fun)

# ============================================================================
# INDEPENDENT EVALUATION FUNCTION
# ============================================================================
def evaluate_agent_independently(soed_instance, agent_get_design, n_traj=10000):
    np.random.seed(random_state)
    rewards_hist = np.zeros((n_traj, soed_instance.n_stage + 1))
    dcs_hist = np.zeros((n_traj, soed_instance.n_stage, soed_instance.n_design))
    
    for ep in range(n_traj):
        t_Ny = np.random.uniform(0.2, 0.8)
        t_Py = np.random.uniform(-0.8, -0.2)
        theta = np.array([t_Ny, t_Py])
        
        xp = np.array(soed_instance.init_xp)
        d_hist, y_hist = [], []
        
        for t in range(soed_instance.n_stage):
            d_hist_arr = np.array(d_hist) if len(d_hist) > 0 else np.empty((0, soed_instance.n_design))
            y_hist_arr = np.array(y_hist) if len(y_hist) > 0 else np.empty((0, soed_instance.n_obs))

            if hasattr(agent_get_design, 'get_design'):
                try:
                    d = agent_get_design.get_design(t, d_hist_arr, y_hist_arr)
                except TypeError:
                    d = agent_get_design.get_design(t, d_hist=d_hist_arr, y_hist=y_hist_arr)
            else:
                d = agent_get_design(t, d_hist_arr, y_hist_arr)
                
            d_hist.append(d)
            
            G = soed_instance.m_f(t, theta.reshape(1, -1), np.array(d).reshape(1, -1), xp.reshape(1, -1)).flatten()
            
            noise_std = noise_val * (1.0 + np.abs(G))
            y = np.random.normal(loc=G, scale=noise_std)
            y_hist.append(y)
            
            xp = soed_instance.xp_f(xp, t, d, y)
            dcs_hist[ep, t, :] = d

        # Native total reward computation with stage-wise breakdown
        d_hist_np = np.array(d_hist)
        y_hist_np = np.array(y_hist)
        _, reward_stage_hist = soed_instance.get_total_reward(d_hist_np, y_hist_np, return_reward_hist=True)
        
        rewards_hist[ep, :] = reward_stage_hist
        
    return rewards_hist, dcs_hist

# ============================================================================
# GREEDY AND BATCH AGENT CLASSES
# ============================================================================
class PGGreedyAgent:
    def __init__(self, soed_instance):
        self.soed = soed_instance
        self.policy_net = copy.deepcopy(soed_instance.actor_net).double()
        self.critic_net = copy.deepcopy(soed_instance.critic_net).double()
        self.N = self.soed.n_stage
        self.Nd = self.soed.n_design
        self.Ny = getattr(self.soed, 'n_obs', 1)
        self.target_dim = self.N + (self.N - 1) * self.Nd + (self.N - 1) * self.Ny
        
        self._adapt_input_layer(self.policy_net, self.target_dim)
        self._adapt_input_layer(self.critic_net, self.target_dim)

    def _adapt_input_layer(self, net, target_dim):
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                setattr(net, name, nn.Linear(target_dim, module.out_features).double())
                return True
            else:
                if self._adapt_input_layer(module, target_dim): 
                    return True
        return False

    def _get_input_tensor(self, stage, d_hist=None, y_hist=None):
        ek = np.zeros(self.N)
        ek[int(stage)] = 1.0
        
        d_pad = np.zeros((self.N - 1) * self.Nd)
        if d_hist is not None and len(d_hist) > 0:
            d_flat = np.array(d_hist).flatten()
            d_pad[:len(d_flat)] = d_flat
            
        y_pad = np.zeros((self.N - 1) * self.Ny)
        if y_hist is not None and len(y_hist) > 0:
            y_flat = np.array(y_hist).flatten()
            y_pad[:len(y_flat)] = y_flat
            
        feat = np.concatenate([ek, d_pad, y_pad])
        return torch.tensor(feat, dtype=torch.float64).unsqueeze(0)
    
    def get_design(self, stage, d_hist=None, y_hist=None):
        if isinstance(d_hist, dict):
            y_hist = d_hist.get('y_hist', [])
            d_hist = d_hist.get('d_hist', [])
            
        inp = self._get_input_tensor(stage, d_hist, y_hist)
        with torch.no_grad():
            out = self.policy_net(inp).cpu().numpy().flatten()
            
        for j in range(self.soed.n_design):
            bnd = self.soed.design_bounds[j]
            out[j] = np.clip(out[j], bnd[0], bnd[1])
        return out

    def train_offline(self, num_updates=100, batch_size=1000):
        print("\nTraining PG-Greedy Agent...")
        opt_actor = optim.Adam(self.policy_net.parameters(), lr=3e-3)
        opt_critic = optim.Adam(self.critic_net.parameters(), lr=5e-3)
        scheduler_actor = optim.lr_scheduler.ExponentialLR(opt_actor, gamma=0.98)
        scheduler_critic = optim.lr_scheduler.ExponentialLR(opt_critic, gamma=0.98)
        
        design_noise_scale, design_noise_decay = 0.3, 0.98
        
        for update in tqdm(range(num_updates), desc="Greedy Updates"):
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            
            log_probs_list = [[] for _ in range(self.soed.n_stage)]
            values_list = [[] for _ in range(self.soed.n_stage)]
            rewards_list = [[] for _ in range(self.soed.n_stage)]
            
            for ep in range(batch_size):
                t_Ny = np.random.uniform(0.2, 0.8)
                t_Py = np.random.uniform(-0.8, -0.2)
                theta = np.array([t_Ny, t_Py])
                
                xp = np.array(self.soed.init_xp)
                d_hist_ep, y_hist_ep = [], []
                
                ep_log_probs = []
                ep_values = []
                
                for t in range(self.soed.n_stage):
                    inp_tensor = self._get_input_tensor(t, d_hist_ep, y_hist_ep)
                    action_mean = self.policy_net(inp_tensor)
                    value = self.critic_net(inp_tensor)
                    
                    noise = torch.randn_like(action_mean, dtype=torch.float64) * design_noise_scale
                    action_taken = (action_mean + noise).detach()
                    
                    log_term = torch.tensor([np.log(2 * np.pi * design_noise_scale**2)], dtype=torch.float64)
                    log_prob = -0.5 * torch.sum(((action_taken - action_mean) / design_noise_scale)**2 + log_term)
                    
                    d = action_taken.cpu().numpy().flatten()
                    for j in range(self.soed.n_design):
                        bnd = self.soed.design_bounds[j]
                        d[j] = np.clip(d[j], bnd[0], bnd[1])
                    d_hist_ep.append(d)
                    
                    G = self.soed.m_f(t, theta.reshape(1,-1), d.reshape(1,-1), xp.reshape(1,-1)).flatten()
                    
                    obs_noise_std = noise_val * (1.0 + np.abs(G))
                    y = np.random.normal(loc=G, scale=obs_noise_std)
                    y_hist_ep.append(y)
                    xp = self.soed.xp_f(xp, t, d, y)
                    
                    ep_log_probs.append(log_prob)
                    ep_values.append(value)

                _, stage_rewards = self.soed.get_total_reward(
                    np.array(d_hist_ep), np.array(y_hist_ep), return_reward_hist=True
                )
                
                cum_rew = 0.0
                for t in range(self.soed.n_stage):
                    step_rew = stage_rewards[t] if t < self.soed.n_stage - 1 else stage_rewards[t] + stage_rewards[-1]
                    cum_rew += step_rew
                    log_probs_list[t].append(ep_log_probs[t])
                    values_list[t].append(ep_values[t])
                    rewards_list[t].append(cum_rew)
            
            actor_loss, critic_loss = 0.0, 0.0
            for t in range(self.soed.n_stage):
                st_log_probs = torch.stack(log_probs_list[t]).double()
                st_values = torch.cat(values_list[t]).squeeze().double()
                st_rewards = torch.tensor(rewards_list[t], dtype=torch.float64)
                
                adv = st_rewards - st_values.detach()
                actor_loss += -(st_log_probs * adv).mean()
                critic_loss += F.mse_loss(st_values, st_rewards)
            
            actor_loss.backward()
            critic_loss.backward()
            opt_actor.step()
            opt_critic.step()
            scheduler_actor.step()
            scheduler_critic.step()
            design_noise_scale *= design_noise_decay

class PGBatchAgent:
    def __init__(self, soed_instance):
        self.soed = soed_instance
        self.policy_net = copy.deepcopy(soed_instance.actor_net).double()
        self.critic_net = copy.deepcopy(soed_instance.critic_net).double()
        self._reduce_to_one_hot(self.policy_net, self.soed.n_stage)
        self._reduce_to_one_hot(self.critic_net, self.soed.n_stage)

    def _reduce_to_one_hot(self, net, n_stages):
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                setattr(net, name, nn.Linear(n_stages, module.out_features).double())
                return True
            else:
                if self._reduce_to_one_hot(module, n_stages): 
                    return True
        return False

    def _get_input_tensor(self, stage):
        one_hot = np.zeros(self.soed.n_stage)
        one_hot[int(stage)] = 1.0
        return torch.tensor(one_hot, dtype=torch.float64).unsqueeze(0)

    def get_design(self, stage, d_hist=None, y_hist=None):
        inp = self._get_input_tensor(stage)
        with torch.no_grad():
            out = self.policy_net(inp).cpu().numpy().flatten()
            
        for j in range(self.soed.n_design):
            bnd = self.soed.design_bounds[j]
            out[j] = np.clip(out[j], bnd[0], bnd[1])
        return out

    def train_offline(self, num_updates=100, batch_size=1000):
        print("\nTraining PG-Batch Agent...")
        design_noise_scale, design_noise_decay = 0.3, 0.98
        
        opt_actor = optim.Adam(self.policy_net.parameters(), lr=3e-3)
        opt_critic = optim.Adam(self.critic_net.parameters(), lr=5e-3)
        scheduler_actor = optim.lr_scheduler.ExponentialLR(opt_actor, gamma=0.98)
        scheduler_critic = optim.lr_scheduler.ExponentialLR(opt_critic, gamma=0.98)
        
        for update in tqdm(range(num_updates), desc="Batch Updates"):
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            
            log_probs = [[] for _ in range(self.soed.n_stage)]
            values = [[] for _ in range(self.soed.n_stage)]
            term_rewards = []
            
            for ep in range(batch_size):
                t_Ny = np.random.uniform(0.2, 0.8)
                t_Py = np.random.uniform(-0.8, -0.2)
                theta = np.array([t_Ny, t_Py])
                
                xp = np.array(self.soed.init_xp)
                d_hist_ep, y_hist_ep, ep_log_probs, ep_values = [], [], [], []
                
                for t in range(self.soed.n_stage):
                    inp = self._get_input_tensor(t)
                    action_mean = self.policy_net(inp)
                    value = self.critic_net(inp)
                    
                    noise = torch.randn_like(action_mean, dtype=torch.float64) * design_noise_scale
                    action_taken = (action_mean + noise).detach()
                    
                    log_term = torch.tensor([np.log(2 * np.pi * design_noise_scale**2)], dtype=torch.float64)
                    log_prob = -0.5 * torch.sum(((action_taken - action_mean) / design_noise_scale)**2 + log_term)
                    
                    d = action_taken.cpu().numpy().flatten()
                    for j in range(self.soed.n_design):
                        bnd = self.soed.design_bounds[j]
                        d[j] = np.clip(d[j], bnd[0], bnd[1])
                    d_hist_ep.append(d)
                    
                    G = self.soed.m_f(t, theta.reshape(1,-1), d.reshape(1,-1), xp.reshape(1,-1)).flatten()
                    
                    obs_noise_std = noise_val * (1.0 + np.abs(G))
                    y = np.random.normal(loc=G, scale=obs_noise_std)
                    y_hist_ep.append(y)
                    xp = self.soed.xp_f(xp, t, d, y)
                    
                    ep_log_probs.append(log_prob)
                    ep_values.append(value)
                    
                total_reward = self.soed.get_total_reward(np.array(d_hist_ep), np.array(y_hist_ep))
                term_rewards.append(total_reward)
                
                for t in range(self.soed.n_stage):
                    log_probs[t].append(ep_log_probs[t])
                    values[t].append(ep_values[t])
                    
            actor_loss = torch.tensor(0.0, dtype=torch.float64)
            critic_loss = torch.tensor(0.0, dtype=torch.float64)
            term_rewards_tensor = torch.tensor(term_rewards, dtype=torch.float64)
            
            for t in range(self.soed.n_stage):
                st_log_probs = torch.stack(log_probs[t]).double()
                st_values = torch.cat(values[t]).squeeze().double()
                
                adv = term_rewards_tensor - st_values.detach()
                actor_loss += -(st_log_probs * adv).mean()
                critic_loss += F.mse_loss(st_values, term_rewards_tensor)
            
            actor_loss.backward()
            critic_loss.backward()
            opt_actor.step()
            opt_critic.step()
            scheduler_actor.step()
            scheduler_critic.step()
            design_noise_scale *= design_noise_decay

# ============================================================================
# TRAJECTORY CONTOUR PLOTTING FUNCTION
# ============================================================================
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
        n_grid_local = int(np.sqrt(xb.shape[0]))
        ax = axes[i]
        ax.set_aspect("auto")

        cf = ax.contourf(
            xb[:, 0].reshape(n_grid_local, n_grid_local),
            xb[:, 1].reshape(n_grid_local, n_grid_local),
            xb[:, -1].reshape(n_grid_local, n_grid_local),
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
        ax.set_title(f"$p(\\theta|I_{{{i+1}}})$", fontsize=11)
        ax.grid(True, ls="--", alpha=0.5)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{title}, True $\\theta = ({theta_val[0]}, {theta_val[1]})$", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"semiconductor_trajectory_{file_suffix}.png", dpi=300)
    plt.show()

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
# INITIALIZE & TRAIN ALL 3 AGENTS WITH FIXED NOISE = 0.05
# ============================================================================
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

print(f"\n{'='*55}")
print(f"TRAINING PG-sOED AGENT (NOISE SCALE = {noise_val})")
print(f"{'='*55}")
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

greedy_agent = PGGreedyAgent(soed)
greedy_agent.train_offline(num_updates=100, batch_size=1000)

batch_agent = PGBatchAgent(soed)
batch_agent.train_offline(num_updates=100, batch_size=1000)

# Generate posteriors for all agents
true_theta = np.array([0.65, -0.3])
agent_list = [
    (soed, "PG-sOED", "pg_soed"),
    (greedy_agent, "PG-Greedy", "greedy"),
    (batch_agent, "PG-Batch", "batch")
]

all_mses = {}
for agent_inst, agent_name, file_sfx in agent_list:
    stage_mses = plot_semiconductor_trajectories(
        agent_instance=agent_inst,            
        soed_instance=soed,                     
        title=f"{agent_name} posterior contours ($\\sigma={noise_val}$)", 
        file_suffix=f"{file_sfx}_gaussian_8stage_noise_{noise_val}",  
        theta_val=true_theta, 
        x_lims=(0.2, 0.8),  
        y_lims=(-0.8, -0.2),
        noise_base_scale=noise_val
    )
    all_mses[agent_name] = stage_mses

# ============================================================================
# EVALUATION AND VISUALIZATION OF 10,000 TRAJECTORIES
# ============================================================================
print("\nEvaluating all 3 agents across 10,000 trajectories...")
all_rewards = {}
all_U_states = {}

colors = {
    "PG-sOED": "dodgerblue",
    "PG-Greedy": "springgreen",
    "PG-Batch": "darkorange"
}

for agent_inst, agent_name, _ in agent_list:
    rewards_hist, dcs_hist = evaluate_agent_independently(soed, agent_inst, n_traj=10000)
    final_rewards = rewards_hist[:, -1]
    all_rewards[agent_name] = final_rewards
    
    n_traj_eval = dcs_hist.shape[0]
    U_states = np.zeros((n_traj_eval, n_stage + 1)) 
    current_U = np.full(n_traj_eval, init_phys_state[0])

    for k in range(n_stage):
        U_states[:, k] = current_U
        current_U = np.clip(current_U + dcs_hist[:, k, 0], -4.0, 4.0)
    U_states[:, n_stage] = current_U
    all_U_states[agent_name] = U_states

# Physical state trajectories plot
plt.figure(figsize=(12, 6))
stages = np.arange(n_stage + 1)

for agent_name in colors:
    plt.plot(
        stages, 
        all_U_states[agent_name].T,  
        color=colors[agent_name], 
        linewidth=1.0, 
        alpha=0.015
    )

plt.axhline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
plt.xlim(0, n_stage)
plt.ylim(-4.1, 4.1)
plt.xticks(stages, [f"Stage {k}" for k in stages], fontsize=11)
plt.xlabel("Stage $k$", fontsize=12)
plt.ylabel("Physical State (Voltage $U_k$)", fontsize=12)
plt.title(f"Physical state trajectories ($U_k$) comparison at $\\sigma={noise_val}$", fontsize=14)
plt.grid(True, ls=":", alpha=0.5)
handles = [mlines.Line2D([], [], color=colors[name], label=name, linewidth=2.0) for name in colors]
plt.legend(handles=handles, loc="upper right", fontsize=11)
plt.tight_layout()
plt.savefig("semiconductor_design_trajectories_agents_comparison.png", dpi=300)
plt.show()

# Reward histogram
plt.figure(figsize=(8, 5))
bins_reward = np.linspace(-0.1, 2.0, 80)
for agent_name in colors:
    plt.hist(
        all_rewards[agent_name], 
        alpha=0.6, 
        bins=bins_reward, 
        color=colors[agent_name], 
        label=agent_name, 
        edgecolor='none'
    )

plt.xlabel('Reward', fontsize=11)
plt.ylabel('Counts', fontsize=11)
plt.title(f'Histogram of rewards comparison ($\\sigma={noise_val}$)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig("figure_rewards_histograms_agents_comparison.png", dpi=300)
plt.show()

# Metrics tables
print("\n+" + "-"*63 + "+")
print(f"| {'EVALUATION METRICS (10,000 TRAJECTORIES, sigma=0.05)':^61} |")
print("+" + "-"*15 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*16 + "+")
print(f"| {'Method':<13} | {'Mean Reward':<12} | {'Std Dev':<12} | {'Max Reward':<14} |")
print("+" + "-"*15 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*16 + "+")
for agent_name in colors:
    rw = all_rewards[agent_name]
    print(f"| {agent_name:<13} | {np.mean(rw):<12.6f} | {np.std(rw):<12.6f} | {np.max(rw):<14.6f} |")
print("+" + "-"*15 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*16 + "+\n")

# ============================================================================
# POSTERIOR MSE TABLE ACROSS STAGES
# ============================================================================
print("\n+" + "-"*92 + "+")
print(f"| {'POSTERIOR MEAN SQUARED ERROR (MSE) VS TRUE THETA':^90} |")
print("+" + "-"*15 + "+" + "-"*76 + "+")

header = f"| {'Method':<13} |"
for s in range(1, n_stage + 1):
    header += f" {'Stage ' + str(s):<7} |"
print(header)
print("+" + "-"*15 + "+" + "-"*76 + "+")

for agent_name in colors:
    row = f"| {agent_name:<13} |"
    for mse in all_mses[agent_name]:
        row += f" {mse:<7.4f} |"
    print(row)

print("+" + "-"*15 + "+" + "-"*76 + "+\n")