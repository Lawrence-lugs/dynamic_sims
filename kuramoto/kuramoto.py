import os, math, json, itertools, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

OUTDIR = Path("./outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

def simulate_kuramoto_fast(L=50, T=20.0, dt=0.02, K_mean=1.0, K_std=0.0, omega_mean=0.0, omega_std=0.5,
                           init_phase_std=None, seed=None, periodic=True, record_snapshots=False, snapshot_dt=0.5):
    rng = np.random.default_rng(seed)
    N = L*L
    omega = rng.normal(loc=omega_mean, scale=omega_std, size=(L,L))
    
    if init_phase_std is None:
        theta = rng.uniform(-np.pi, np.pi, size=(L,L))
    else:
        theta = rng.normal(loc=0.0, scale=init_phase_std, size=(L,L))

    # per-edge coupling fields (for right and down edges)
    K_right = rng.normal(loc=K_mean, scale=K_std, size=(L,L))
    K_down = rng.normal(loc=K_mean, scale=K_std, size=(L,L))
    
    # integration
    times = np.arange(0, T+1e-12, dt)
    r_ts = np.zeros(times.shape)
    var_ts = np.zeros(times.shape)
    psi_ts = np.zeros(times.shape)
    snapshots = []
    snapshot_times = []
    next_snapshot_t = 0.0
    
    for ti, t in enumerate(times):
        z = np.exp(1j*theta).mean()
        r_ts[ti] = np.abs(z)
        psi_ts[ti] = np.angle(z)
        var_ts[ti] = np.var(np.angle(np.exp(1j*theta)))
        if record_snapshots and t >= next_snapshot_t - 1e-12:
            snapshots.append(theta.copy())
            snapshot_times.append(t)
            next_snapshot_t += snapshot_dt
        if ti < len(times)-1:
            # compute coupling term using neighbor rolls
            # right neighbor
            th_right = np.roll(theta, shift=-1, axis=1) if periodic else np.pad(theta[:,1:], ((0,0),(0,1)), mode='constant')
            # left neighbor
            th_left = np.roll(theta, shift=1, axis=1) if periodic else np.pad(theta[:,:-1], ((0,0),(1,0)), mode='constant')
            # down neighbor
            th_down = np.roll(theta, shift=-1, axis=0) if periodic else np.pad(theta[1:,:], ((0,1),(0,0)), mode='constant')
            # up neighbor
            th_up = np.roll(theta, shift=1, axis=0) if periodic else np.pad(theta[:-1,:], ((1,0),(0,0)), mode='constant')
            # coupling contributions (use K_right and K_down appropriately)
            # For interior use K_right for coupling from (i,j) to (i,j+1) and its left counterpart via roll
            # Sum contributions from 4 neighbors; note symmetric treatment ensures same as undirected edges
            # Use vectorized sin differences
            # right contribution to dtheta: K_right * sin(theta_right - theta)
            c_right = K_right * np.sin(th_right - theta)
            # left contribution: need K_right rolled left for the edge to left neighbor (edge stored at left cell)
            K_left = np.roll(K_right, shift=1, axis=1)
            c_left = K_left * np.sin(th_left - theta)
            c_down = K_down * np.sin(th_down - theta)
            K_up = np.roll(K_down, shift=1, axis=0)
            c_up = K_up * np.sin(th_up - theta)
            coupling = c_right + c_left + c_down + c_up
            # deterministic Kuramoto : dtheta/dt = omega + coupling
            # integrate with RK4 (vectorized)
            def rhs(th):
                # compute neighbor terms for th
                tr = np.roll(th, -1, axis=1) if periodic else np.pad(th[:,1:], ((0,0),(0,1)), mode='constant')
                tl = np.roll(th, 1, axis=1) if periodic else np.pad(th[:,:-1], ((0,0),(1,0)), mode='constant')
                td = np.roll(th, -1, axis=0) if periodic else np.pad(th[1:,:], ((0,1),(0,0)), mode='constant')
                tu = np.roll(th, 1, axis=0) if periodic else np.pad(th[:-1,:], ((1,0),(0,0)), mode='constant')
                cr = K_right * np.sin(tr - th)
                cl = np.roll(K_right, 1, axis=1) * np.sin(tl - th)
                cd = K_down * np.sin(td - th)
                cu = np.roll(K_down, 1, axis=0) * np.sin(tu - th)
                return omega + (cr + cl + cd + cu)
            k1 = rhs(theta)
            k2 = rhs(theta + 0.5*dt*k1)
            k3 = rhs(theta + 0.5*dt*k2)
            k4 = rhs(theta + dt*k3)
            theta = theta + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
            theta = (theta + np.pi) % (2*np.pi) - np.pi
    result = {
        "L": L, "T": T, "dt": dt, "K_mean": K_mean, "K_std": K_std,
        "omega_mean": omega_mean, "omega_std": omega_std, "init_phase_std": init_phase_std,
        "times": times, "r_ts": r_ts, "psi_ts": psi_ts, "var_ts": var_ts, "final_theta": theta,
        "snapshots": snapshots, "snapshot_times": snapshot_times,
        "K_right": K_right, "K_down": K_down, "omega": omega
    }
    return result

# Run a faster demo (smaller L to finish quickly)
print("Running optimized demo...")
demo_res = simulate_kuramoto_fast(L=40, T=30.0, dt=0.03, K_mean=1.0, K_std=0.05, omega_mean=0.0, omega_std=0.6, seed=2, record_snapshots=True, snapshot_dt=0.5)
print("Demo done. Saving plots...")

# Plot r(t)
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(demo_res["times"], demo_res["r_ts"], label="r(t)")
ax1.plot(demo_res["times"], demo_res["var_ts"]/np.max(demo_res["var_ts"]) * np.max(demo_res["r_ts"]), label="scaled var")
ax1.set_xlabel("Time"); ax1.set_ylabel("r(t)")
ax1.legend()
fig1_path = OUTDIR / "demo_order_variance_vec.png"
fig1.savefig(fig1_path, dpi=150)

# Save three snapshots
snapshots = demo_res["snapshots"]
fig2_path = None
if len(snapshots) >= 3:
    fig2, axes2 = plt.subplots(1,3, figsize=(12,4))
    for ax, s, t in zip(axes2, [snapshots[0], snapshots[len(snapshots)//2], snapshots[-1]], [demo_res["snapshot_times"][0], demo_res["snapshot_times"][len(snapshots)//2], demo_res["snapshot_times"][-1]]):
        im = ax.imshow(s, origin="lower", vmin=-np.pi, vmax=np.pi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t:.2f}")
    fig2.tight_layout()
    fig2_path = OUTDIR / "demo_snapshots_three_vec.png"
    fig2.savefig(fig2_path, dpi=150)

# Parameter sweep (smaller to save time)
print("Running smaller parameter sweep...")
K_vals = [0.2, 0.6, 1.0, 1.6]
omega_std_vals = [0.0, 0.3, 0.6, 1.0]
sweep_summary = []
for K, ws in itertools.product(K_vals, omega_std_vals):
    res = simulate_kuramoto_fast(L=24, T=20.0, dt=0.03, K_mean=K, K_std=0.0, omega_mean=0.0, omega_std=ws, seed=1, record_snapshots=False)
    sweep_summary.append({"K":K, "omega_std":ws, "final_r": float(res["r_ts"][-1]), "mean_r": float(res["r_ts"].mean())})

heatmap_final_r = np.array([[ next(item["final_r"] for item in sweep_summary if item["K"]==K and item["omega_std"]==ws)
                              for ws in omega_std_vals] for K in K_vals])

fig3, ax3 = plt.subplots(figsize=(6,5))
im3 = ax3.imshow(heatmap_final_r, origin="lower", aspect="auto", extent=[min(omega_std_vals), max(omega_std_vals), min(K_vals), max(K_vals)])
ax3.set_xlabel("omega_std"); ax3.set_ylabel("K_mean"); ax3.set_title("Final r(T)")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
fig3_path = OUTDIR / "sweep_heatmap_final_r_vec.png"
fig3.savefig(fig3_path, dpi=150)

with open(OUTDIR / "sweep_summary.json","w") as f:
    json.dump(sweep_summary, f, indent=2)

from IPython.display import Image, display
print("Saved plots to:", OUTDIR)
display(Image(str(fig1_path)))
if fig2_path is not None:
    display(Image(str(fig2_path)))
display(Image(str(fig3_path)))

readme = f"""Kuramoto 2D Grid Simulation Project (vectorized)
Files produced in {OUTDIR}:
- demo_order_variance_vec.png : r(t) and scaled variance for demo run
- demo_snapshots_three_vec.png : three phase field snapshots
- sweep_heatmap_final_r_vec.png : heatmap of final order parameter r(T) for a (K_mean, omega_std) sweep
- sweep_summary.json : numeric summary of the sweep results

Tips:
- Increase L to explore finite-size effects.
- Set K_std>0 to introduce heterogeneity in couplings.
- Add noise: in rhs, add + sigma_noise * rng.normal(size=(L,L))/sqrt(dt) each step.
"""
with open(OUTDIR / "README_vec.txt","w") as f:
    f.write(readme)

print("Done.")
