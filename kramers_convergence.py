import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import time

# Reproducibility

np.random.seed(2024)

# Simulation parameters
T   = 1           # time horizon
dt  = 1e-5          # step size  
N   = int(T / dt)   # steps
Np  = 1500          # Monte Carlo paths
eps = 0.5          # threshold for probability bound  
q0  = 1.0           # initial position
p0  = 0.5           # initial velocity 

# u grid 
mu_vals = np.array([ 0.6, 0.5, 0.4, 0.20, 0.12, 0.07, 0.04, 0.02, 0.01, 0.005
                    , 0.002, 0.001, 0.0005, 0.0002, 0.0001
                    ])

# ── Coefficient functions (Lipschitz + bounded)
#   b(t,q) = sin(t) − tanh(q)    (Lipschitz constant K=1, bounded by 2)
#   sigma (t,q) = 0.5           (constant, satisfies all assumptions trivially)
def b_fn(t, q): return np.sin(t) - np.tanh(q)
def sigma_fn(t, q): return 0.5 * np.ones(np.shape(q), dtype=float)

t_grid = np.arange(N + 1) * dt

# Pre-generate Brownian increments 
print(f"Generating {Np}×{N} Brownian increments … ", end="", flush=True)
dW_all = np.sqrt(dt) * np.random.randn(Np, N)
print("done.\n")

# Storage
det_max     = np.zeros(len(mu_vals))  
sto_maxEmn  = np.zeros(len(mu_vals))  
sto_EmaxSq  = np.zeros(len(mu_vals))  
sto_prob    = np.zeros(len(mu_vals))   
sto_Emax    = np.zeros(len(mu_vals))   

# Main simulation loop
for idx, mu in enumerate(mu_vals):
    t_start = time.time()
    print(f"  μ = {mu:.4f}", end="  |  ", flush=True)
    
    # (i) DETERMINISTIC 
    # Euler–Maruyama for both, no noise.
    qi = float(q0);  qm = float(q0);  vm = float(p0)
    det_e = 0.0
    for k in range(N):
        tk       = t_grid[k]
        bi_ito   = b_fn(tk, qi)
        bi_mu    = b_fn(tk, qm)
        
        qi_new   = qi + bi_ito * dt

        qm_new   = qm + vm * dt
        vm_new   = vm + (bi_mu - vm) / mu * dt
        qi, qm, vm = qi_new, qm_new, vm_new
        det_e    = max(det_e, abs(qm - qi))
    det_max[idx] = det_e

    # (ii–v) STOCHASTIC  (Monte Carlo over Np paths)
    # All paths share the SAME Brownian increments.

    qi = np.full(Np, q0, dtype=float)
    qm = np.full(Np, q0, dtype=float)
    vm = np.full(Np, p0, dtype=float)

    max_e_path = np.zeros(Np)   
    maxEmn_run = 0.0            

    for k in range(N):
        tk   = t_grid[k]
        dwk  = dW_all[:, k]      

        # Coefficients evaluated at CURRENT state (standard Euler–Maruyama)
        bi_ito  = b_fn(tk, qi)
        si_ito  = sigma_fn(tk, qi)
        bi_mu   = b_fn(tk, qm)
        si_mu   = sigma_fn(tk, qm)

        # Itô SDE update
        qi = qi + bi_ito * dt + si_ito * dwk

        # Kramers update  (q first with old v, then v — explicit Euler)
        qm_new = qm + vm * dt
        vm     = vm + (bi_mu - vm) / mu * dt + si_mu / mu * dwk
        qm     = qm_new

        # Pointwise error
        err          = np.abs(qm - qi)
        max_e_path   = np.maximum(max_e_path, err)
        maxEmn_run   = max(maxEmn_run, np.mean(err))

    sto_maxEmn[idx]  = maxEmn_run
    sto_EmaxSq[idx]  = np.mean(max_e_path**2)
    sto_prob[idx]    = np.mean(max_e_path > eps)
    sto_Emax[idx]    = np.mean(max_e_path)

    elapsed = time.time() - t_start
    print(f"det max={det_max[idx]:.4f}  "
          f"E[max|e|²]={sto_EmaxSq[idx]:.5f}  "
          f"ℙ(·>{eps})={sto_prob[idx]:.3f}  "
          f"({elapsed:.1f}s)")


def fit_slope(x, y):
    mask = y > 0
    sl, ic, r, *_ = linregress(np.log(x[mask]), np.log(y[mask]))
    return sl, np.exp(ic), r**2

def best_fit_line(x, alpha, y):
    """Return C·f(x) where f(x)=x^alpha, C fitted by least squares."""
    C = np.exp(np.mean(np.log(y) - alpha * np.log(x)))
    return C * x**alpha, C

sl_i,   C_i,   r2_i   = fit_slope(mu_vals, det_max)
sl_ii,  C_ii,  r2_ii  = fit_slope(mu_vals, sto_maxEmn)
sl_iii, C_iii, r2_iii = fit_slope(mu_vals, sto_EmaxSq)
sl_v,   C_v,   r2_v   = fit_slope(mu_vals, sto_Emax)

# For (iii) and (v)
ref_mulog  = mu_vals * np.log(1.0 / mu_vals)
ref_sqrtml = np.sqrt(ref_mulog)
C_iii_log  = np.exp(np.mean(np.log(sto_EmaxSq) - np.log(ref_mulog)))
C_v_log    = np.exp(np.mean(np.log(sto_Emax)   - np.log(ref_sqrtml)))


print("\n" + "="*68)
print(f"{'Statement':<12} {'Fitted slope':>14}  {'Theory':>12}  {'R²':>6}")
print("-"*68)
print(f"{'(i)  det':<12} {sl_i:>14.3f}  {'1.000':>12}  {r2_i:>6.4f}")
print(f"{'(ii) sto':<12} {sl_ii:>14.3f}  {'0.500':>12}  {r2_ii:>6.4f}")
print(f"{'(iii) sto':<12} {sl_iii:>14.3f}  {'≈1 + log':>12}  {r2_iii:>6.4f}")
print(f"{'(v)  sto':<12} {sl_v:>14.3f}  {'≈0.5+log':>12}  {r2_v:>6.4f}")
print("="*68)


COLORS  = ["#2E86AB", "#E84855", "#3BB273", "#F18F01", "#8338EC", "#023E8A"]
mu_fine = np.logspace(np.log10(mu_vals[-1]) - 0.1,
                      np.log10(mu_vals[0])  + 0.1, 200)

fig = plt.figure(figsize=(17, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

kw_data = dict(marker='o', linewidth=2.0, markersize=7,
               color=COLORS[0], label='Empirical', zorder=5)
kw_ref  = dict(linestyle='--', linewidth=2.0, color=COLORS[1], alpha=0.85, zorder=4)
kw_ref2 = dict(linestyle=':',  linewidth=2.0, color=COLORS[2], alpha=0.85, zorder=3)

# ─── (i)max |et| = O(u)
ax = fig.add_subplot(gs[0, 0])
ax.loglog(mu_vals, det_max, **kw_data)
ref1, _ = best_fit_line(mu_fine, 1.0,
          np.interp(mu_fine, mu_vals[::-1], det_max[::-1]))
ax.loglog(mu_fine, C_i * mu_fine, **kw_ref,
          label=f'$C\\mu$  (slope={sl_i:.2f}, $R^2$={r2_i:.4f})')
ax.set_xlabel('$\\mu$', fontsize=12)
ax.set_ylabel('$\\max_t\\,|e_t|$', fontsize=12)
ax.set_title('(i) Deterministic ($\\sigma=0$)\n'
             '$\\max_{{[0,T]}}|e_t| = O(\\mu)$', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.25)

# ─── (ii) max_t E[|et|] = O(sqrt{u})
ax = fig.add_subplot(gs[0, 1])
ax.loglog(mu_vals, sto_maxEmn, **kw_data)
ax.loglog(mu_fine, C_ii * np.sqrt(mu_fine), **kw_ref,
          label=f'$C\\sqrt{{\\mu}}$  (slope={sl_ii:.2f})')
ax.set_xlabel('$\\mu$', fontsize=12)
ax.set_ylabel('$\\max_t\\,\\mathbb{E}[|e_t|]$', fontsize=12)
ax.set_title('(ii) Stochastic (pointwise)\n'
             '$\\max_t\\,\\mathbb{E}[|e_t|] = O(\\sqrt{\\mu})$',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.25)

# ─── (iii) E[max_t |et|^2] = O(u log(1/u))
ax = fig.add_subplot(gs[0, 2])
ax.loglog(mu_vals, sto_EmaxSq, **kw_data)
ax.loglog(mu_fine, C_iii_log * mu_fine * np.log(1/mu_fine), **kw_ref,
          label=f'$C\\mu\\log(1/\\mu)$  (slope≈{sl_iii:.2f})')
ax.loglog(mu_fine, C_iii * mu_fine, **kw_ref2,
          label=f'$C\\mu$ (plain, slope=1)')
ax.set_xlabel('$\\mu$', fontsize=12)
ax.set_ylabel('$\\mathbb{E}[\\max_t\\,|e_t|^2]$', fontsize=12)
ax.set_title('(iii) Uniform $L^2$\n'
             '$\\mathbb{E}[\\max_t|e_t|^2] = O(\\mu\\log(1/\\mu))$',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.25)

# ─── (iv) Markov probability bound
ax = fig.add_subplot(gs[1, 0])
markov_bnd = sto_EmaxSq / eps**2
ax.loglog(mu_vals, sto_prob,   marker='o', linewidth=2, markersize=7,
          color=COLORS[0], label=f'$\\mathbb{{P}}(\\max|e_t|>{eps})$ empirical', zorder=5)
ax.loglog(mu_vals, markov_bnd, **kw_ref,
          label='Markov bound $= \\mathbb{E}[\\max|e|^2]/\\varepsilon^2$')
ax.set_xlabel('$\\mu$', fontsize=12)
ax.set_ylabel(f'Probability', fontsize=12)
ax.set_title(f'(iv) Markov Inequality Check ($\\varepsilon={eps}$)\n'
             f'$\\mathbb{{P}}(\\max_t|e_t|>\\varepsilon) \\leq '
             f'\\mathbb{{E}}[\\max|e|^2]/\\varepsilon^2$',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.25)

# ─── (v) 𝔼[max_t |et|] = O( sqrt(u log(1/u)) 
ax = fig.add_subplot(gs[1, 1])
ax.loglog(mu_vals, sto_Emax, **kw_data)
ax.loglog(mu_fine, C_v_log * np.sqrt(mu_fine * np.log(1/mu_fine)), **kw_ref,
          label=f'$C\\sqrt{{\\mu\\log(1/\\mu)}}$  (slope≈{sl_v:.2f})')
ax.loglog(mu_fine, C_v * np.sqrt(mu_fine), **kw_ref2,
          label=f'$C\\sqrt{{\\mu}}$ (plain, slope=0.5)')
ax.set_xlabel('$\\mu$', fontsize=12)
ax.set_ylabel('$\\mathbb{E}[\\max_t\\,|e_t|]$', fontsize=12)
ax.set_title('(v) $L^1$ Uniform (Jensen from (iii))\n'
             '$\\mathbb{E}[\\max_t|e_t|] = O(\\sqrt{\\mu\\log(1/\\mu)})$',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.25)


ax = fig.add_subplot(gs[1, 2])
labels_data = [
    (det_max,    '(i) $\\max|e_t|$, $\\sigma=0$',        's'),
    (sto_maxEmn, '(ii) $\\max_t\\mathbb{E}[|e_t|]$',     '^'),
    (sto_EmaxSq, '(iii) $\\mathbb{E}[\\max|e_t|^2]$',    'D'),
    (sto_Emax,   '(v) $\\mathbb{E}[\\max|e_t|]$',        'o'),
]
for (y, lbl, mk), col in zip(labels_data, COLORS):
    ax.loglog(mu_vals, y, marker=mk, linewidth=2, markersize=6,
              color=col, label=lbl)

anchor = 0.20
anchor_idx = 0
ax.loglog(mu_fine,
          det_max[anchor_idx] * (mu_fine/anchor)**1.0,
          'k-',  alpha=0.35, linewidth=1.4, label='slope 1')
ax.loglog(mu_fine,
          sto_Emax[anchor_idx] * (mu_fine/anchor)**0.5,
          'k--', alpha=0.35, linewidth=1.4, label='slope 1/2')

ax.set_xlabel('$\\mu$', fontsize=12)
ax.set_ylabel('Error metric', fontsize=12)
ax.set_title('All Rates — Summary', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, which='both', alpha=0.25)


fig.suptitle(
    'Smoluchowski–Kramers Approximation: Numerical Verification of All Convergence Rates\n'
    r'$b(t,q)=\sin(t)-tanh(q)$,  $\sigma=0.5$,  '
    f'$T={T}$,  $\\Delta t={dt}$,  $N_{{\\mathrm{{paths}}}}={Np}$,  '
    f'$\\varepsilon={eps}$',
    fontsize=13, y=1.01
)

plt.savefig('\your\local\path', dpi=300, bbox_inches='tight')

print("\nPlot saved to kramers_convergence.png")
