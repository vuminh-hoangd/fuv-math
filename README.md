# Research seminar - undergrad FUV

A proof of the Smoluchowski–Kramers Approximation

## Convergence of Second-Order to First-Order Stochastic Dynamics

Consider the second-order stochastic differential equation arising from Newton's second law:

$$
\mu \ddot{q}_t^\mu = b(t, q_t^\mu) + \sigma(t, q_t^\mu)\dot{W}_t - \dot{q}_t^\mu,
\quad q_0^\mu = q,\quad \dot{q}_0^\mu = p,
\tag{1}
$$

and the associated first-order Itô stochastic differential equation:

$$
dq_t = b(t, q_t) \, dt + \sigma(t, q_t) \, dW_t, \quad q_0 = q.
\tag{2}
$$

Under **Lipschitz continuity** and **uniform boundedness** assumptions on $b$ and $\sigma$, we have the following Convergence Result (Conjecture 3.1, Part iii)

For any $\epsilon > 0$ and $T > 0$, there exists a constant $C > 0$ such that:

$$
\mathbb{P}\left( \max_{0 \leq t \leq T} |q_t^\mu - q_t| > \epsilon \right) =
O(\mu) \xrightarrow{\mu \to 0} 0
$$

---
