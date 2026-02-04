# From DDPM to DDIM — a “researcher’s notebook” (step-by-step, with proofs)

> Goal: Re-derive DDIM starting from DDPM, **without pulling formulas from nowhere**.  
> We will:  
> 1) write down DDPM, 2) identify the obstacle (slow sampling), 3) derive **jump posteriors** in DDPM,  
> 4) explain mathematically what “fresh noise” means and why it grows with big jumps,  
> 5) design a controlled-noise family that leads naturally to **DDIM**.

---

## 0. Setup and notation

We consider vectors in $\mathbb{R}^d$ with isotropic Gaussians. All covariance matrices are scalar multiples of $I$.

Define:
- Noise schedule: $\beta_t \in (0,1)$
- $\alpha_t = 1 - \beta_t$
- Cumulative product: $\bar\alpha_t = \prod_{i=1}^t \alpha_i$

---

## 1. DDPM forward process

### 1.1 Markov noising dynamics
DDPM defines a **forward Markov chain**:
$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{\alpha_t}\,x_{t-1}, \ (1-\alpha_t)I\big).
$$

Equivalently (reparameterization):
$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t,\quad \epsilon_t\sim\mathcal{N}(0,I).
$$

---

## 2. Key closed form: marginal $q(x_t \mid x_0)$

### 2.1 Claim
$$
q(x_t \mid x_0) = \mathcal{N}\Big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\Big).
$$

### 2.2 Proof (by induction)
Base case $t=1$:
- $\bar\alpha_1=\alpha_1$
- $q(x_1\mid x_0) = \mathcal{N}(\sqrt{\alpha_1}x_0, (1-\alpha_1)I)$ ✅

Inductive step: assume true for $t-1$.
We have:
- $x_{t-1} = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon$, with $\epsilon\sim\mathcal{N}(0,I)$
- Then:
$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$
Substitute $x_{t-1}$:
$$
x_t=\sqrt{\alpha_t\bar\alpha_{t-1}}x_0 + \sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\epsilon + \sqrt{1-\alpha_t}\,\epsilon_t
$$
Mean becomes $\sqrt{\alpha_t\bar\alpha_{t-1}}x_0 = \sqrt{\bar\alpha_t}x_0$ since $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$.

Variance (independent Gaussians add):
$$
\alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t)
= 1 - \alpha_t\bar\alpha_{t-1}
= 1-\bar\alpha_t.
$$
So:
$$
x_t\mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I). \quad \blacksquare
$$

### 2.3 Important reparameterization
From the marginal:
$$
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,I).
$$
This “single-noise-vector view” will become the intuition engine for DDIM.

---

## 3. DDPM reverse: why sampling is slow

### 3.1 Exact posterior for adjacent step
Using Bayes + Markov:
$$
q(x_{t-1}\mid x_t, x_0)\ \propto\ q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0),
$$
which is Gaussian with closed-form mean and variance.

### 3.2 Practical sampling
At generation time $x_0$ is unknown, so we train a network $\epsilon_\theta(x_t,t)$ and estimate:
$$
\hat x_0(x_t) = \frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

Then we sample step-by-step (often 1000 steps).  
**Obstacle**: too many steps.

---

## 4. Researcher question: “Can we jump from $t$ to $s$ directly?”

### 4.1 Yes — DDPM math already allows jump transitions
For any $t>s$, the forward chain implies:
$$
q(x_t\mid x_s)=\mathcal{N}\Big(\sqrt{\tfrac{\bar\alpha_t}{\bar\alpha_s}}\,x_s,\ \Big(1-\tfrac{\bar\alpha_t}{\bar\alpha_s}\Big)I\Big).
$$

**Derivation sketch**: Compose the linear Gaussians from $s+1$ to $t$. The mean multiplies by $\prod_{i=s+1}^t\sqrt{\alpha_i}$ and the variance accumulates to the stated form (same induction style as Section 2).

Define:
$$
r := \frac{\bar\alpha_t}{\bar\alpha_s}=\prod_{i=s+1}^t \alpha_i \in (0,1).
$$
Then:
$$
q(x_t\mid x_s)=\mathcal{N}\big(\sqrt{r}\,x_s,\ (1-r)I\big).
$$

### 4.2 Jump posterior exists: $q(x_s\mid x_t,x_0)$
By Bayes and Markov:
$$
q(x_s\mid x_t,x_0)\ \propto\ q(x_t\mid x_s)\,q(x_s\mid x_0).
$$

We already know:
- Prior:
$$
q(x_s\mid x_0)=\mathcal{N}\big(\sqrt{\bar\alpha_s}x_0,\ (1-\bar\alpha_s)I\big)
$$
- Likelihood:
$$
q(x_t\mid x_s)=\mathcal{N}\big(\sqrt{r}\,x_s,\ (1-r)I\big)
$$

So the posterior is Gaussian. Next we **derive its mean and variance**.

---

## 5. Proof: derive the exact jump posterior variance and mean

Because everything is isotropic, we can do 1D algebra and it applies per dimension.

### 5.1 Write likelihood as a Gaussian over $x_s$
Likelihood in standard form:
$$
x_t = \sqrt{r}\,x_s + \sqrt{1-r}\,u,\quad u\sim\mathcal{N}(0,1)
$$
So:
$$
q(x_t\mid x_s)\propto \exp\left(-\frac{(x_t-\sqrt{r}x_s)^2}{2(1-r)}\right)
$$
Treating this as a function of $x_s$, it is Gaussian with:
- precision (inverse variance): $\lambda_\text{like} = \frac{r}{1-r}$
- mean contribution: proportional to $\frac{\sqrt{r}}{1-r}x_t$

### 5.2 Prior over $x_s$
$$
q(x_s\mid x_0)\propto \exp\left(-\frac{(x_s-\sqrt{\bar\alpha_s}x_0)^2}{2(1-\bar\alpha_s)}\right)
$$
So:
- precision: $\lambda_\text{prior}=\frac{1}{1-\bar\alpha_s}$
- mean: $\sqrt{\bar\alpha_s}x_0$

### 5.3 Product of Gaussians → posterior precision and mean
Posterior precision:
$$
\lambda_\text{post}=\lambda_\text{prior}+\lambda_\text{like}
= \frac{1}{1-\bar\alpha_s} + \frac{r}{1-r}
$$
So posterior variance:
$$
\operatorname{Var}(x_s\mid x_t,x_0)=\frac{1}{\lambda_\text{post}}
= \left(\frac{1}{1-\bar\alpha_s} + \frac{r}{1-r}\right)^{-1}
$$

Now simplify. Put over common denominator:
$$
\frac{1}{1-\bar\alpha_s}+\frac{r}{1-r}
= \frac{1-r + r(1-\bar\alpha_s)}{(1-\bar\alpha_s)(1-r)}
= \frac{1-r\bar\alpha_s}{(1-\bar\alpha_s)(1-r)}
$$
But $r\bar\alpha_s=\bar\alpha_t$, so $1-r\bar\alpha_s = 1-\bar\alpha_t$. Therefore:
$$
\boxed{
\tilde\beta_{t\to s}
:= \operatorname{Var}(x_s\mid x_t,x_0)
= \frac{(1-\bar\alpha_s)(1-r)}{1-\bar\alpha_t}
= \frac{1-\bar\alpha_s}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right)
}
$$

This is the **DDPM jump posterior variance**.

Posterior mean: weighted by precisions:
$$
\mu_{s\mid t}=
\tilde\beta_{t\to s}\left(
\frac{\sqrt{\bar\alpha_s}}{1-\bar\alpha_s}x_0
+\frac{\sqrt{r}}{1-r}x_t
\right)
$$
You can expand and simplify to the common closed form:
$$
\boxed{
\mu_{s\mid t}(x_t,x_0)
=
\sqrt{\frac{\bar\alpha_t}{\bar\alpha_s}}\frac{1-\bar\alpha_s}{1-\bar\alpha_t}\,x_t
+
\frac{\bar\alpha_s-\bar\alpha_t}{\sqrt{\bar\alpha_s}(1-\bar\alpha_t)}\,x_0
}
$$

---

## 6. “Explained noise” vs “fresh uncertainty” (the core intuition, with math)

### 6.1 Strip off the known signal from $x_0$
Define:
$$
w := x_t - \sqrt{\bar\alpha_t}x_0,\qquad
y := x_s - \sqrt{\bar\alpha_s}x_0.
$$
From Section 2:
- $y\sim \mathcal{N}(0,(1-\bar\alpha_s)I)$
- $w\sim \mathcal{N}(0,(1-\bar\alpha_t)I)$

From the $t\mid s$ transition:
$$
x_t=\sqrt{r}x_s+\sqrt{1-r}u,\quad u\sim\mathcal{N}(0,I)
$$
Subtract the $x_0$ part (note $\sqrt{\bar\alpha_t}=\sqrt{r}\sqrt{\bar\alpha_s}$):
$$
w = \sqrt{r}y + \sqrt{1-r}u.
$$

Interpretation:
- $y$ is the “noise part” already present at time $s$
- $u$ is **new independent noise** injected between $s$ and $t$
- $w$ (observed from $x_t$ and $x_0$) mixes both

### 6.2 Conditional Gaussian decomposition
Because $y,w$ are jointly Gaussian, the conditional is:
$$
y \mid w = \underbrace{\mathbb{E}[y\mid w]}_{\text{explained by }w}
+ \underbrace{\text{residual}}_{\text{fresh uncertainty}}
$$
with:
$$
\mathbb{E}[y\mid w]=\frac{\operatorname{Cov}(y,w)}{\operatorname{Var}(w)}w
$$
Compute covariances (per dimension; isotropic extends to vectors):
- $\operatorname{Var}(y)=1-\bar\alpha_s$
- $\operatorname{Var}(w)=1-\bar\alpha_t$ (can be checked from $w=\sqrt{r}y+\sqrt{1-r}u$)
- $\operatorname{Cov}(y,w)=\sqrt{r}\operatorname{Var}(y)=\sqrt{r}(1-\bar\alpha_s)$

Thus:
$$
\mathbb{E}[y\mid w]=\frac{\sqrt{r}(1-\bar\alpha_s)}{1-\bar\alpha_t}w
$$
and residual variance:
$$
\operatorname{Var}(y\mid w)=\operatorname{Var}(y)-\frac{\operatorname{Cov}(y,w)^2}{\operatorname{Var}(w)}
=\tilde\beta_{t\to s}.
$$

Finally:
$$
x_s=\sqrt{\bar\alpha_s}x_0 + y
= \underbrace{\sqrt{\bar\alpha_s}x_0 + \mathbb{E}[y\mid w]}_{\text{determined by }x_t,x_0}
+ \underbrace{\sqrt{\tilde\beta_{t\to s}}z}_{\text{fresh uncertainty}},\quad z\sim\mathcal{N}(0,I).
$$

✅ This is the precise meaning of:

> “Part of the randomness is explained by what’s already in $x_t$; part remains fresh.”

---

## 7. Why big jumps inject a lot of fresh noise (math breakdown)

Recall:
$$
\tilde\beta_{t\to s}=\frac{1-\bar\alpha_s}{1-\bar\alpha_t}(1-r),
\quad r=\prod_{i=s+1}^t \alpha_i.
$$

As the gap grows, $r$ shrinks because you multiply more numbers $<1$:
- larger $(t-s)$ ⇒ smaller $r$ ⇒ larger $(1-r)$

Approximation for small $\beta$:
$$
\log r = \sum_{i=s+1}^t \log(1-\beta_i)\approx -\sum_{i=s+1}^t \beta_i
\quad\Rightarrow\quad
r\approx \exp\left(-\sum_{i=s+1}^t\beta_i\right)
$$
So large gaps make $r\to 0$, hence $(1-r)\to 1$.

In the high-noise regime (large $t$), typically $1-\bar\alpha_t\approx 1$, so:
$$
\tilde\beta_{t\to s}\approx 1-\bar\alpha_s
$$
which can be **O(1)**. That means the jump update includes:
$$
+\sqrt{\tilde\beta_{t\to s}}z
$$
with large magnitude noise. With only a few remaining steps, it’s hard to “correct” that much injected randomness → quality drops.

---

## 8. Researcher brainstorm: “Do we really need to inject all that fresh noise?”

### 8.1 Key insight
The marginal at time $s$ has total variance $(1-\bar\alpha_s)I$.  
But how that variance splits into:
- “noise shared with $x_t$” vs
- “freshly injected noise”
is a design choice **if we are willing to sample from a different reverse process**.

So let’s create a **family** of samplers that **keep the same marginal noise level** at time $s$ but **scale the fresh noise**.

---

## 9. DDIM: define adjustable fresh noise via $\eta$

### 9.1 Define scaled injection variance
Let:
$$
\sigma_{t\to s}^2 := \eta^2\,\tilde\beta_{t\to s},\quad \eta\in[0,1].
$$
- $\eta=1$: DDPM-like injection
- $\eta=0$: no new noise injected (deterministic)

### 9.2 Preserve the correct total variance at time $s$
We want $x_s$ to have the right scale: signal $\sqrt{\bar\alpha_s}\hat x_0$ plus total noise $\sqrt{1-\bar\alpha_s}$.

Write an update form:
$$
x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + A\,\hat\varepsilon + \sigma_{t\to s} z
$$
We require the **noise variance** from the last two terms to match $(1-\bar\alpha_s)$:
$$
A^2 + \sigma_{t\to s}^2 = 1-\bar\alpha_s
\quad\Rightarrow\quad
A = \sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}.
$$

This yields the DDIM jump update:
$$
\boxed{
x_s
=
\sqrt{\bar\alpha_s}\,\hat x_0
+
\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\,\hat\varepsilon
+
\sigma_{t\to s}\,z,\quad z\sim\mathcal{N}(0,I)
}
$$

### 9.3 Plug in $\hat x_0$ and $\hat\varepsilon$ from the trained model
Given $x_t$, compute:
$$
\hat\varepsilon = \varepsilon_\theta(x_t,t),
\qquad
\hat x_0 = \frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\varepsilon}{\sqrt{\bar\alpha_t}}.
$$

### 9.4 Final closed form for $\sigma_{t\to s}^2$
Using Section 5:
$$
\tilde\beta_{t\to s}
= \frac{1-\bar\alpha_s}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right)
$$
so:
$$
\boxed{
\sigma_{t\to s}^2
=
\eta^2\,
\frac{1-\bar\alpha_s}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right)
}
$$

---

## 10. Special cases (sanity checks)

### 10.1 $\eta=1$ (DDPM-like stochasticity)
$$
\sigma_{t\to s}^2 = \tilde\beta_{t\to s}
$$
This recovers the DDPM-style amount of injected noise for the jump.

### 10.2 $\eta=0$ (deterministic DDIM)
$$
\sigma_{t\to s}=0
\Rightarrow
x_s=\sqrt{\bar\alpha_s}\hat x_0 + \sqrt{1-\bar\alpha_s}\hat\varepsilon
$$
No new random $z$ at intermediate steps. All randomness comes from the initial $x_T$.

### 10.3 Adjacent step $s=t-1$
The jump variance reduces to the familiar DDPM posterior variance for one step.

---

## 11. Full DDIM sampling algorithm (ready-to-implement pseudocode)

Choose a timestep subset:
$$
T=t_K>t_{K-1}>\dots>t_0=0
$$
(e.g., 50 steps instead of 1000).

```python
# DDIM sampler (discrete schedule)
# Inputs: eps_model(x, t) -> predicted noise
#         alphas, alpha_bars, timesteps = [tK, ..., t0]
#         eta in [0, 1]
# Output: x0 sample

x = sample_standard_normal(shape)  # x_T

for idx in range(len(timesteps) - 1):
    t = timesteps[idx]
    s = timesteps[idx + 1]  # s < t

    abar_t = alpha_bars[t]
    abar_s = alpha_bars[s]

    eps_hat = eps_model(x, t)

    x0_hat = (x - (1 - abar_t)**0.5 * eps_hat) / (abar_t**0.5)

    # DDIM sigma^2 for jump t -> s
    sigma2 = (eta**2) * (1 - abar_s) / (1 - abar_t) * (1 - abar_t / abar_s)
    sigma = sigma2**0.5

    # coefficient to keep correct total variance at time s
    c = (1 - abar_s - sigma2)**0.5

    z = sample_standard_normal(shape) if eta > 0 else 0.0

    x = (abar_s**0.5) * x0_hat + c * eps_hat + sigma * z

return x  # x_0
