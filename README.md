# Longitudinal MPC

This repository implement speed planning via MPC.

# Basic MPC

$$J = 
\min_{x_k, u_k} \sum_{k = 0} ^ {N-1} \left[\left(x_k - x_{k,r}\right)^T Q \left(x_k - x_{k,r}\right) + u_k^T R u_k\right] + \\
\left(x_N - x_{N,r}\right)^T Q_N \left(x_N - x_{N,r}\right)
$$

$$s.t. \\
x_{k + 1} = Ax_k + Bu_k \\
\left[
\begin{matrix}
x_{min, k}  \\
u_{min, k}
\end{matrix}
\right] 
\leq
\left[
\begin{matrix}
x_k  \\
u_k
\end{matrix}
\right]
\leq
\left[
\begin{matrix}
x_{max, k}  \\
u_{max, k}
\end{matrix}
\right] 
$$
where 
- $x_k$ is state variable; 
- $x_{k, r}$ is state reference; 
- $u_k$ is control variable; 
- $Q_N$ is usually solved by an algebraic discrete Riccati equation.

# Longitudinal MPC

In this section, we will modify $J$ to fit for a speed planning problem. 

$$J = 
\min_{x_k, u_k} \sum_{k = 0} ^ {N-1} \left[\left(x_k - x_{k,r}\right)^T Q \left(x_k - x_{k,r}\right) + u_k^T R u_k + \dot {u}_k^T \dot{R} \dot{u}_k\right] + \\
\left(x_N - x_{N,r}\right)^T Q_N \left(x_N - x_{N,r}\right) + \sigma_k^T W \sigma_k
$$ 

where, specifically, in this equation, 
- $x_k$ represents $[s_k, \dot{s_k}]^T$;
- $x_{k, r}$ represents $[s_{k, r}, \dot{s}_{k, r}]^T$;
- $u_k$ represents $\ddot{s}_k$;
- $\sigma_k$ represents slack variables.

# To a Quadratic Problem

We have a standard QP formulation as below.

$$
J = \min_z \frac{1}{2}z^T P z + q^T z \\ 
s.t. \\
z_{min} \leq Az \leq z_{max}
$$

## Cost

First of all, $\dot{u}_k$ could be written as 
$$\dot{u}_k = \frac{u_k - u_{k-1}}{\Delta t}
$$
where $k = 1, ..., N - 1$.

Perticular,
$$\dot{u}_0 = \frac{u_0 - u_{-1}}{\Delta t_{-1}}
$$
where $-1$ index means the previous control command. For simplicity, we choose $\Delta t_{-1}$ equals $\Delta t$.

Thus, the control part is 
$$\begin{aligned}
u_k^T R u_k + \dot {u}_k^T \dot{R} \dot{u}_k 
&= u_k^T R u_k + \frac{(u_k - u_{k-1})^T}{\Delta t} \dot R \frac{u_k - u_{k-1}}{\Delta t} \\
&= u_k^T(R + \frac{\dot R}{\Delta t^2})u_k + u_{k-1}^T\frac{\dot R}{\Delta t^2}u_{k-1}  \\
& - u_k^T \frac{\dot R}{\Delta t^2} u_{k-1} - u_{k-1}^T \frac{\dot R}{\Delta t^2} u_{k}
\end{aligned}
$$

Then, we have
$$Q = \left[\begin{matrix}
Q & & & \\
& \ddots \\
& & Q & \\
& & & Q_N \\
& & & & R + \frac{\dot R}{\Delta t^2} & -\frac{\dot R}{\Delta t^2} \\
& & & & -\frac{\dot R}{\Delta t^2} & \ddots &  \ddots \\
& & & & & \ddots & R + 2\frac{\dot R}{\Delta t^2} & -\frac{\dot R}{\Delta t^2}  \\
& & & & & &  -\frac{\dot R}{\Delta t^2} & R + \frac{\dot R}{\Delta t^2} \\
& & & & & & & & w \\
& & & & & & & & & \ddots \\
& & & & & & & & & & w
\end{matrix}
\right]
$$

$$
q = \left[\begin{matrix}
-Qx_{0, ref} & \dots  &-Q_N x_{N, ref} & 
-\frac{\dot R}{\Delta t^2}u_{-1} & 0 & 
0
\end{matrix}
\right]^T
$$

## Constraints

### Equality Constraints

### Inequality Constraints