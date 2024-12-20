# MPC for Planning

This repository provides an example of MPC optimiser for planning. One should be aware that this is just one of the implementations of solving an MPC problem, and there also remains some improvement. 

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
A_{ineq}
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
\right]$$
where 
- $x_k$ is state variable; 
- $x_{k, r}$ is state reference; 
- $u_k$ is control variable; 
- $Q_N$ is usually solved by an algebraic discrete Riccati equation.

# Longitudinal MPC

In this section, we will modify $J$ to fit for a speed planning problem. 

$$J = 
\min_{x_k, u_k} \sum_{k = 0} ^ {N-1} \left[\left(x_k - x_{k,r}\right)^T Q \left(x_k - x_{k,r}\right) + u_k^T R u_k + \dot {u}_k^T \dot{R} \dot{u}_k\right] + \\
\left(x_N - x_{N,r}\right)^T Q_N \left(x_N - x_{N,r}\right) + \sigma_k^T W \sigma_k$$ 

where, specifically, in this equation, 
- $x_k$ represents $[s_k, \dot{s_k}]^T$;
- $x_{k, r}$ represents $[s_{k, r}, \dot{s}_{k, r}]^T$;
- $u_k$ represents $\ddot{s}_k$;
- $\sigma_k$ represents slack variables.

## To a Quadratic Problem

We have a standard QP formulation as below.

$$J = \min_z \frac{1}{2}z^T P z + q^T z \\ 
s.t. \\
z_{min} \leq Az \leq z_{max}$$

### Cost

First of all, $\dot{u}_k$ could be written as 
$$\dot{u}_k = \frac{u_k - u_{k-1}}{\Delta t}$$
where $k = 1, ..., N - 1$.

Perticular,
$$\dot{u}_0 = \frac{u_0 - u_{-1}}{\Delta t_{-1}}$$
where $-1$ index means the previous control command. For simplicity, we choose $\Delta t_{-1}$ equals $\Delta t$.

Thus, the control part is 
$$\begin{aligned}
u_k^T R u_k + \dot {u}_k^T \dot{R} \dot{u}_k 
&= u_k^T R u_k + \frac{(u_k - u_{k-1})^T}{\Delta t} \dot R \frac{u_k - u_{k-1}}{\Delta t}\\
&= u_k^T(R + \frac{\dot R}{\Delta t^2})u_k + u_{k-1}^T\frac{\dot R}{\Delta t^2}u_{k-1}\\
& - u_k^T \frac{\dot R}{\Delta t^2} u_{k-1} - u_{k-1}^T \frac{\dot R}{\Delta t^2} u_{k}
\end{aligned}$$

Then, we have
$$P = \left[\begin{matrix}
Q \\
& \ddots\\
& & Q \\
& & & Q_N\\
& & & & R + 2\frac{\dot R}{\Delta t^2} & -\frac{\dot R}{\Delta t^2}\\
& & & & -\frac{\dot R}{\Delta t^2} & \ddots &  \ddots\\
& & & & & \ddots & R + 2\frac{\dot R}{\Delta t^2} & -\frac{\dot R}{\Delta t^2}\\
& & & & & &  -\frac{\dot R}{\Delta t^2} & R + \frac{\dot R}{\Delta t^2}\\
& & & & & & & & w\\
& & & & & & & & & \ddots\\
& & & & & & & & & & w
\end{matrix}
\right]$$

$$q = \left[\begin{matrix}
-Qx_{0, ref} & \dots  &-Q_N x_{N, ref} & 
-\frac{\dot R}{\Delta t^2}u_{-1} & 0 & 
0
\end{matrix}
\right]^T$$

### Constraints

#### Equality Constraints

We consider an error differential equation
$$\dot x(t) = A_c x(t) +B_c u(t)$$
where $x(t)$ is state variable; $u(t)$ is control variable.

Here, we have
$$
x_k = \left[s_k, \dot s_k\right]^T \\
u_k = [\ddot s_k]
$$

Then,
$$
\begin{aligned}
A_c &= \left[
\begin{matrix}
0 & 1 \\
0 & 0
\end{matrix}
\right] \\
B_c &= \left[
\begin{matrix}
0 \\
1
\end{matrix}
\right]
\end{aligned}
$$

Thus, the equality constraints 
$$x_{k + 1} = (I + A_c \Delta t) x_k + B_c \Delta t u_k$$ 
could be written in the following format
$$
b_{eq} \leq A_{eq}z \leq b_{eq}
$$
where
$$
A_{eq} = \left[
\begin{matrix}
-I & & & & 0 \\
I + A_c \Delta t & -I & & & B_c  \Delta t \\
& \ddots & \ddots & & & \ddots & \ddots\\
& & I + A_c \Delta t & -I & & & B_c  \Delta t & 0
\end{matrix}
\right]
$$

$$
b_{eq} = \left[
\begin{matrix}
-x_0 \\
0 \\
\vdots \\
0
\end{matrix}
\right]
$$

####ß Inequality Constraints

Generally, maximum ranges of acceleration and jerk of the ego vehicle constrain a speed planning problem. Slack variables are often applied to relax station and velocity constraints. In this section, we only support to relax $s_k$ and $\dot s_k$ with slack variables, denoting as $\sigma_{x,k}$ for upper bounds and $\mu_{x, k}$ for lower bounds.

$$
\begin{aligned}
A_{ineq} &= 
\left[
\begin{matrix}
1_{x, k} &  & -1_{\sigma_{x,k}} \\
1_{x,k} & & & 1_{\mu_{x, k}} \\
& 1_{u, k} \\
& a_{\dot u, k} \\
& & 1_{\sigma_{x,k}}
\end{matrix}
\right] \\
b_{ineq, lower} &= \left[
\begin{matrix}
-\infty \\
x_{min, k} \\
u_{min, k} \\
b_{\dot u, k, min} \\
0_{\sigma_{x, k}}
\end{matrix}
\right] \\ 
b_{ineq, upper} &= \left[
\begin{matrix}
x_{max, k} \\
+\infty \\
u_{max, k} \\
b_{\dot u, k, max} \\
\sigma_{max, x, k}
\end{matrix}
\right]
\end{aligned}
$$

where 
$$\begin{aligned}
a_{\dot u, k} &= \left[
\begin{matrix}
1 \\
-1 & 1 \\
& \ddots & \ddots \\
& & -1 & 1
\end{matrix}
\right] \\
b_{\dot u, k, max} &= 
\left[
\begin{matrix}
u_{-1} + j_{max}\Delta t \\
j_{max}\Delta t \\ 
\vdots \\ 
j_{max}\Delta t
\end{matrix}
\right] \\
b_{\dot u, k, min} &= 
\left[
\begin{matrix}
u_{-1} + j_{min}\Delta t \\
j_{min}\Delta t \\ 
\vdots \\ 
j_{min}\Delta t
\end{matrix}
\right]
\end{aligned}
$$

Please Note that we do **NOT** support to constrain lower slack variables $\mu_{x, k}$ of $x_k$.

# Lateral MPC

Similar to Longitudinal MPC, we start Lateral MPC with its state transition equation. With vehicle dynamics, we have

$$
\dot{x}(t) = A_cx(t) + B_cu(t) + \tilde{B}_c\tilde{u}(t)
$$

where 
$$\begin{aligned}
x(0) &= x_{init} \\
A_c &= \left[
\begin{matrix}
0 & 1 & 0 & 0 \\
0 & -\frac{C_{\alpha f} + C_{\alpha r}}{mv_t} & \frac{C_{\alpha f} + C_{\alpha r}}{m} & \frac{-C_{\alpha f}l_f + C_{\alpha r}l_r}{mv_t} \\
0 & 0 & 0 & 1 \\
0 & -\frac{C_{\alpha f}l_f - C_{\alpha r}l_r}{I_zv_t} & \frac{C_{\alpha f}l_f - C_{\alpha r}l_r}{I_z} & -\frac{C_{\alpha f}l_f^2 + C_{\alpha r}l_r^2}{I_zv_t}
\end{matrix}
\right] \\
B_c &= \left[
\begin{matrix}
0 \\
\frac{C_{\alpha f}}{m} \\
0 \\
\frac{C_{\alpha f}l_f}{I_z}
\end{matrix}
\right] \\
\tilde{B}_c &= \left[
\begin{matrix}
0 \\
-\frac{C_{\alpha f}l_f - C_{\alpha r}l_r}{mv_t} - v_t \\
0 \\
-\frac{C_{\alpha f}l_f^2 - C_{\alpha r}l_r^2}{I_zv_t}
\end{matrix}
\right]
\end{aligned} 
$$

To discretise the problem, two-step Euler method is applied on $x_t$ while forward Euler method on $u_t$. Then we have,

$$
\begin{aligned}
x_{t + 1}  &= \left(I + A_cx_tdt\right) + B_cu_tdt + \tilde{B}_c\tilde{u}_tdt \\
&= \left(I + A_cdt\right)\frac{x_{t+1} + x_t}{2} + B_cu_tdt + \tilde{B}_c\tilde{u}_tdt \\
&= \left(2I - A_c dt\right)^{-1}\left(2I+ A_c dt\right)x_t + \left(2I - A_c dt\right)^{-1}B_c u_t dt + \left(2I - A_c dt\right)^{-1}\tilde{B}_c\tilde{u}_t dt
\end{aligned}
$$

where 
$$
\begin{aligned}
x_k &= \left[
\begin{matrix}
l_k \\
\dot{l}_k \\
\psi_{s, k} \\
\dot \psi_{s, k}
\end{matrix}
\right] \\
u_k &= \left[\delta_k\right] \\
\dot u_k &= \frac{u_k - u_{k - 1}}{\Delta t}\\
\tilde{u}_k &= \left[\kappa_{s, k}v_k\right]
\end{aligned}
$$
