import numpy as np
from model import StateSpaceModel
from utils import forward_operator, mh_sample_tau_gamma, proximal_operator, get_gradient_Q
import time

# Global variables (adjust as needed)
N = 10         # Number of time steps
nvis = 351     # Number of visibilities
npixel2 = 4096 # Number of pixels in the latent space

def run_em_algorithm(visibilities, dirty, FOp, model_images, tau, n_iterations=1):
    """
    Run the EM algorithm.
    Parameters:
      - visibilities, dirty, FOp, model_images: data inputs 
      - tau: initial tau parameter (or scaling)
    """
    # Create the state-space instance.
    model = StateSpaceModel(
        m0=dirty,            # initial state
        S0=np.eye(npixel2)*1e-3,
        A=np.eye(npixel2),    # A is not assumed to be identity in general
        Gamma=np.eye(npixel2)*1e-4, 
        C=FOp,
        Sigma=tau*np.eye(nvis)
    )
    

    X = visibilities  # observed data
    # Initialize the model parameters
    
    smoothed_estimates = []
    MMSE = []
    tau_old = 0.9 * np.ones(N)

    for i in range(n_iterations):
        print(f"Iteration {i+1} of {n_iterations}")
        # (A) E-step: compute sufficient statistics
        E1, E2, E3, E4, mu_hat, V_hat, tau_old = compute_E_step(X, dirty, model, tau_old)

        # Compute MMSE based on the real parts of the smoothed estimate.
        MMSE_val = np.sum((np.square(np.subtract(np.real(mu_hat), model_images)))) / npixel2
        MMSE.append(MMSE_val)
        print("====== MMSE (smoothed estimates) ========= {}".format(MMSE[-1]))
        smoothed_estimates.append(mu_hat)

        # (B) M-step: update model parameters (using the method defined in your model)
        model.update_parameters(E1, E2, E3, E4, nvis, npixel2, tau)
        print(f"Iteration {i+1} completed.")

    return model, smoothed_estimates, MMSE

def compute_E_step(X, dirty, model, tau_old):
    """
    Performs the E-step computations.
    X: observed data (an array-like of length N)
    """
    gamma = 1e-5        # Step size for the proximal update (tunable)
    mh_burn_in = 100
    nu = 2.5            # Parameter for gradient and sampling

    t0 = time.time()
    # --- (A) Forward pass (Kalman Filter) ---
    V = get_V(tau_old, model)
    mu = get_mu(V, tau_old, model, X)
    print('KF run, took {:.2f}s'.format(time.time() - t0))
    t0 = time.time()
    
    # --- (B) Backward pass (RTS Smoother) ---
    V_hat = get_V_hat(V, model)
    mu_hat = get_mu_hat(mu, V_hat, model)
    print('KS run, took {:.2f}s'.format(time.time() - t0))
    t0 = time.time()

    # --- Regularization via Proximal Update ---
    grad_Q = get_gradient_Q(mu_hat, nu, dirty, model.A, model.C, model.Gamma, X)
    z = mu_hat + gamma * grad_Q
    mu_hat = proximal_operator(z, 10e-2)

    # --- (C) Sample tau_t (using Metropolis-Hastings) ---
    tau_temp = tau_old[0]  # current value for first time step
    for i in range(mh_burn_in):
        tau_temp = mh_sample_tau_gamma(
            current_tau=tau_temp,
            y=X[0],
            x=mu_hat[0],
            C=model.C,
            Sigma=model.Sigma,
            nu=2.5,
            shape_prop=2.5, 
            scale_prop=(tau_temp / 2.5)
        )
    
    for t in range(N):
        tau_temp = mh_sample_tau_gamma(
            current_tau=tau_temp,
            y=X[t],
            x=mu_hat[t],
            C=model.C,
            Sigma=model.Sigma,
            nu=2.5,
            shape_prop=2.5, 
            scale_prop=(tau_temp / 2.5)
        )
        tau_old[t] = tau_temp

    print("Sampling run, took {:.2f}s".format(time.time() - t0))
    t0 = time.time() 

    # --- (D) Compute E-step sufficient statistics ---
    E1 = get_E1(mu_hat)
    E2, E3 = get_E2_E3(V, mu_hat, V_hat, model)
    E4 = get_E4(mu_hat, V_hat)
    
    return E1, E2, E3, E4, mu_hat, V_hat, tau_old

# --- Helper Functions ---
# These functions implement the same mathematical operations as in the original code.

def P(V_n, model):
    # Compute P = A*V_n*A^T + Gamma.
    return model.A @ V_n @ model.A.T + model.Gamma

def K(P_n, tau_n, model):
    # Kalman gain: K_n = P_n * C^H * [C * P_n * C^H + (1/tau_n)*Sigma]^-1.
    num = P_n @ model.C.T.conj()
    denom = model.C @ P_n @ model.C.T.conj() + (1.0/tau_n) * model.Sigma
    return num @ np.linalg.inv(denom)

def get_V(tau, model):
    # Preallocate Vres with shape (N, npixel2, npixel2)
    I = np.eye(npixel2)
    Vres = np.empty((N, npixel2, npixel2), dtype=model.S0.dtype)
    K0 = K(model.S0, tau[0], model)
    Vres[0] = (I - K0 @ model.C) @ model.S0
    for n in range(1, N):
        P_prev = P(Vres[n-1], model)
        K_n = K(P_prev, tau[n], model)
        Vres[n] = (I - K_n @ model.C) @ P_prev
    return Vres

def get_mu(V, tau, model, X):
    # Preallocate mu with shape (N, npixel2)
    mu = np.empty((N, npixel2), dtype=model.m0.dtype)
    P0 = P(V[0], model)
    K0 = K(P0, tau[0], model)
    m0_col = model.m0.reshape(-1, 1)
    X0_col = X[0].reshape(-1, 1)
    mu[0] = (m0_col + K0 @ (X0_col - model.C @ m0_col)).ravel()
    for n in range(1, N):
        P_prev = P(V[n-1], model)
        K_prev = K(P_prev, tau[n], model)
        mu_prev = model.A @ mu[n-1].reshape(-1, 1)
        Xn_col = X[n].reshape(-1, 1)
        sub_n = Xn_col - model.C @ (model.A @ mu[n-1].reshape(-1, 1))
        mu[n] = (mu_prev + K_prev @ sub_n).ravel()
    return mu

def J(V_n, P_n, model):
    # Smoothing gain: J = V_n * A^T * inv(P_n)
    return V_n @ model.A.T @ np.linalg.inv(P_n)

def get_V_hat(V, model):
    # Preallocate V_hat with same shape as V.
    V_hat = np.empty_like(V)
    V_hat[-1] = V[-1]
    for n in range(N-2, -1, -1):
        P_n = P(V[n], model)
        J_n = J(V[n], P_n, model)
        V_hat[n] = V[n] + J_n @ (V_hat[n+1] - P_n) @ J_n.T
    return V_hat

def get_mu_hat(mu, V, model):
    # Preallocate mu_hat with same shape as mu.
    mu_hat = np.empty_like(mu)
    mu_hat[-1] = mu[-1]
    for n in range(N-2, -1, -1):
        P_n = P(V[n], model)
        J_n = J(V[n], P_n, model)
        # Ensure proper reshaping so that subtraction happens elementwise.
        mu_hat[n] = (mu[n].reshape(-1, 1) + J_n @ (mu_hat[n+1].reshape(-1, 1) - model.A @ mu[n].reshape(-1, 1))).ravel()
    return mu_hat

def get_E1(mu_hat):
    # E1 is the smoothed state estimates.
    return mu_hat

def get_E2_E3(V, mu_hat, V_hat, model):
    E2 = []
    E3 = []
    for n in range(1, N):
        P_prev = P(V[n-1], model)
        J_prev = J(V[n-1], P_prev, model)
        mu_hat_n_T = mu_hat[n].reshape(-1, 1)
        mu_hat_n_minus1 = mu_hat[n-1].reshape(1, -1)
        E2_n = V_hat[n] @ J_prev.T + mu_hat_n_T @ mu_hat_n_minus1
        E2.append(E2_n)
        E3.append(E2_n.T)
    return np.array(E2), np.array(E3)

def get_E4(mu_hat, V_hat):
    E4 = []
    for n in range(N):
        mu_hat_n = mu_hat[n].reshape(1, -1)
        E4_n = V_hat[n] + mu_hat_n.T @ mu_hat_n
        E4.append(E4_n)
    return np.array(E4)

# The remaining functions to update other parameters (m0, S0, A, Gamma, etc.)
def get_m0_new(E1):
    return E1[0]

def get_S0_new(E1, E4):
    E1_0 = E1[0].reshape(-1, 1)
    return E4[0] - E1_0 @ E1_0.T

def get_A_new(E2, E4):
    return np.sum(E2, axis=0) @ np.linalg.inv(np.sum(E4[:N-1], axis=0))

def get_Gamma_new(E2, E3, E4, A_new):
    elems = []
    for n in range(1, N):
        elems_n = E4[n] - A_new @ E3[n-1] - E2[n-1] @ A_new.T + A_new @ E4[n-1] @ A_new.T
        elems.append(elems_n)
    return np.sum(elems, axis=0) / (N - 1)

def get_C_new_former(E1, X):
    elems = []
    for n in range(N):
        x_n = X[n].reshape(-1, 1)
        E1_n = E1[n].reshape(1, -1)
        elems.append(x_n @ E1_n)
    return np.sum(elems, axis=0)

def get_C_new(E1, E4, X):
    return get_C_new_former(E1, X) @ np.linalg.inv(np.sum(E4, axis=0))

def get_Sigma_new(E1, E4, C_new, X):
    elems = []
    for n in range(N):
        x_n = X[n].reshape(-1, 1)
        E1_n = E1[n].reshape(1, -1)
        term1 = x_n @ x_n.T
        term2 = C_new @ E1_n.T @ x_n
        term3 = np.trace(C_new @ (E4[n] @ C_new.T))
        elems.append(np.squeeze(term1) - 2 * np.squeeze(term2) + term3)
    return np.sum(elems, axis=0) / (N * nvis)

def get_alpha_new(E2, E3, E4, A_new):
    trace_sum = 0
    for n in range(1, N):
        residual = E4[n] - A_new @ E3[n-1] - E2[n-1] @ A_new.T + A_new @ E4[n-1] @ A_new.T
        trace_sum += np.trace(residual)
    alpha_new = trace_sum / ((N - 1) * npixel2)
    print('alpha : ', alpha_new)
    return np.real(alpha_new)

def get_sigma2_new(E1, E4, C_new, X):
    residual_sum = 0.0
    for n in range(N):
        x_n = X[n].reshape(1, -1)
        E1_n = E1[n].reshape(1, -1)
        E4_n = E4[n]
        term1 = x_n @ x_n.T
        term2 = C_new @ E1_n.T @ x_n
        term3 = np.trace(C_new @ (E4_n @ C_new.T))
        residual_sum += np.squeeze(term1) - 2 * np.squeeze(term2) + term3
    sigma2_new = residual_sum / (N * nvis)
    print('sigma2 : ', sigma2_new)
    return sigma2_new
