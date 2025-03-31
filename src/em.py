import numpy as np
from model import StateSpaceModel
from utils import forward_operator, mh_sample_tau_gamma, proximal_operator, get_gradient_Q
import time

# Global variables (adjust as needed)
N = 10       # Number of time steps
nvis = 351   # Number of visibilities
npixel2 = 4096  # Number of pixels in the latent space

def run_em_algorithm(visibilities, dirty, FOp, model_images, tau, n_iterations=1):
    """
    Run the EM algorithm.
    Parameters:
      - visibilities, dirty, FOp, model_images: data inputs 
      - tau: initial tau parameter (or scaling)
    """
    # Create a model. Note: ensure StateSpaceModel is defined with the needed attributes.
    model = StateSpaceModel(
        m0=dirty,  # initial state (adjust if needed)
        S0=np.eye(npixel2)*1e-3,
        A=np.eye(npixel2),
        Gamma=np.eye(npixel2)*1e-4, 
        C=FOp,
        Sigma=tau*np.eye(nvis)
    )
    
    # Here, we assume X represents the observed data; choose one of visibilities or dirty.
    X = visibilities # Adjust if needed.
    
    smoothed_estimates = []
    MMSE = []
    tau_old = 0.9 * np.ones(N)

    for i in range(n_iterations):
        print(f"Iteration {i+1} of {n_iterations}")
        # E-step: compute sufficient statistics
        E1, E2, E3, E4, mu_hat, V_hat, tau_old = compute_E_step(X, dirty, model, tau_old)

        MMSE.append(np.sum(np.square(np.subtract(np.real(mu_hat), model_images))) / npixel2)
        print("====== MMSE (smoothed estimates) ========= {}".format(MMSE[-1]))
        smoothed_estimates.append(mu_hat)

        # M-step: update model parameters
        model.update_parameters(E1, E2, E3, E4, nvis, npixel2, tau)
        print(f"Iteration {i+1} completed.")

    return model, smoothed_estimates, MMSE

def compute_E_step(X, dirty, model, tau_old):
    """
    Performs the E-step computations.
    X: observed data (assumed to be an array-like with length N)
    """
    gamma = 1e-5        # Step size for the proximal update (tunable)
    mh_burn_in = 100

    t0 = time.time()

    # (A) E-step: forward pass (Kalman Filter)
    V = get_V(tau_old, model)
    mu = get_mu(V, tau_old, model, X)
    print('KF run, took {:.2f}s'.format(time.time() - t0))
    t0 = time.time()
    
    # (B) Backward pass (RTS Smoother)
    V_hat = get_V_hat(V, model)
    mu_hat = get_mu_hat(mu, V, model)
    print('KS run, took {:.2f}s'.format(time.time() - t0))
    t0 = time.time()

    # Regularization via Proximal Update
    nu = 2.5
    grad_Q = get_gradient_Q(mu_hat, nu, dirty, model.A, model.C, model.Gamma, X)
    z = mu_hat + gamma * grad_Q
    mu_hat = proximal_operator(z, 10e-2)

    # (C) Sample tau_t (using Metropolis-Hastings)
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
        # print("tau_old[{}]: {}".format(t, tau_old[t]))

    print("Sampling run, took {:.2f}s".format(time.time() - t0))
    t0 = time.time() 

    # (D) Compute E-step sufficient statistics
    E1 = get_E1(mu_hat)
    E2, E3 = get_E2_E3(V, mu_hat, V_hat, model)
    E4 = get_E4(mu_hat, V_hat)
    
    return E1, E2, E3, E4, mu_hat, V_hat, tau_old

# --- Helper functions: update these functions to accept 'model' (and 'X' if needed) ---

def P(V_n, model):
    res = np.matmul(model.A, V_n)
    res = np.matmul(res, model.A.transpose())
    res = res + model.Gamma  # A*V_n*A' + Gamma
    return res

def K(P_n, tau_n, model):
    # Kalman gain: K_n = P_n C^H [C P_n C^H + (1/tau_n)*Sigma]^-1
    num = np.matmul(P_n, model.C.transpose().conj())
    denom = np.matmul(np.matmul(model.C, P_n), model.C.transpose().conj()) + (1.0/tau_n)*model.Sigma
    K_n = np.matmul(num, np.linalg.inv(denom))
    return K_n

def get_V(tau, model):
    Vres = []
    I = np.eye(npixel2)
    K_minus1 = K(model.S0, tau[0], model)
    V_0 = np.matmul((I - np.matmul(K_minus1, model.C)), model.S0)
    Vres.append(V_0)
    for n in range(1, N):
        P_n_minus1 = P(Vres[n-1], model)
        K_n = K(P_n_minus1, tau[n], model)
        V_n = np.matmul((I - np.matmul(K_n, model.C)), P_n_minus1)
        Vres.append(V_n)
    return np.array(Vres)

def get_mu(V, tau, model, X):
    mu = []
    # Initial step
    P_0 = P(V[0], model)
    K_0 = K(P_0, tau[0], model)
    m0_col = model.m0.reshape(-1, 1)
    X0_col = X[0].reshape(-1, 1)
    mu_0 = (m0_col + np.matmul(K_0, (X0_col - np.matmul(model.C, m0_col)))).ravel()
    mu.append(mu_0)
    for n in range(1, N):
        P_n_minus1 = P(V[n-1], model)
        K_n_minus1 = K(P_n_minus1, tau[n], model)
        mu_n_former = np.matmul(model.A, mu[n-1].reshape(-1, 1))
        Xn_col = X[n].reshape(-1, 1)
        sub_n = Xn_col - np.matmul(model.C, np.matmul(model.A, mu[n-1].reshape(-1, 1)))
        mu_n = (mu_n_former + np.matmul(K_n_minus1, sub_n)).ravel()
        mu.append(mu_n)
    return np.array(mu)

def J(V_n, P_n, model):
    return np.matmul(np.matmul(V_n, model.A.transpose()), np.linalg.inv(P_n))

def get_V_hat(V, model):
    V_hat_rev = []
    V_hat_rev.append(V[N-1])
    for n in range(1, N):
        V_n = V[N-n-1]
        P_n = P(V_n, model)
        J_n = J(V_n, P_n, model)
        V_hat_n = V_n + np.matmul(np.matmul(J_n, (V_hat_rev[n-1] - P_n)), J_n.transpose())
        V_hat_rev.append(V_hat_n)
    V_hat_rev = np.array(V_hat_rev)
    V_hat = np.flip(V_hat_rev, axis=0)
    return V_hat

def get_mu_hat(mu, V, model):
    mu_hat_rev = []
    mu_hat_rev.append(mu[N-1])
    for n in range(1, N):
        mu_n = mu[N-n-1]
        mu_n_T = np.array(mu_n).reshape(-1, 1)
        mu_hat_prev = np.array(mu_hat_rev[n-1]).reshape(-1, 1)
        P_n = P(V[N-n-1], model)
        J_n = J(V[N-n-1], P_n, model)
        mu_hat_n = (mu_n_T + np.matmul(J_n, (mu_hat_prev - np.matmul(model.A, mu_n_T)))).ravel()
        mu_hat_rev.append(mu_hat_n)
    mu_hat_rev = np.array(mu_hat_rev)
    mu_hat = np.flip(mu_hat_rev, axis=0)
    return mu_hat

def get_E1(mu_hat):
    return mu_hat

def get_E2_E3(V, mu_hat, V_hat, model):
    E2 = []
    E3 = []
    for n in range(1, N):
        P_n_minus1 = P(V[n-1], model)
        J_n_minus1 = J(V[n-1], P_n_minus1, model)
        mu_hat_n_T = np.array(mu_hat[n]).reshape(-1, 1)
        mu_hat_n_minus1 = np.array(mu_hat[n-1]).reshape(1, -1)
        E2_n = np.matmul(V_hat[n], J_n_minus1.transpose()) + np.matmul(mu_hat_n_T, mu_hat_n_minus1)
        E2.append(E2_n)
        E3.append(E2_n.transpose())
    return np.array(E2), np.array(E3)

def get_E4(mu_hat, V_hat):
    E4 = []
    for n in range(N):
        mu_hat_n = np.array(mu_hat[n]).reshape(1, -1)
        E4_n = V_hat[n] + np.matmul(mu_hat_n.transpose(), mu_hat_n)
        E4.append(E4_n)
    return np.array(E4)

def get_m0_new(E1):
    return E1[0]

def get_S0_new(E1, E4):
    E1_0 = np.array(E1[0]).reshape(-1, 1)
    return E4[0] - np.matmul(E1_0, E1_0.transpose())

def get_A_new(E2, E4):
    return np.matmul(np.sum(E2, axis=0), np.linalg.inv(np.sum(E4[:N-1], axis=0)))

def get_Gamma_new(E2, E3, E4, A_new):
    elems = []
    for n in range(1, N):
        elems_n = E4[n] - np.matmul(A_new, E3[n-1]) - np.matmul(E2[n-1], A_new.transpose()) \
                  + np.matmul(np.matmul(A_new, E4[n-1]), A_new.transpose())
        elems.append(elems_n)
    return np.sum(elems, axis=0) / (N - 1)

def get_C_new_former(E1, X):
    elems = []
    for n in range(N):
        x_n = np.array(X[n]).reshape(-1, 1)
        E1_n = np.array(E1[n]).reshape(1, -1)
        elems_n = np.matmul(x_n, E1_n)
        elems.append(elems_n)
    return np.sum(elems, axis=0)

def get_C_new(E1, E4, X):
    return np.matmul(get_C_new_former(E1, X), np.linalg.inv(np.sum(E4, axis=0)))

def get_Sigma_new(E1, E4, C_new, X):
    elems = []
    for n in range(N):
        x_n = np.array(X[n]).reshape(-1, 1)
        E1_n = np.array(E1[n]).reshape(1, -1)
        term1 = np.matmul(x_n, x_n.transpose())
        term2 = np.matmul(np.matmul(C_new, E1_n.transpose()), x_n)
        term3 = np.trace(np.matmul(C_new, np.matmul(E4[n], C_new.transpose())))
        elems.append(np.squeeze(term1) - 2 * np.squeeze(term2) + term3)
    return np.sum(elems, axis=0) / (N * nvis)

def get_alpha_new(E2, E3, E4, A_new):
    trace_sum = 0
    for n in range(1, N):
        residual = E4[n] - np.matmul(A_new, E3[n-1]) - np.matmul(E2[n-1], A_new.transpose()) \
                   + np.matmul(np.matmul(A_new, E4[n-1]), A_new.transpose())
        trace_sum += np.trace(residual)
    alpha_new = trace_sum / ((N - 1) * npixel2)
    print('alpha : ', alpha_new)
    return np.real(alpha_new)

def get_sigma2_new(E1, E4, C_new, X):
    residual_sum = 0.0
    for n in range(N):
        x_n = np.array(X[n]).reshape(1, -1)
        E1_n = np.array(E1[n]).reshape(1, -1)
        E4_n = E4[n]
        term1 = np.matmul(x_n, x_n.transpose())
        term2 = np.matmul(np.matmul(C_new, E1_n.transpose()), x_n)
        term3 = np.trace(np.matmul(C_new, np.matmul(E4_n, C_new.transpose())))
        residual_sum += np.squeeze(term1) - 2 * np.squeeze(term2) + term3
    sigma2_new = residual_sum / (N * nvis)
    print('sigma2 : ', sigma2_new)
    return sigma2_new
