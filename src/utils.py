from robii.astro.astro import generate_directions
from scipy.constants import speed_of_light
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.stats import gamma as gamma_dist


# src/

npixel2 = 4096
nvis = 351


def load_data(file_path):
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print("Error loading data:", e)
        return None

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.txt")
    try:
        with open(output_file, "w") as f:
            f.write(str(results))
        print(f"Results saved to {output_file}")
    except Exception as e:
        print("Error saving results:", e)

def plot_results(results):
    # Add your plotting code here
    plt.figure()
    # e.g., plt.plot(results['some_metric'])
    plt.title("EM Algorithm Results")
    plt.show()


def save_image(img, path, npixel=64, cmap='Spectral_r'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.imshow(img.reshape(npixel, npixel), cmap=cmap)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, dpi=1200, bbox_inches='tight', pad_inches=0)
    plt.close()


def forward_operator(uvw, freq, npixel, cellsize=None):
    """
    Used to initialize the model matrix

    """
    wl = speed_of_light / freq
    if cellsize is None:
        cellsize = np.min(wl)/(2*np.max(np.abs(uvw)))

    # cellsize = np.rad2deg(cellsize)
    # fov = cellsize*npixel
    lmn = generate_directions(npixel, cellsize).reshape(3,-1)
    uvw = uvw.reshape(-1,3)
    # add conjugate
    # uvw = np.concatenate((uvw, -uvw), axis=0)
    # print(uvw.shape)

    return np.exp(-1j*(2*np.pi/np.min(wl))* uvw @ lmn)


def complex_normal(mean, Cov, diag=True, rng=np.random.RandomState(0)): 

        n_samples = len(mean)

        if diag:
            
            _y_real = rng.normal(size=n_samples) * np.sqrt(np.real(Cov/2)) + np.real(mean)
            _y_imag = rng.normal(size=n_samples) * np.sqrt(np.real(Cov/2)) + np.imag(mean)
            _y = _y_real + 1j*_y_imag
            _y = _y.reshape((-1,1))

        else:
            if not np.allclose(Cov,Cov.T.conjugate(), atol=1e-08):
                raise ValueError("Covariance matrix must be symetric.")
            
            SIGMA = np.zeros((n_samples*2, n_samples*2))

            SIGMA[0: n_samples, 0:n_samples] = np.real(Cov)
            SIGMA[n_samples: 2*n_samples, n_samples:2*n_samples] = np.real(Cov)
            
            SIGMA[0: n_samples, n_samples:2*n_samples] = -np.imag(Cov)
            SIGMA[n_samples: 2*n_samples, 0: n_samples] = np.imag(Cov)

            SIGMA = (1/2)*SIGMA


            S,D,V = np.linalg.svd(SIGMA)

            # if not np.allclose(S,V.T.conjugate(), rtol=1):
            #     print(S - V.T.conjugate())
            #     raise ValueError("SVD - Covariance matrix is not symetric")

            S = np.dot(S, np.diag(np.sqrt(D)))

            MU = np.zeros(2*n_samples)
            MU[0:n_samples] = np.real(mean).reshape(-1)
            MU[n_samples:2*n_samples] = np.imag(mean).reshape(-1)
        
            _y = np.dot(S , rng.normal(0, 1, 2*n_samples)) + MU
            _y = (_y[0:n_samples] + 1j*_y[n_samples::]).reshape((-1,1))




        return _y


import numpy as np

def mvdr_beamformer(y, H, R_n, response_vector=None):
    """
    MVDR beamformer implementation.
utils.py
    
    Args:
        y: np.ndarray
            Observation vector of shape (M,).
        H: np.ndarray
            Steering matrix of shape (M, N).
        R_n: np.ndarray
            Noise covariance matrix of shape (M, M).
        response_vector: np.ndarray or None
            Desired response vector of shape (N,).
            If None, defaults to an all-ones vector.
    
    Returns:
        x_hat: np.ndarray
            Estimate of the signal of interest of shape (N,).
    """
    # Dimensions
    M, N = H.shape
    
    # Default response vector (all ones if not specified)
    if response_vector is None:
        response_vector = np.ones(N, dtype=H.dtype)
    
    # Compute the inverse of the noise covariance matrix
    R_n_inv = np.linalg.inv(R_n)
    
    # Compute the MVDR weights
    H_Rn_inv = R_n_inv @ H  # (M, M) x (M, N) = (M, N)
    numerator = H_Rn_inv @ np.linalg.inv(H.T.conj() @ H_Rn_inv) @ response_vector  # (M,)
    denominator = response_vector.conj().T @ np.linalg.inv(H.T.conj() @ H_Rn_inv) @ response_vector  # scalar
    w_mvdr = numerator / denominator  # (M,)
    
    # Apply the beamformer to observations
    x_hat = w_mvdr.conj().T @ y  # scalar or vector depending on H
    
    return x_hat



def mh_sample_tau_gamma(
    current_tau,
    y,
    x,
    C,
    Sigma,
    nu,
    shape_prop=2.5,
    scale_prop=1.0
):
    """
    Metropolis-Hastings update for tau>0 with a Gamma proposal.
    We do NOT use logs anywhere; we compute acceptance ratio in raw space.

    Args:
    ----
      current_tau: float, current value of tau
      y:  complex residual or observation? shape (m,)
      x:  complex or real state? shape (n,) [dim depends on your model]
      C:  observation matrix (m x n) complex or real
      Sigma: covariance matrix (m x m) for the noise
      nu: shape param for the prior Gamma(tau; nu/2, rate=nu/2) or similar
      shape_prop, scale_prop: parameters for the proposal Gamma dist

    Returns:
    --------
      new_tau: accepted or rejected sample
    """

    # =====================
    # (1) Propose tau' ~ Gamma(shape_prop, scale_prop)
    # =====================
    scale_prop = current_tau / shape_prop
    proposed_tau = gamma_dist.rvs(a=shape_prop, scale=scale_prop)

    # =====================
    # (2) Evaluate target posterior: pi(tau) = p(y|x,tau)* p(tau)
    #     in RAW (non-log) space
    # =====================
    # We'll define small helper function here:

    def posterior_tau(tau_val):
        # if tau_val <= 0 => posterior=0
        if tau_val <= 0:
            return 0.0

        # Residual
        r = y - C @ x  # shape (m,) possibly complex

        # dimension (m)
        m = len(r)

        # Mahalanobis distance under Sigma^-1
        invSigma = np.eye(nvis) #la.inv(Sigma) no need to inverse Sigma for now as we assumed Sigma = I
        # r^H Sigma^-1 r
        rSr = np.vdot(r, invSigma @ r).real

        # likelihood p(y|x, tau) ~ tau^m * exp(-tau*r^H Sigma^-1 r)
        # ignoring any constants wrt tau
        like = (tau_val**m) * np.exp(-tau_val * rSr)

        # prior p(tau) ~ tau^(nu/2 - 1)* exp(- (nu/2)* tau)
        # ignoring constants
        prior = (tau_val**(0.5*nu - 1.0)) * np.exp(-0.5*nu*tau_val)

        return like * prior

    p_current = posterior_tau(current_tau)
    p_proposed= posterior_tau(proposed_tau)

    # =====================
    # (3) Evaluate proposal q(tau'|tau), q(tau|tau')
    #     since shape_prop, scale_prop are fixed (do not depend on current_tau),
    #     q(tau'|tau) = q(tau'), and q(tau|tau')= q(tau).
    #     In other words, it's an 'independent' proposal.
    # =====================
    q_proposed_tau  = gamma_dist.pdf(proposed_tau, a=shape_prop, scale=scale_prop)
    q_current_tau   = gamma_dist.pdf(current_tau, a=shape_prop, scale=scale_prop)

    # =====================
    # (4) acceptance ratio in raw space
    # =====================
    # ratio = [ p_proposed * q(current_tau) ] / [ p_current * q(proposed_tau) ]
    #        = ( p_proposed / p_current ) * ( q_current / q_proposed )
    # check for zeros
    if p_current == 0 or q_proposed_tau == 0:
        # if the denominator is 0 => ratio can be infinite or 0 depending on numerator
        # handle carefully
        if p_proposed==0:
            ratio = 1.0  # effectively => accept prob=?
            # Actually p_proposed=0 => we definitely don't want to accept
            ratio = 0.0
        else:
            ratio = np.inf  # might => forced accept
    else:
        ratio = (p_proposed * q_current_tau) / (p_current * q_proposed_tau)

    # =====================
    # (5) accept/reject
    # =====================
    u = np.random.rand()
    if u < ratio:
        return proposed_tau
    else:
        return current_tau


# Define the proximal operator for the l1 norm (soft thresholding)
def proximal_operator(z, alpha):
    # For each element in z, perform soft-thresholding:
    # alpha = gamma * lam
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)


def get_gradient_Q(mu_hat, nu, dirty_images, F, H, Q, X):
    """
    Compute the gradient of the SEM Q function with respect to the latent states x,
    evaluated at the current state estimates mu_hat, under the new Student's t observation model.
    
    The observation log-likelihood is now modeled as:
    
        log p(y_t | x_t) = const - ((nu+m)/2)*log(1 + (1/nu)*||y_t - H*x_t||^2)
    
    whose gradient with respect to x_t is:
    
        grad_obs = ((nu+m)/nu) * (Re{H^H (y_t - H*x_t)}) / (1 + (1/nu)*||y_t - H*x_t||^2)
    
    The overall gradient also includes the state prior terms.
    
    Parameters:
        mu_hat (np.ndarray): Array of shape (T, n) representing the current state estimates.
        nu (float): Degrees of freedom for the Student's t likelihood.
        m (int): Dimension of the observation vector y_t.
    
    Returns:
        grad (np.ndarray): Array of shape (T, n) containing the gradient of Q with respect to x at each time step.
    """

    Sigma1_inv = 1/1e-3 * np.eye(npixel2)
    mu1 = dirty_images.reshape(npixel2, 1)/2500
    Q_inv = np.linalg.inv(Q)

    y = X
    m = nvis
    T, n = mu_hat.shape
    grad = np.zeros_like(mu_hat)
    mu1_local = mu1.reshape(n,)  # local copy for initial state mean

    # For t = 0 (first time step)
    residual = y[0] - H @ mu_hat[0]
    r2 = np.sum(np.abs(residual)**2)
    grad_obs = ((nu + m) / nu) * (np.real(H.T.conj() @ residual)) / (1 + (1/nu) * r2)
    grad[0] = (grad_obs -
               Sigma1_inv @ (mu_hat[0] - mu1_local) +
               F.T @ Q_inv @ (mu_hat[1] - F @ mu_hat[0]))
    
    # For t = 1,...,T-2 (interior time steps)
    for t in range(1, T - 1):
        residual = y[t] - H @ mu_hat[t]
        r2 = np.sum(np.abs(residual)**2)
        grad_obs = ((nu + m) / nu) * (np.real(H.T.conj() @ residual)) / (1 + (1/nu) * r2)
        grad[t] = (grad_obs -
                   Q_inv @ (mu_hat[t] - F @ mu_hat[t - 1]) +
                   F.T @ Q_inv @ (mu_hat[t + 1] - F @ mu_hat[t]))
    
    # For t = T-1 (last time step)
    residual = y[T - 1] - H @ mu_hat[T - 1]
    r2 = np.sum(np.abs(residual)**2)
    grad_obs = ((nu + m) / nu) * (np.real(H.T.conj() @ residual)) / (1 + (1/nu) * r2)
    grad[T - 1] = (grad_obs -
                   Q_inv @ (mu_hat[T - 1] - F @ mu_hat[T - 2]))
    
    return grad


