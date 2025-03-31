import numpy as np
# src/model.py

# Global variables
N = 10    # Number of time steps
nvis = 351  # Number of visibilities
npixel2 = 4096  # Number of pixels in the latent space

class StateSpaceModel:
    def __init__(self, m0, S0, A, Gamma, C, Sigma):
        self.m0 = m0
        self.S0 = S0
        self.A = A
        self.Gamma = Gamma
        self.C = C
        self.Sigma = Sigma

    def update_parameters(self, E1, E2, E3, E4, nvis, npixel2, gamma_variable):
        # Implement your M-step update logic here.    
        m0_new   = get_m0_new(E1)
        S0_new   = get_S0_new(E1, E4)
        A_new    = self.A  # or get_A_new(E2, E4) if you do that
        Gamma_new= np.eye(npixel2)*get_alpha_new(E2, E3, E4, A_new)
        C_new    = self.C
        Sigma_new= gamma_variable*np.eye(nvis)
        
        # store new params
        self.m0 = m0_new
        self.S0 = S0_new
        self.A = A_new
        self.Gamma = Gamma_new
        self.C = C_new
        self.Sigma = Sigma_new



def get_m0_new(E1):
    return E1[0]

def get_S0_new(E1, E4):
    E1_0 = np.array([E1[0]])
    E1_0_T = E1_0.transpose()
    return np.subtract(
        E4[0],
        np.matmul(E1_0_T, E1_0)
    )

def get_A_new(E2, E4):
    return np.matmul(
        np.sum(E2, axis=0),
        np.linalg.inv(np.sum(E4[:N-1], axis=0))
    )

def get_Gamma_new(E2, E3, E4, A_new):
    elems = np.empty((0,npixel2,npixel2))
    for n in range(1, N):
        elems_n = np.add(
            np.subtract(
                np.subtract(
                    E4[n],
                    np.matmul(
                        A_new,
                        E3[n-1]
                    )
                ),
                np.matmul(
                    E2[n-1],
                    A_new.transpose()
                )
            ),
            np.matmul(
                np.matmul(
                    A_new,
                    E4[n-1]
                ),
                A_new.transpose()
            )
        )
        elems = np.vstack((elems, [elems_n]))
    return np.sum(elems, axis=0) / (N-1)

def get_C_new_former(E1):
    elems = np.empty((0,nvis,npixel2))
    for n in range(N):
        x_n_T = np.array([X[n]]).transpose()
        E1_n = np.array([E1[n]])
        elems_n = np.matmul(x_n_T, E1_n)
        elems = np.vstack((elems, [elems_n]))
    return np.sum(elems, axis=0)

def get_C_new(E1, E4):
    return np.matmul(
        get_C_new_former(E1),
        np.linalg.inv(np.sum(E4, axis=0))
    )

def get_Sigma_new(E1, E4, C_new):
    elems = np.empty((0,nvis,nvis))
    for n in range(N):
        x_n = np.array([X[n]])
        x_n_T = x_n.transpose()
        E1_n = np.array([E1[n]])
        E1_n_T = E1_n.transpose()
        elem_n = np.add(
            np.subtract(
                np.subtract(
                    np.matmul(x_n_T, x_n),
                    np.matmul(
                        np.matmul(
                            C_new,
                            E1_n_T
                        ),
                        x_n
                    )
                ),
                np.matmul(
                    np.matmul(
                        x_n_T,
                        E1_n
                    ),
                    C_new.transpose()
                )
            ),
            np.matmul(
                np.matmul(
                    C_new,
                    E4[n]
                ),
                C_new.transpose()
            )
        )
        elems = np.vstack((elems, [elem_n]))
    return np.sum(elems, axis=0) / N


def get_alpha_new(E2, E3, E4, A_new):
    """
    Estimate the scalar parameter alpha for Q = alpha * I.

    Parameters:
    - E2, E3, E4: Expectation terms computed in the E-step
    - A_new: Updated transition matrix
    - npixel2: Dimension of the latent space (n_x)
    
    Returns:
    - alpha_new: Estimated scalar parameter for Q
    """
    trace_sum = 0  # To accumulate the trace values
    for n in range(1, N):
        # Compute the residual terms
        residual = np.subtract(
            np.subtract(
                E4[n],
                np.matmul(A_new, E3[n - 1])
            ),
            np.matmul(E2[n - 1], A_new.transpose())
        )
        residual = np.add(
            residual,
            np.matmul(
                np.matmul(A_new, E4[n - 1]),
                A_new.transpose()
            )
        )
        trace_sum += np.trace(residual)  # Accumulate the trace of the residual
    
    # Normalize the trace sum by (N-1) and divide by n_x
    alpha_new = trace_sum / ((N - 1) * npixel2)
    print(' alpha : ', alpha_new)
    return np.real(alpha_new)

def get_sigma2_new(E1, E4, C_new):
    """
    Estimate the scalar sigma^2 based on the posterior expectations.

    Parameters:
        E1 (numpy.ndarray): Posterior means of latent states, shape (N, nx).
        E4 (numpy.ndarray): Posterior covariances of latent states, shape (N, nx, nx).
        C_new (numpy.ndarray): Measurement matrix, shape (nvis, nx).

    Returns:
        float: Estimated scalar sigma^2.
    """
    residual_sum = 0.0
    for n in range(N):
        x_n = np.array([X[n]])  # Observation y_n
        E1_n = np.array([E1[n]])  # Posterior mean E[x_n]
        E1_n_T = E1_n.transpose()  # Transpose of posterior mean
        E4_n = E4[n]  # Posterior covariance E[x_n x_n^T]
        
        # Calculate terms for residual expectation
        term1 = np.matmul(x_n, x_n.T)  # y_n^T y_n
        term2 = np.matmul(np.matmul(C_new, E1_n_T), x_n)  # H E[x_n] y_n^T
        term3 = np.trace(np.matmul(C_new, np.matmul(E4_n, C_new.T)))  # tr(H E[x_n x_n^T] H^T)
        
        # Accumulate the residual norm for this time step
        residual_sum += np.squeeze(term1) - 2 * np.squeeze(term2) + term3
    
    # Compute sigma^2
    sigma2_new = residual_sum / (N * nvis)
    print(' sigma2 : ', sigma2_new)
    return sigma2_new
