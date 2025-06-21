import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import ttest_rel
from numpy.linalg import inv, eigvals

# Simulated time series data for 3 channels, 1000 time points, 10 subjects
np.random.seed(42)
n_subjects = 10
n_channels = 3
n_timepoints = 1000
data1 = np.random.randn(n_subjects, n_channels, n_timepoints)
data2 = np.random.randn(n_subjects, n_channels, n_timepoints) * 1.1  # Slightly different condition

def compute_mvar_coefficients(X, order=5):
    # Estimate MVAR coefficients using least squares
    T = X.shape[-1]
    Y = X[:, order:]
    Z = np.concatenate([X[:, order - k - 1:T - k - 1] for k in range(order)], axis=0)
    A = Y @ Z.T @ inv(Z @ Z.T)
    return A.reshape(n_channels, order, n_channels)

def compute_dtf(A, freqs, fs):
    order = A.shape[1]
    H = np.zeros((n_channels, n_channels, len(freqs)), dtype=np.complex_)
    I = np.eye(n_channels)
    for f_idx, f in enumerate(freqs):
        exp_term = sum(A[:, k, :] * np.exp(-2j * np.pi * f * (k + 1) / fs) for k in range(order))
        H[:, :, f_idx] = inv(I - exp_term)
    dtf = np.abs(H) ** 2
    dtf_norm = dtf / dtf.sum(axis=1, keepdims=True)
    return dtf_norm

# Parameters
fs = 100  # Hz
freqs = np.linspace(8, 12, 20)  # Alpha band

# Compute dDTF for each subject
dtf1_all = []
dtf2_all = []

for s in range(n_subjects):
    A1 = compute_mvar_coefficients(data1[s])
    A2 = compute_mvar_coefficients(data2[s])
    dtf1 = compute_dtf(A1, freqs, fs)
    dtf2 = compute_dtf(A2, freqs, fs)
    dtf1_all.append(dtf1.mean(axis=-1))  # Average over freq
    dtf2_all.append(dtf2.mean(axis=-1))

dtf1_all = np.stack(dtf1_all)
dtf2_all = np.stack(dtf2_all)

# Compute connectivity strength and directionality
conn_strength1 = dtf1_all.mean(axis=0)
conn_strength2 = dtf2_all.mean(axis=0)

directionality = (conn_strength1 - conn_strength1.T) / (conn_strength1 + conn_strength1.T + 1e-8)

# Paired t-test
p_values = np.zeros((n_channels, n_channels))
for i in range(n_channels):
    for j in range(n_channels):
        _, p = ttest_rel(dtf1_all[:, i, j], dtf2_all[:, i, j])
        p_values[i, j] = p

# Visualize
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

im1 = axs[0].imshow(conn_strength1, cmap='viridis')
axs[0].set_title('Connectivity Strength (Condition 1)')
plt.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(directionality, cmap='seismic', vmin=-1, vmax=1)
axs[1].set_title('Directionality Index')
plt.colorbar(im2, ax=axs[1])

im3 = axs[2].imshow(p_values, cmap='hot_r')
axs[2].set_title('P-values (Condition1 vs Condition2)')
plt.colorbar(im3, ax=axs[2])

plt.tight_layout()
plt.show()
