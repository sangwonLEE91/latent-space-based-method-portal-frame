import numpy as np
import torch
from simulator import *
from utils import *
import pickle
from VAE_residual import VAE
import os
from scipy.stats import uniform

test = 1

N0 = 10000  # Number of training samples
Nv0 = 1000  # Number of validation samples

learning_rate = 0.0001  # Learning rate for the VAE training
train_patience = 100  # Patience for stopping during VAE training

z_dim = 10 # Dimension of the latent space for the VAE
Ns = 500 # Number of samples at each TMCMC step
batch_size = 128 # Batch size for VAE training
nstep = 128 # Number of frequency steps in frequency response function
kmin = 10 # cutoff frequency step for the frequency response function


trained_model = True  # True if you want to use the trained model, False if you want to train a new one

root = f'result/test{test}/'
os.makedirs(root, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

### Define true parameters and load observation
true_theta = np.array([3, 3 + np.log10(5), np.log10(2)]) # Ground truth parameters for the portal frame model
y_obs = np.load('observation/response_obs.npy')
print(f'y_obs shape: {y_obs.shape}, true_theta: {true_theta}')

### Priors based on the case
prior1 = uniform(1, 4)
prior2 = uniform(1, 4)
prior3 = uniform(0, 1)
prior = [prior1, prior2, prior3]
prior_pdf = lambda theta: prior_p(theta, prior=prior)
range_theta = [[1, 1, 0], [4, 4, 1]]
prior_rnd = lambda n_samples: LHS(n_samples, bounds=range_theta)


### Constants  for TMCMC
beta2 = 0.1  # square of scaling parameter
Nst = 30  # number of preallocated steps

### Preallocation
samples = {'pop': [None] * Nst}  # samples at each step
logL_j = [None] * Nst  # log-likelihood at each step
# Z_j = np.zeros(Nst)  # intermediate marginal likelihood
p_j = np.zeros(Nst)  # tempering parameter at each step

### Initialize the first step
j = 1
p_j[0] = 0
print(f'TMCMC iteration: j = {j - 1}, p_j = {p_j[j - 1]}')


### Load the El Centro earthquake data
dt = 0.02
time_series, excitation = load_elcentro()
ndata = len(time_series)
taper = taper_maker(dt, 1, ndata)
sim = lambda obs: multi_sim(obs, taper=taper, time_series=time_series, excitation=excitation, kmin=kmin, nstep=nstep,
                            add_noise=True)

### Sample from the prior distribution
samples['pop'][j - 1] = prior_rnd(Ns)  # initial MCMC samples

### Generate dataset
if trained_model != True:  # If you want to train a new model
    labels = prior_rnd(N0)
    Y0 = sim(labels)
    Dataset = [Y0, labels]
label_val = prior_rnd(Nv0)
Y0_val = sim(label_val)
Dataset_val = [Y0_val, label_val]

scatter_plot(samples['pop'][j - 1], true_theta,root, j)
Ncall = N0 + Nv0

Ndim = samples['pop'][j - 1].shape[1]  # number of dimensions

### Training of VAE
net = VAE(z_dim, y_obs.shape[1], nstep, device).to(device)
save_torch = f'torch_model/t{test}.pth'
if trained_model != True: # If you want to train a new model
    train_vae(net, Dataset, device, save_torch, train_patience, LR=learning_rate, valid_dataset=Dataset_val)
    net.load_state_dict(torch.load(save_torch, map_location=device, weights_only=True))
else: # If you want to use the trained model
    net.load_state_dict(torch.load(save_torch, map_location=device, weights_only=True))
net.eval()

#### calculate p(z|D) and p(z) with trained VAE and dataset
z_D, z = initialize_vae(y_obs, net.enc, device, Dataset_val)

#### Sequential importance sampling
while p_j[j - 1] < 1:
    repeat = True
    print(f'TMCMC iteration: j = {j}, ', end='')
    # Y_sample = sim(samples['pop'][j - 1])
    logL_j[j - 1] = Lfun_vae(samples['pop'][j - 1], net.enc, sim, device, z_D, z)
    Ncall += Ns
    # Calculate the tempering parameter (Bisection method)
    p_j[j] = tempering_parameter(p_j[j - 1], logL_j[j - 1])
    log_fD_T_adjust = np.max(logL_j[j - 1])
    print(f'p_j = {p_j[j]}')

    # Compute the intermediate marginal likelihood
    # Z_j[j - 1] = np.mean(np.exp((logL_j[j - 1] - log_fD_T_adjust) * (p_j[j] - p_j[j - 1]))) + np.exp((p_j[j] - p_j[j - 1]) * log_fD_T_adjust)

    # Compute the weighted mean and covariance matrix for the proposal PDF
    w_j, S_j = cal_wj_Sj(p_j[j] - p_j[j - 1], logL_j[j - 1], log_fD_T_adjust, samples['pop'][j - 1], Ns, Ndim,
                         beta2)
    # Generate conditional samples using MH algorithm
    samples_j, logL_jp = MHalgorithm_vae(w_j, Ns, samples['pop'][j - 1], logL_j[j - 1], S_j, net.enc, sim, device,
                                         z_D, z, p_j[j], prior_pdf, n_burn=0)
    Ncall += Ns

    samples['pop'][j] = samples_j
    logL_j[j] = logL_jp

    scatter_plot(samples_j, true_theta,root, j)

    j += 1

print('N call = ', Ncall)
# Sample from the posterior distribution
m = j

samples['pop'] = samples['pop'][:m]

samples['post'] = samples_j
logL_j = logL_j[:m]
p_j = p_j[:m]
# Z_j = Z_j[:m - 1]
# Z_m = np.prod(Z_j)

np.save(root + f'logL_t{test}.npy', np.array(logL_j))
np.save(root + f'p_j_t{test}.npy', p_j)
with open(root + f'samples_t{test}.pickle', 'wb') as fw:
    pickle.dump(samples, fw)
