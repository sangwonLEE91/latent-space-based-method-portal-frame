# latent-space-based-method-portal-frame

## Project Overview
This repository implements a Bayesian model-updating pipeline for a 2-D portal-frame structure by directly applying the latent-space likelihood estimation approach of Lee et al. (2024).  
OpenSeesPy generates frequency-response data under El Centro ground motion, a Variational Autoencoder (VAE) evaluates the likelihood of candidate parameters in a compact latent space as described in the paper, and Transitional Markov Chain Monte Carlo (TMCMC) produces posterior samples of the structural parameters.

## Key Components
| File / Folder            | Purpose                                                                                        |
|--------------------------|------------------------------------------------------------------------------------------------|
| `Model_updating_main.py` | Coordinates TMCMC sampling, likelihood evaluation via the VAE, and tempering adaptation.       |
| `utils.py`               | Helper functions: Latin-hypercube sampling, weight updates, covariance calculation, plotting. |
| `simulator.py`           | Defines the portal-frame model and runs static/dynamic analyses with OpenSeesPy.               |
| `VAE_residual.py`        | Implements the VAE used for latent-space likelihood evaluation.                                |
| `observation/`           | Measured frequency-response data (`response_obs.npy`).                                         |
| `el_centro.npy`          | El Centro earthquake record applied as dynamic loading.                                        |
| `torch_model/`           | Saved VAE weights.                                                                             |
| `result/`                | Generated posterior samples, log-likelihood traces, tempering schedules, and figures.         |

## Quick Usage
1. Edit configuration variables in `Model_updating_main.py` (e.g., `z_dim`, number of particles `Ns`, learning rates).  
2. Run: python Model_updating_main.py
3. In the `result/` directory you will find:
- posterior samples (`samples.npy`)
- log-likelihood
- tempering factors
- scatter/histogram plots

## Related Publication
If this code is helpful for your research, please cite:

```bibtex
@article{Lee24,
  author  = {Sangwon Lee and Taro Yaoyama and Yuma Matsumoto and Takenori Hida and Tatsuya Itoi},
  title   = {Latent Space-Based Likelihood Estimation Using a Single Observation for Bayesian Updating of a Nonlinear Hysteretic Model},
  journal = {ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering},
  volume  = {10},
  number  = {4},
  pages   = {04024072},
  year    = {2024},
  doi     = {10.1061/AJRUA6.RUENG-1305}
}
