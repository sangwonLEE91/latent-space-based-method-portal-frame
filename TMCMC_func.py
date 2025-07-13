import numpy as np
from scipy.stats import multivariate_normal
import torch
from train import train_sequential_MVAE, train_sequential_VAE
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc
import copy

def prior_r(Nsam, prior):
    # generate uncorrelat
    npar = len(prior)
    samples = np.zeros((Nsam, npar))
    for i in range(npar):
        samples[:, i] = prior[i].rvs(Nsam)
    return samples


def LHS(Nsam, bounds):
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    n_dim = len(min_bounds)
    sampler = qmc.LatinHypercube(d=n_dim)
    samples_SN = sampler.random(n=Nsam)
    scaled_samples = qmc.scale(samples_SN, min_bounds, max_bounds)
    return scaled_samples


def prior_p(theta, prior):
    npar = len(prior)
    if len(theta.shape) == 2:
        p = np.ones(len(theta))
        for i in range(len(theta)):
            for j in range(npar):
                p[i] *= prior[j].pdf(theta[i, j])
    else:
        p = 1
        for j in range(npar):
            p *= prior[j].pdf(theta[j])
    return p

def scatter_plot(samples, dataset, added, true_theta):
    columns = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
    # dataset = dataset_loaded[i]
    df = pd.DataFrame(samples, columns=columns)
    if len(dataset) != 0:
        df2 = pd.DataFrame(dataset, columns=columns)
    if len(added) != 0:
        df3 = pd.DataFrame(added, columns=columns)
    # 원하는 축 범위를 정의합니다.
    axis_ranges = {r'$\theta_1$': [0, 4], r'$\theta_2$': [0, 4], r'$\theta_3$': [0, 1]}
    N_bins = 20
    g = sns.pairplot(df, plot_kws={'color': 'gray', 's': 1, 'alpha': 0.5},
                     diag_kws={'color': 'gray', 'edgecolor': 'black', 'bins': N_bins})
    # 첫 번째 히스토그램 서브플롯의 y축 틱 라벨 제거 (기존 코드 유지)
    g.axes[0, 0].set_yticks([])
    g.axes[0, 0].set_ylabel('')
    # 축 레이블 설정 (기존 코드 유지)
    for ax in g.axes.flatten():
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
    variables = df.columns
    axes = g.axes

    # 각 축에 대해 df2 데이터를 추가로 플롯
    for ii, var1 in enumerate(variables):
        for jj, var2 in enumerate(variables):
            ax = g.axes[ii, jj]
            if ii != jj:
                # df2 데이터 추가
                if len(dataset) != 0:
                    ax.scatter(df2[var2], df2[var1], color='C0', s=4, label='Data 2', alpha=0.8)
                if len(added) != 0:
                    ax.scatter(df3[var2], df3[var1], color='C1', s=9, label='Data 2', alpha=0.8)
                pass
            else:
                # 히스토그램 df2 추가
                # ax.hist(df2[var1], bins=N_bins, color='C1', edgecolor='black', alpha=0.8)
                pass

    # 각 서브플롯에 대해 축 범위를 설정합니다.
    for ii, var1 in enumerate(variables):
        for jj, var2 in enumerate(variables):
            ax = axes[ii, jj]
            if ii == jj:
                # 대각선 히스토그램에 대해 x축 범위 설정
                ax.set_xlim(axis_ranges[var1])
            else:
                # 산점도에 대해 x축과 y축 범위 설정
                ax.set_xlim(axis_ranges[var2])
                ax.set_ylim(axis_ranges[var1])
                ax.axhline(true_theta[ii], 0, 1, linewidth=1, linestyle=':', color='r')
                ax.axvline(true_theta[jj], 0, 1, linewidth=1, linestyle=':', color='r')
                ax.plot(true_theta[jj], true_theta[ii], linestyle='', marker='o', markersize=5, color='r',
                        label='Target')
    plt.show()



def gauss_gauss_kl(mean1, var1, mean2, var2):
    epsilon_val = 1e-8  # Small constant to avoid NaN
    _var2 = var2 + epsilon_val
    _kl = np.log(_var2) - np.log(var1) \
          + (var1 + (mean1 - mean2) ** 2) / _var2 - 1
    return 0.5 * np.mean(np.sum(_kl))

def cal_log_hpvc(z_D, z_theta, z, beta):
    z_dim = len(z_D[0])
    LH = 0.
    for i in range(z_dim):
        mu1 = z_D[0][i]
        mu2 = z_theta[0][i]
        mu3 = z[0][i]
        vr1 = z_D[1][i]
        vr2 = z_theta[1][i]
        vr3 = z[1][i]

        c0 = np.sqrt(vr3 / ( (2 * np.pi * vr1)**(beta) * vr2))
        a = ((beta) / (2 * vr1)) + (1 / (2 * vr2)) - (1 / (2 * vr3))
        b = -((beta) * mu1 / vr1) - (mu2 / vr2) + (mu3 / vr3)
        c = ((beta)* mu1 ** 2 / (2 * vr1)) + (mu2 ** 2 / (2 * vr2)) - (mu3 ** 2 / (2 * vr3))
        LH += np.log(c0) + ((b ** 2 - 4 * a * c) / (4 * a)) + np.log(np.sqrt(np.pi / a))
    return LH

def Hpvc(samples,Pj,dpj, encoder, device, z_D, z, threshold=0.1):
    encoder.eval()
    N = len(samples)
    Y_loader = data2loader_l(samples, batch_size=128, shuffle=False)
    z_theta_mu = []
    z_theta_var = []
    for counter, label in enumerate(Y_loader):
        label = label.to(device)
        _, m, v = encoder(label)
        z_theta_mu += m.cpu().detach().numpy().tolist()
        z_theta_var += v.cpu().detach().numpy().tolist()
    z_theta_mu = np.array(z_theta_mu)
    z_theta_var = np.array(z_theta_var)
    log_w_list = []
    for j in range(N):
        z_theta = [z_theta_mu[j, :].tolist(), z_theta_var[j, :].tolist()]
        log_h = cal_log_hpvc(z_D, z_theta, z, Pj)
        log_w_list.append(dpj * log_h)
    log_w = np.array(log_w_list)
    log_w -= np.max(log_w)

    # 2) 지수 변환 & NaN/Inf 제거
    w_unn = np.exp(log_w)
    w_unn = np.nan_to_num(w_unn, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) 0.1 이상인 normalized 가중치는 나중에 버리기 위해 일단 보류
    #    -> 일단 정규화
    total = np.sum(w_unn)
    if total > 0:
        wpvc = w_unn / total
    else:
        wpvc = np.ones_like(w_unn) / N

    # 4) 임계값 초과 가중치 제거
    mask = wpvc > threshold
    if np.any(mask):
        wpvc[mask] = 0.0
        # 5) 남은 값으로 재정규화
        rem = wpvc.sum()
        if rem > 0:
            wpvc /= rem
        else:
            # 만약 전부 잘렸다면 균등분포로 복구
            wpvc = np.ones_like(wpvc) / N

    return wpvc

def Lfun_vae(Y_label, encoder, sim, device, z_D, z):
    encoder.eval()
    N = len(Y_label)
    Y_data = sim(Y_label)

    Y_loader = data2loader_l(Y_data, batch_size=128, shuffle=False, droplast=False)
    z_theta_mu = []
    z_theta_var = []
    for counter, wave in enumerate(Y_loader):
        wave = wave.to(device)
        _, m, v = encoder(wave)
        z_theta_mu += m.cpu().detach().numpy().tolist()
        z_theta_var += v.cpu().detach().numpy().tolist()
    z_theta_mu = np.array(z_theta_mu)
    z_theta_var = np.array(z_theta_var)

    logL = []
    for j in range(N):
        z_theta = [z_theta_mu[j, :].tolist(), z_theta_var[j, :].tolist()]
        # 적분
        logL.append(cal_loglikelihood(z_D, z_theta, z))
    return np.array(logL)

def Lfun(Y_label, encoder, device, z_D, z):
    encoder.eval()
    N = len(Y_label)
    Y_loader = data2loader_l(Y_label, batch_size=64, shuffle=False, droplast=False)
    z_theta_mu = []
    z_theta_var = []
    for counter, label in enumerate(Y_loader):
        label = label.to(device)
        _, m, v = encoder(label)
        z_theta_mu += m.cpu().detach().numpy().tolist()
        z_theta_var += v.cpu().detach().numpy().tolist()
    z_theta_mu = np.array(z_theta_mu)
    z_theta_var = np.array(z_theta_var)

    logL = []
    for j in range(N):
        z_theta = [z_theta_mu[j, :].tolist(), z_theta_var[j, :].tolist()]
        # 적분
        logL.append(cal_loglikelihood(z_D, z_theta, z))
    return np.array(logL)
def cal_likelihood(f_obs, theta):
    log_LH = np.zeros((theta.shape[0],))
    for i in range(len(log_LH)):
        if any(theta[i, :] < 0):
            log_LH[i] = 10 ** (-50)
        else:
            omega,_ = eigenvalue_problem(theta[i, :])
            f = omega/(2*np.pi)
            log_LH[i] = - measure_of_fit(f, f_obs) / (2* (1 / 16) ** 2)
    return log_LH

def cal_likelihood_s(f_obs,theta):
    if any(theta < 0):
        log_LH = 10 ** (-50)
    else:
        omega, _ = eigenvalue_problem(theta)
        f = omega / (2 * np.pi)
        log_LH = - measure_of_fit(f, f_obs) / (2 * (1 / 16) ** 2)
    return log_LH

def measure_of_fit(f,f_hat, ld=[1, 1, 1, 1, 1]):
    Mx = 0
    for i in range(len(f_hat)):
        Mx += ld[i] ** 2 * (f[i] ** 2 / f_hat[i] ** 2 - 1) ** 2
    return Mx


def Lfun_chain(Y_cands, Nsample, encoder, device, z_D, z):
    candidates = np.vstack(Y_cands)
    logL = Lfun(candidates, encoder, device, z_D, z)
    Nchain = len(Nsample)
    logL_cand = [None] * Nchain
    iii = 0
    for i in range(Nchain):
        logL_cand[i] = np.zeros((Nsample[i],))
        for ii in range(Nsample[i]):
            logL_cand[i][ii] = logL[iii]
            iii += 1
    return logL_cand

def MHalgorithm(w_j, Ns, samples, logL_j1, S_j, MVAE_enc_w, device, z_D, z, p_jj, prior_pdf, n_burn=1):

    w_csum = np.cumsum(w_j)
    Nseed = np.zeros(Ns, dtype=int)
    Nseed = np.searchsorted(w_csum, np.random.rand(Ns), side='right')

    samples_seed = samples[Nseed, :]
    logL_seed = logL_j1[Nseed]
    logP_seed = np.log(prior_pdf(samples_seed))  # prior

    for _ in range(n_burn):
        noise = np.random.multivariate_normal(mean=np.zeros(S_j.shape[0]), cov=S_j, size=Ns)
        samples_cand = samples_seed + noise
        pdf_cand = prior_pdf(samples_cand)
        # 1) 結果を -inf で埋めた配列を用意
        logP_cand = np.full_like(pdf_cand, fill_value=-np.inf, dtype=float)

        # 2) 正の要素だけに対して np.log を適用
        positive = pdf_cand > 0
        logP_cand[positive] = np.log(pdf_cand[positive])
        logL_cand = Lfun(samples_cand, MVAE_enc_w, device, z_D, z)  # likelihood

        # acceptance
        delta_log = p_jj * (logL_cand - logL_seed) + (logP_cand - logP_seed)
        alpha_vec = np.exp(np.clip(delta_log, a_min=-100, a_max=100))  # to prevent under/overflow
        u = np.random.rand(Ns)
        accept_mask = (u <= np.minimum(1.0, alpha_vec))
        samples_j = np.where(accept_mask[:, None], samples_cand, samples_seed)
        logL_j = np.where(accept_mask, logL_cand, logL_seed)

        #initial for next iteration
        samples_seed = copy.deepcopy(samples_j)
        logL_seed = copy.deepcopy(logL_j)
        logP_seed = np.where(accept_mask, logP_cand, logP_seed)

    return samples_j, logL_j


def MHalgorithm_vae(w_j, Ns, samples, logL_j1, S_j, VAE_enc, sim, device, z_D, z, p_jj, prior_pdf, n_burn=1):
    w_csum = np.cumsum(w_j)
    Nseed = np.zeros(Ns, dtype=int)
    Nseed = np.searchsorted(w_csum, np.random.rand(Ns), side='right')

    samples_seed = samples[Nseed, :]
    logL_seed = logL_j1[Nseed]
    logP_seed = np.log(prior_pdf(samples_seed))  # prior

    for _ in range(n_burn):
        noise = np.random.multivariate_normal(mean=np.zeros(S_j.shape[0]), cov=S_j, size=Ns)
        samples_cand = samples_seed + noise
        # --- prior の対数を計算する部分はそのまま ---
        pdf_cand = prior_pdf(samples_cand)
        logP_cand = np.full_like(pdf_cand, fill_value=-np.inf, dtype=float)
        positive = pdf_cand > 0
        logP_cand[positive] = np.log(pdf_cand[positive])

        # --- logL_cand を“小さな負値”で初期化 ---
        n_cand = samples_cand.shape[0]
        logL_cand = np.full((n_cand,), fill_value=-1e10, dtype=float)

        # --- 各サンプル行が全要素 正かどうかを判定 ---
        valid = np.all(samples_cand > 0, axis=1)

        # --- 正の行だけで Lfun_vae を呼び出して書き換え ---
        if np.any(valid):
            logL_vals = Lfun_vae(samples_cand[valid], VAE_enc, sim, device, z_D, z)
            logL_cand[valid] = logL_vals

        # acceptance
        delta_log = p_jj * (logL_cand - logL_seed) + (logP_cand - logP_seed)
        alpha_vec = np.exp(np.clip(delta_log, a_min=-100, a_max=100))  # to prevent under/overflow
        u = np.random.rand(Ns)
        accept_mask = (u <= np.minimum(1.0, alpha_vec))
        samples_j = np.where(accept_mask[:, None], samples_cand, samples_seed)
        logL_j = np.where(accept_mask, logL_cand, logL_seed)

        # initial for next iteration
        samples_seed = copy.deepcopy(samples_j)
        logL_seed = copy.deepcopy(logL_j)
        logP_seed = np.where(accept_mask, logP_cand, logP_seed)

    return samples_j, logL_j

def MHalgorithm_ordinary(w_j,Ns, Ndim, samples, logL_j1,S_j, p_jj, prior_pdf, prior, f_obs):
    w_csum = np.cumsum(w_j)
    Nseed = np.zeros(Ns, dtype=int)
    for i in range(Ns):
        Nseed[i] = np.searchsorted(w_csum, np.random.rand(), side='right')

    idx = np.unique(Nseed).astype(int)
    Nchain = len(idx)
    Nsample = np.array([np.sum(Nseed == idx_val) for idx_val in idx])

    samples_seed = samples[idx, :]
    logL_seed = logL_j1[idx]

    samples_jp1 = [None] * Nchain
    logL_jp1 = [None] * Nchain
    samples_cands = [None] * Nchain
    for i in range(Nchain):
        samples_lead = samples_seed[i, :]
        logL_lead = logL_seed[i]
        samples_jp1[i] = np.zeros((Nsample[i], Ndim))
        logL_jp1[i] = np.zeros(Nsample[i])
        for ii in range(Nsample[i]):
            # Generate a candidate sample
            while True:
                samples_cand = multivariate_normal.rvs(mean=samples_lead, cov=S_j)
                if prior_pdf(samples_cand, prior):
                    break

            logL_cand = cal_likelihood_s(f_obs,samples_cand)

            # Compute the acceptance ratio
            alpha = np.exp(p_jj * (logL_cand - logL_lead)) * prior_pdf(samples_cand, prior) / prior_pdf(samples_lead, prior)

            # Rejection step
            if np.random.rand() <= min(1, alpha):
                samples_jp1[i][ii, :] = samples_cand
                logL_jp1[i][ii] = logL_cand
                samples_lead = samples_cand
                logL_lead = logL_cand
            else:
                samples_jp1[i][ii, :] = samples_lead
                logL_jp1[i][ii] = logL_lead

    samples_j = np.vstack(samples_jp1)
    logL_jp = np.hstack(logL_jp1)
    return samples_j, logL_jp

def tempering_parameter(p_j1, logL_j1):
    lb_p = p_j1
    ub_p = 2
    log_fD_T_adjust = np.max(logL_j1)
    while (ub_p - lb_p) / ((ub_p + lb_p) / 2) > 1e-6:
        b = (ub_p + lb_p) / 2
        w_jk = np.exp((b - p_j1) * (logL_j1 - log_fD_T_adjust))
        cov_w = np.std(w_jk) / np.mean(w_jk)
        if cov_w > 1:
            ub_p = b
        else:
            lb_p = b
    return min(1, b)
def cal_wj_Sj(d_p_j, logL_j1, log_adjust, samples_j1, Ns, Ndim, beta2):
    w_j = np.exp((d_p_j) * (logL_j1 - log_adjust))
    w_j /= np.sum(w_j)
    samples_wm = np.dot(samples_j1.T, w_j)  # sample weighted mean
    S_j = np.zeros((Ndim, Ndim))  # covariance matrix
    for i in range(Ns):
        diff = (samples_j1[i, :].T - samples_wm)
        S_j += beta2 * w_j[i] * np.outer(diff, diff)
    S_j = (S_j + S_j.T) / 2  # enforce symmetry
    return w_j, S_j


def samples_add(w_pvc, samples_j, N_add, Simulator):
    non_zero_indices = np.nonzero(w_pvc)[0]  # Get indices of non-zero entries
    non_zero_count = len(non_zero_indices)
    # Adjust N_add to be the minimum of N_add and the number of non-zero entries
    N_add_adjusted = min(N_add, non_zero_count)
    # print('N_add_adjusted', N_add_adjusted)
    # p=w_pvc를 확률로 사용하여 N_mc개의 샘플을 비복원으로 리샘플링
    resampled_indices = np.random.choice(len(samples_j), N_add_adjusted, replace=False, p=w_pvc)
    # 리샘플된 인덱스를 기반으로 새로운 샘플 배열 생성
    resampled_samples = samples_j[resampled_indices, :]
    Y = Simulator(resampled_samples)
    return resampled_samples, Y, N_add_adjusted

def samples_add2(w_pvc, samples_j, N_add, Simulator):
    non_zero_indices = np.nonzero(w_pvc)[0]  # Get indices of non-zero entries
    non_zero_count = len(non_zero_indices)
    # Adjust N_add to be the minimum of N_add and the number of non-zero entries
    N_add_adjusted = min(N_add, non_zero_count)
    # print('N_add_adjusted', N_add_adjusted)
    # p=w_pvc를 확률로 사용하여 N_mc개의 샘플을 비복원으로 리샘플링
    resampled_indices = np.random.choice(len(samples_j), N_add_adjusted, replace=False, p=w_pvc)
    # 리샘플된 인덱스를 기반으로 새로운 샘플 배열 생성
    resampled_samples = samples_j[resampled_indices, :]
    Y = Simulator(resampled_samples)
    return resampled_samples, Y, N_add_adjusted

def train_mvae(net, Dataset,Dataset_val, device, save_torch, patience, batch_size, LR, trained_model=None ):
    train_data = Dataset[0]
    #train_data += np.random.normal(0, 0.02, size=train_data.shape)
    train_label = Dataset[1]
    valid_data = Dataset_val[0]
    valid_label = Dataset_val[1]

    train_loader = data2loader(train_data, train_label, batch_size=batch_size, shuffle=True)
    valid_loader = data2loader(valid_data, valid_label, batch_size=batch_size, shuffle=True)
    train_sequential_MVAE(net, patience, train_loader, valid_loader, device, save_torch,
                                 check_index=[0, 1, 2, 3], learning_r=LR, trained_model=trained_model)


def train_vae(net, Dataset, device, save_torch, epochs, LR, trained_model=None, valid_dataset = None):
    train_data = Dataset[0]
    #train_data += np.random.normal(0, 0.02, size=train_data.shape)
    if valid_dataset!= None:
        valid_data = valid_dataset[0]
    else:
        valid_data = train_data[-64:, :, :, :]
    train_loader = data2loader_l(train_data, batch_size=64, shuffle=True, droplast=False)
    valid_loader = data2loader_l(valid_data, batch_size=64, shuffle=True, droplast=False)
    VAE = train_sequential_VAE(net, epochs, train_loader, valid_loader, device, save_torch,  check_index=[0,1,2,3,4], learning_r=LR, trained_model=trained_model)
    return VAE

class CustomDataset(Dataset):
    def __init__(self, data):
        self.x_data = data.tolist()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        return x

class CustomDataset_label(Dataset):
    def __init__(self, data, label):
        self.x_data = data.tolist()
        self.label_data = label.tolist()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        labels = torch.FloatTensor(self.label_data[idx])
        return x, labels
def data2loader(train_data, train_label, batch_size, shuffle=True, droplast=True):
    dataset = CustomDataset_label(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=droplast)  # num_workers=2코어수
    del dataset
    return train_loader


def data2loader_l(train_label, batch_size, shuffle=True, droplast=False):
    dataset = CustomDataset(train_label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=droplast)  # num_workers=2코어수
    del dataset
    return train_loader


def initialize_vae(Y_exp, enc, device, Dataset):
    data = Dataset[0]
    valid_loader = data2loader_l(data, batch_size=64, shuffle=False, droplast=False)
    z_p = []
    for counter, wave in enumerate(valid_loader):
        wave = wave.to(device)
        zz_p, mu, var = enc(wave)
        z_p += zz_p.cpu().detach().numpy().tolist()

    z_p = np.array(z_p)
    z_mu = np.mean(z_p, axis=0)
    z_s2 = np.var(z_p, axis=0)
    z = [z_mu, z_s2]

    z_D = []
    Y_exp = torch.FloatTensor(Y_exp).to(device)
    _, z_D_mu, z_D_var = enc(Y_exp)
    z_D_mu = z_D_mu[0].cpu().detach().numpy()
    z_D_s2 = z_D_var[0].cpu().detach().numpy()
    z_D = [z_D_mu, z_D_s2]
    return z_D, z

def initialize(Y_exp, enc, device, Dataset):
    data = Dataset[0]
    labels = Dataset[1]
    valid_loader = data2loader(data, labels, batch_size=64, shuffle=False, droplast=False)
    z_p = []
    for counter, (wave, label) in enumerate(valid_loader):
        wave = wave.to(device)
        zz_p, mu, var = enc(wave)
        z_p += zz_p.cpu().detach().numpy().tolist()

    z_p = np.array(z_p)
    z_mu = np.mean(z_p, axis=0)
    z_s2 = np.var(z_p, axis=0)
    z = [z_mu, z_s2]

    z_D = []
    Y_exp = torch.FloatTensor(Y_exp).to(device)
    _, z_D_mu, z_D_var = enc(Y_exp)
    z_D_mu = z_D_mu[0].cpu().detach().numpy()
    z_D_s2 = z_D_var[0].cpu().detach().numpy()
    z_D = [z_D_mu, z_D_s2]
    return z_D, z

def cal_log_dt(z_D, z, beta):
    z_dim = len(z_D[0])
    dt = 0.
    for i in range(z_dim):
        mu1 = z_D[0][i]
        mu3 = z[0][i]
        vr1 = z_D[1][i]
        vr3 = z[1][i]

        c0 = np.sqrt(1 / ((2 * np.pi)**(beta+1) * (vr1 ** beta) * vr3))
        a = (beta / (2 * vr1)) + (1 / (2 * vr3))
        b = -(mu1* beta / (vr1 )) - (mu3 / vr3)
        c = ( (mu1 ** 2) * beta / (2 * vr1 )) + (mu3 ** 2 / (2 * vr3))
        dt += np.log(c0) + ((b ** 2 - 4 * a * c) / (4 * a)) + np.log(np.sqrt(np.pi / a))
    return dt


def cal_loglikelihood(z_D, z_theta, z):
    z_dim = len(z_D[0])
    LH = 0.
    for i in range(z_dim):
        mu1 = z_D[0][i]
        mu2 = z_theta[0][i]
        mu3 = z[0][i]
        vr1 = z_D[1][i]
        vr2 = z_theta[1][i]
        vr3 = z[1][i]

        c0 = np.sqrt(vr3 / (2 * np.pi * vr1 * vr2))
        a = (1 / (2 * vr1)) + (1 / (2 * vr2)) - (1 / (2 * vr3))
        b = -(mu1 / vr1) - (mu2 / vr2) + (mu3 / vr3)
        c = (mu1 ** 2 / (2 * vr1)) + (mu2 ** 2 / (2 * vr2)) - (mu3 ** 2 / (2 * vr3))
        LH += np.log(c0) + ((b ** 2 - 4 * a * c) / (4 * a)) + np.log(np.sqrt(np.pi / a))

    return LH


def cal_integ(z_theta, z):
    z_dim = len(z_theta[0])
    value = 0.
    for i in range(z_dim):
        mu2 = z_theta[0][i]
        mu3 = z[0][i]
        vr2 = z_theta[1][i]
        vr3 = z[1][i]

        c0 = np.sqrt(vr3 / vr2)
        a = (1 / (2 * vr2)) - (1 / (2 * vr3))
        b = - (mu2 / vr2) + (mu3 / vr3)
        c = (mu2 ** 2 / (2 * vr2)) - (mu3 ** 2 / (2 * vr3))
        if a < 0:
            return -99
        else:
            value += np.log(c0) + ((b ** 2 - 4 * a * c) / (4 * a)) + np.log(np.sqrt(np.pi / a))

    return value


def log_likelihood(th, enc, data_mean, yfun, Neig, data, dof, device, z_D, z):
    print(Neig, data, dof, th)
    ysim = yfun(Neig, data.astype('float32'), dof, th.astype('float32'))
    ysim = np.ascontiguousarray(np.array(ysim)).T
    # Downsample and subtract mean
    ysim = ysim[:, ::2] - data_mean[:, ::2]  # Match downsampled data size
    nstep = 1024
    # Downsampled data to fit the required shape
    y_sim = np.zeros((ysim.shape[0], nstep))

    # Align downsampled data on the right
    y_sim[:, -ysim.shape[1]:] = ysim
    y_sim = torch.tensor(y_sim).to(device)
    # net에 넣기
    _, z_theta_mu, z_theta_var = enc(y_sim.view(1, y_sim.shape[0], 1, y_sim.shape[1]))
    # print(theta, z_theta_mu,z_theta_var)
    z_theta = [z_theta_mu[0].cpu().detach().numpy(), z_theta_var[0].cpu().detach().numpy()]
    # 적분
    logL = cal_loglikelihood(z_D, z_theta, z)

    return logL



def B_dis_output(sample_1, sample_2, Nbin):
    """
    Return the Bhattacharrya distance between y_sim and y_exp based on the binning algorithm

    Parameters:
    sample_1 -- matrix of random samples
    sample_2 -- matrix of random samples
    Nbin     -- number of bins

    Returns:
    bd -- the Bhattacharrya distance (scalar value)
    """

    # row of the sample corresponds to the Monte Carlo sample
    # column of the sample corresponds to each output
    Nrow_1, Ncolumn_1 = sample_1.shape
    Nrow_2, Ncolumn_2 = sample_2.shape

    if Ncolumn_1 != Ncolumn_2:
        raise ValueError('Dim of the two samples must be equal to each other!')

    max_1 = np.max(sample_1, axis=0)
    min_1 = np.min(sample_1, axis=0)
    max_2 = np.max(sample_2, axis=0)
    min_2 = np.min(sample_2, axis=0)

    ub = np.max(np.vstack([max_1, max_2]), axis=0)
    lb = np.min(np.vstack([min_1, min_2]), axis=0)

    # following treatment is necessary to avoid unexpected error
    ub = ub + np.abs(ub * 0.0001)
    lb = lb - np.abs(lb * 0.0001)

    bins = [np.linspace(lb[i], ub[i], Nbin + 1) for i in range(Ncolumn_1)]

    count_1, _ = np.histogramdd(sample_1, bins=bins)
    count_2, _ = np.histogramdd(sample_2, bins=bins)

    count_ratio_1 = count_1 / Nrow_1
    count_ratio_2 = count_2 / Nrow_2

    # Flattening the counts for easier computation
    count_ratio_1_flat = count_ratio_1.flatten()
    count_ratio_2_flat = count_ratio_2.flatten()

    dis = np.sqrt(count_ratio_1_flat * count_ratio_2_flat)
    bd = -np.log(np.sum(dis))

    return bd