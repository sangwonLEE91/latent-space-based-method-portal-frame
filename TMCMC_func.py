import numpy as np
from scipy.stats import multivariate_normal
import torch
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
    Y_loader = data2loader(samples, batch_size=128, shuffle=False)
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

    Y_loader = data2loader(Y_data, batch_size=128, shuffle=False, droplast=False)
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



def measure_of_fit(f,f_hat, ld=[1, 1, 1, 1, 1]):
    Mx = 0
    for i in range(len(f_hat)):
        Mx += ld[i] ** 2 * (f[i] ** 2 / f_hat[i] ** 2 - 1) ** 2
    return Mx



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



def train_vae(net, Dataset, device, save_torch, epochs, LR, trained_model=None, valid_dataset = None):
    train_data = Dataset[0]
    #train_data += np.random.normal(0, 0.02, size=train_data.shape)
    if valid_dataset!= None:
        valid_data = valid_dataset[0]
    else:
        valid_data = train_data[-64:, :, :, :]
    train_loader = data2loader(train_data, batch_size=64, shuffle=True, droplast=False)
    valid_loader = data2loader(valid_data, batch_size=64, shuffle=True, droplast=False)
    VAE = training(net, epochs, train_loader, valid_loader, device, save_torch,  check_index=[0,1,2,3,4], learning_r=LR, trained_model=trained_model)
    return VAE


def training(net,  patience, trainloader, valid_loader, device, save_root, check_index, learning_r=0.00001,
                          trained_model=None, pretrained=0):
    if trained_model != None:
        net.load_state_dict(torch.load(trained_model, map_location=device))
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_r)
    best_loss = 10 ** 30
    i_pain = 0
    epoch = 0
    while i_pain < patience:
        # print(epoch, best_loss)
        epoch += 1
        for counter, wave in enumerate(trainloader):
            wave = wave.to(device)
            optimizer.zero_grad()
            loss, kl, rec = net.loss(wave)  # .view(-1, wave.shape[1], wave.shape[2], wave.shape[3])
            loss.backward()
            optimizer.step()

        running_loss = []
        for counter, wave in enumerate(valid_loader):
            wave = wave.to(device)
            loss, kl, rec = net.loss(wave)
            running_loss.append(loss.cpu().detach().numpy())

        loss_average = np.average(running_loss)

        if best_loss > loss_average:
            torch.save(net.state_dict(), save_root)
            best_loss = copy.deepcopy(loss_average)
            if epoch > 500:
                print(f'e{epoch}, ', end='')
            i_pain = 0
        else:
            i_pain += 1

        if epoch % 1000 == 0:
            # print(z_dim, beta, epochs)
            data_iter = iter(valid_loader)
            # 첫번째 배치를 추출합니다.
            wave = next(data_iter)

            # 필요하다면 device로 전송합니다.
            wave = wave.to(device)
            net.eval()
            rec1, _ = net(wave)
            net.waveshow(wave[check_index, :, :, :].cpu().detach().numpy(),
                         rec1[check_index, :, :, :].cpu().detach().numpy(),
                         check_index)
            net.train()

class CustomDataset(Dataset):
    def __init__(self, data):
        self.x_data = data.tolist()
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        return x

def data2loader(train_label, batch_size, shuffle=True, droplast=False):
    dataset = CustomDataset(train_label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=droplast)  # num_workers=2코어수
    del dataset
    return train_loader

def initialize_vae(Y_exp, enc, device, Dataset):
    data = Dataset[0]
    valid_loader = data2loader(data, batch_size=64, shuffle=False, droplast=False)
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
