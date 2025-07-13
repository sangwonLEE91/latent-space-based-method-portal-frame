import numpy as np
import torch
import matplotlib.pyplot as plt
import copy

root = str(np.load('C:/root/root.npy'))
font = {'family': 'Times New Roman',
        'color': 'k',
        'weight': 'bold',
        'size': 10,
        'alpha': 0.5}

def train_sequential_MVAE(net,  patience, trainloader, valid_loader, device, save_root, check_index, learning_r=0.00001,
              trained_model=None):
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
        for counter, (wave, label) in enumerate(trainloader):
            wave = wave.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss, kl, kl12, rec = net.loss(wave, label)  # .view(-1, wave.shape[1], wave.shape[2], wave.shape[3])
            loss.backward()
            optimizer.step()

        running_loss = []
        for counter, (wave, label) in enumerate(valid_loader):
            wave = wave.to(device)
            label = label.to(device)
            loss, kl, kl12, rec = net.loss(wave, label)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/inf loss at VALID epoch {epoch}, batch {counter}: {loss.item()}")
                return
            running_loss.append(loss.cpu().detach().numpy())

        loss_average = np.average(running_loss)

        if best_loss > loss_average:
            torch.save(net.state_dict(), save_root)
            best_loss = copy.deepcopy(loss_average)
            if epoch > patience * 20:
                print(f'e{epoch}, ', end='')
            i_pain = 0
        else:
            i_pain += 1

        if epoch % 100000 == 0:
            # print(z_dim, beta, epochs)
            data_iter = iter(valid_loader)
            # 첫번째 배치를 추출합니다.
            wave, label = next(data_iter)

            # 필요하다면 device로 전송합니다.
            wave = wave.to(device)
            label = label.to(device)
            net.eval()
            rec1, _, rec2, _ = net(wave, label)
            net.waveshow(wave[check_index, :, :, :].cpu().detach().numpy(),
                         rec1[check_index, :, :, :].cpu().detach().numpy(),
                         rec2[check_index, :, :, :].cpu().detach().numpy(), check_index)
            net.train()

        # print('loss:', np.average(running_loss),"mean(var):",variance[epoch-1,0])
    net.eval()
    print(f'e{epoch} ')
    data_iter = iter(valid_loader)
    # 첫번째 배치를 추출합니다.
    wave, label = next(data_iter)

    # 필요하다면 device로 전송합니다.
    wave = wave.to(device)
    label = label.to(device)

    net.load_state_dict(torch.load(save_root, map_location=device, weights_only=True))
    net.eval()
    rec1, _, rec2, _ = net(wave, label)
    net.waveshow(wave[check_index, :, :, :].cpu().detach().numpy(),
                 rec1[check_index, :, :, :].cpu().detach().numpy(),
                 rec2[check_index, :, :, :].cpu().detach().numpy(), check_index)
    return


def train_sequential_VAE(net,  patience, trainloader, valid_loader, device, save_root, check_index, learning_r=0.00001,
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
