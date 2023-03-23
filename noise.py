import numpy as np


def wgn(x, snr):
    # x: 输入振动信号形状(a,b); a:样本数; b样本长度
    # snr: 噪声强度，如-8,4,2,0,2,4,8
    # snr=0表示噪声等于振动信号
    # snr>0表示振动信号强于噪声，→∞表示无噪声
    # snr<0表示噪声强于振动信号，→-∞表示无信号
    Ps = np.sum(abs(x) ** 2, axis=1) / len(x)
    Pn = Ps / (10 ** ((snr / 10)))
    row, columns = x.shape
    Pn = np.repeat(Pn.reshape(-1, 1), columns, axis=1)

    noise = np.random.randn(row, columns) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise
