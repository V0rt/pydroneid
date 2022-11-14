import uhd
import scope
import pmt
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import commpy

cp_lengths = [80, 72, 72, 72, 72, 72, 72, 80]


def plot(src: np.float32):
    plt.plot(src)
    # plt.pause(0.1)


def plot(src: np.complex64, ax=None):
    if ax:
        ax.plot(np.real(src))
        ax.plot(np.imag(src))
        ax.grid()
    else:
        plt.plot(np.real(src))
        plt.plot(np.imag(src))
        plt.grid()
    # plt.pause(0.1)


def const(src: np.complex64, ax=None):
    if ax:
        ax.plot(np.real(src), np.imag(src), '.')
        ax.grid()
    else:
        plt.plot(np.real(src), np.imag(src), '.')
        plt.grid()
    # plt.pause(0.1)


def cp_pos(symbol: int):
    cl = cp_lengths[symbol]
    cp = 0
    for i in range(symbol):
        cp += cp_lengths[i] + 1024
    return cl, cp


def get_symbol(burst: np.array, symbol: int):
    cl, cp = cp_pos(symbol)
    print("symbol", symbol, "pos", cp + cl)
    return burst[cp + cl: cp + cl + 1024]

# mapping
# i[211] = 0
# i[212] = 1
# i[512] = 0
# i[812] = 1
# i[813] = 0

def main():
    fft_size = 1024
    fig, ax = plt.subplots(4, 8)
    data = np.fromfile("coarse_time_sync_packet_30.72e6_or_15.36e6.cf32", dtype=np.complex64)[::2] # ::2 is for 30.72e6
    sync = np.fromfile("zc_600_time_conj.cf32", dtype=np.complex64)
    sync_var = np.var(sync)

    # zc1 = commpy.zcsequence(600, 601)
    # plot(zc1)
    # plt.show()

    orig1 = np.fromfile("zc_600_time.cf32", dtype=np.complex64)
    orig2 = np.fromfile("zc_147_time.cf32", dtype=np.complex64)
    orig1 = np.fft.fftshift(np.fft.fft(orig1))
    orig2 = np.fft.fftshift(np.fft.fft(orig2))

    plot(orig1, ax[0, 0])
    plot(orig2, ax[0, 4])

    corr = np.empty_like(data)
    for i in range(0, len(data) - fft_size):
        corr[i] = np.sum(data[i: i + fft_size] * sync)
        data_var = np.var(data[i: i + fft_size])
        corr[i] = corr[i] / np.sqrt(data_var * sync_var)
    pos = np.argmax(corr)

    offset = pos - 80 - (72 + 1024) * 2
    length = (80 + 1024) * 2 + (72 + 1024) * 6
    packet = data[offset: offset + length]
    # plot(packet, ax[0, 3])

    df_mean = 0
    for symbol in range(8):
        cl, cp = cp_pos(symbol)
        r = np.sum(np.conj(packet[cp:cp + cl]) * packet[cp + 1024:cp + 1024 + cl])
        df = np.arctan2(np.imag(r), np.real(r)) / 1024.0
        df_mean += df
        print("symbol", symbol, "df", df)
    df = df_mean / 8.0
    packet *= np.exp(-df * 1.j * np.arange(len(packet)))

    symbols = []
    for i in range(8):
        symbols.append(np.fft.fftshift(np.fft.fft(get_symbol(packet, i))))

    rx = symbols[2]
    plot(rx, ax[0, 1])

    h = orig1 / rx
    plot(h, ax[0, 2])

    rx2 = symbols[4]
    plot(rx2, ax[0, 5])
    h2 = orig2 / rx2
    plot(h2, ax[0, 6])

    symbols[:4] *= h
    symbols[4:8] *= h2
    # symbols = symbols * (h * h2) / np.sqrt(h * h + h2 * h2)

    for symbol in range(8):  # [0, 1, 3, 5, 6, 7]:
        time_domain = get_symbol(packet, symbol)
        plot(time_domain, ax[1, symbol])

        freq_domain = symbols[symbol]
        plot(freq_domain, ax[2, symbol])
        const(freq_domain, ax[3, symbol])

    plt.show()


if __name__ == '__main__':
    main()
