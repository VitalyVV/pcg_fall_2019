from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import perl
import sys


def interpolation(noise, screen_size, res_size):
    tr = np.sqrt(res_size).astype('int64')
    data = noise[:res_size].reshape(tr, tr)
    screen = np.random.rand(screen_size, screen_size)
    res = convolve(screen, data)
    return res

def load_openbci_file(filename, ch_names=None, skiprows=0, max_rows=0):
    """
    Load data from OpenBCI file into mne RawArray for later use
    :param filename: filename for reading in form of relative path from working directory
    :param ch_names: dictionary having all or some channels like this:
            {"fp1":1, "fp2":2, "c3":3, "c4":4, "o1":5, "o2":6, "p3":7, "p4":8}
            Key specifies position on head using 10-20 standard and
            Value referring to channel number on Cyton BCI board
    :return: yes
    """
    if ch_names is None:
        ch_names = {"fp1":0, "fp2":1, "c3":2, "c4":3, "o1":4, "o2":5, "p3":6, "p4":7}

    converter = {i: (float if i < 12 else lambda x: str(x).split(".")[1][:-1])
                 for i in range(0, 13)}

    data = np.loadtxt(filename, comments="%", delimiter=",", converters=converter, usecols=tuple(range(1,3))).T

    data = data[list(ch_names.values())[:2]]
    return data[:2]


def load_art_data(path, pattern):
    """
    Loading meditation data from path.
    :param path: path
    :return: numpy array with 2 coefficients
    """
    # Specifying files directory, select all the files from there which is txt
    datadir = Path(path).glob(pattern)
    # Transferring generator into array of file paths
    return [load_openbci_file(x) for x in datadir]


def main():
    fig = plt.figure()
    window_size = 724

    datas = load_art_data('art_data', '*14*Meditation*')

    datas = np.array(list(map(lambda x: x.mean(axis=0), datas)))
    print(datas)
    datas = datas[0]

    size = 255
    datas = np.array([datas[j * size:j * size + size] for j in range(len(datas) // size)])

    ax = plt.axes(xlim=(0, window_size), ylim=(0, window_size))
    a = np.random.random((window_size, window_size))
    im = plt.imshow(a, interpolation='none')

    pnf = perl.PerlinNoiseFactory(1, unbias=True)

    # initialization function: plot the background of each frame
    def init():
        im.set_data(np.random.random((window_size, window_size)))
        return [im]

    # animation function.  This is called sequentially
    def animate(i):
        kernel_size = 225
        data = datas[i]
        noise = np.array(list(map(lambda x: pnf(x), data)))
        res = interpolation(noise, window_size, kernel_size)
        im.set_array(res)

        return [im]

    anim = FuncAnimation(fig, animate, init_func=init,
                                   frames=10000, interval=20, blit=True)
    # anim.save('basic_animation.mp4', writer='ffmpeg')
    plt.show()

if __name__ == '__main__':
    main()