'''
Majority of the code presented in this file is taken from https://github.com/ModelDBRepository/266508/tree/master
please refer to their paper:
Telenczuk, Bartosz, Maria Telenczuk, and Alain Destexhe. "A kernel-based method to calculate local field potentials from networks of spiking neurons." Journal of neuroscience methods 344 (2020): 108871.

Can also be found here: https://www.sciencedirect.com/science/article/pii/S0165027020302946?via%3Dihub

Table of respective amplitudes:
Layer   amp_i    amp_e
deep    -2       -1.6
soma    30       4.8
sup     -12      2.4
surf    3        -0.8

amp_e = -0.16	# exc uLFP amplitude (deep layer)
amp_i = -0.2	# inh uLFP amplitude (deep layer)

amp_e = 0.48  # exc uLFP amplitude (soma layer)
amp_i = 3  # inh uLFP amplitude (soma layer)

amp_e = 0.24	# exc uLFP amplitude (superficial layer)
amp_i = -1.2	# inh uLFP amplitude (superficial layer)

amp_e = -0.08	# exc uLFP amplitude (surface)
amp_i = 0.3	# inh uLFP amplitude (surface)


'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt



def calc_lfp_classic(cells, sigma, delay, lfp_time):
    """Calculate LFP from cells"""

    # this is a vectorised computation and as such it might be memory hungry
    # for long LFP series/large number of cells it may be more efficient to calculate it through looping

    spt = cells["time"]
    cid = cells["cellid"]
    
    t = lfp_time[:, None] - (delay[None, cid] + spt[None, :])
    kernel_contribs = amp[None, cid] * f_temporal_kernel(t, sigma)
    lfp = kernel_contribs.sum(1)
    return lfp


def f_temporal_kernel(t, tau):
    """function defining temporal part of the kernel"""
    return np.exp(-(t ** 2) / tau)


def calc_lfp_gauss(cells, sigma, delay, lfp_time):
    """Calculate LFP from cells"""

    # this is a vectorised computation and as such it might be memory hungry
    # for long LFP series/large number of cells it may be more efficient to calculate it through looping

    spt = cells["time"]
    cid = cells["cellid"]
    
    t = lfp_time[:, None] - (delay[None, cid] + spt[None, :])
    kernel_contribs = amp[None, cid] * f_temporal_kernel(t, sigma)
    lfp = kernel_contribs.sum(1)
    return lfp
        

dtype = {"names": ["cellid", "time"], "formats": ["i4", "f8"]}
inh_cells = np.loadtxt("analysis/brunel_inh.txt", dtype=dtype)
exc_cells = np.loadtxt("analysis/brunel_exc.txt", dtype=dtype)

Ne = np.max(exc_cells["cellid"])
Ni = np.max(inh_cells["cellid"])
N = (Ne+1) + (Ni+1) 

inh_cells["cellid"] += np.max(exc_cells["cellid"]) # for inhibitory cells, ids start from Ne

# 3. distribute cells in a 2D grid
xmax = 0.2  # size of the array (in mm)
ymax = 0.2
X, Y = np.random.rand(2, N) * np.array([[xmax, ymax]]).T


# adjust time and convert to ms
inh_cells["time"] = (inh_cells["time"] - np.min(exc_cells["time"])) * 1000
exc_cells["time"] = (exc_cells["time"] - np.min(exc_cells["time"])) * 1000


dt = 0.1  # time resolution
npts = int((np.max(exc_cells["time"] )-np.min(exc_cells["time"] )) / dt)  # nb points in LFP vector

xe, ye = xmax / 2, ymax / 2  # coordinates of electrode

va = 200  # axonal velocity (mm/sec)
lambda_ = 0.2  # space constant (mm)
dur = 100  # total duration of LFP waveform
nlfp = int(dur / dt)  # nb of LFP pts
amp_e = 0.7  # uLFP amplitude for exc cells
amp_i = -3.4  # uLFP amplitude for inh cells
sig_i = 2.1  # std-dev of ihibition (in ms)
sig_e = 1.5 * sig_i  # std-dev for excitation

amp_e = 0.48  # exc uLFP amplitude (soma layer)
amp_i = 3  # inh uLFP amplitude (soma layer)

dist = np.sqrt((X - xe) ** 2 + (Y - ye) ** 2)  # distance to  electrode in mm
delay = 10.4 + dist / va  # delay to peak (in ms)
amp = np.exp(-dist / lambda_)
amp[:Ne] *= amp_i
amp[Ne:] *= amp_e

s_e = 2 * sig_e * sig_e
s_i = 2 * sig_i * sig_i

lfp_time = np.arange(npts) * dt
lfp_inh = calc_lfp_gauss(inh_cells, s_i, delay, lfp_time)
lfp_exc = calc_lfp_gauss(exc_cells, s_e, delay, lfp_time)
total_lfp = lfp_inh + lfp_exc


Nstp = 10  # step cell to draw
tick_size = 5
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(exc_cells[::Nstp]["time"], exc_cells[::Nstp]["cellid"], "|", ms=tick_size)
axes[0].plot(inh_cells[::Nstp]["time"], inh_cells[::Nstp]["cellid"], "|", ms=tick_size)

axes[1].plot(lfp_time, total_lfp)
axes[1].set_xlabel("time, ms")
axes[1].set_xlim(0, np.max(exc_cells["time"]))

# prettify graph
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()
# plt.savefig("demo_lfp_kernel.pdf")
