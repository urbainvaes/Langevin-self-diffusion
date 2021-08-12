import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', size=16)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
# matplotlib.rc('figure', figsize=(15, 8))
matplotlib.rc('figure', figsize=(12, 6))
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.ion()

ν = 2

sample_size = {
        1e-3: 5000,
        1e-4: 5000,
        1e-5: 500,
        }

def get_diff(γ, plot=True):
    filename = f"data-gle/time/gle-gamma={γ}.txt"
    data = np.loadtxt(filename, skiprows=1)

    # Check julia code
    underdamped_ts  = data[:, 0]
    underdamped_D = data[:, 1]
    underdamped_σ = data[:, 2]
    underdamped_D_control = data[:, 3]
    underdamped_σ_control = data[:, 4]

    # Number of particles
    nptotal = sample_size[γ]
    limit = .3301106464614402/γ

    if  plot:
        nsigma = 3
        fig, ax = plt.subplots()
        ax.set_xlabel('$t$')
        ax.set_title(r"MC estimation of $\mathbf{E}u(t)$ and $\mathbf{E}v(t)$ for $\gamma = 10^{" + str(int(np.log10(γ))) +  r"},$"
                     + r" with $\pm 3 \sigma$ confidence interval")
        ax.plot(underdamped_ts, underdamped_D, ".-", c='red', label="MC/No control")
        ax.plot(underdamped_ts, underdamped_D_control, ".-", c='green', label="MC/Underdamped")
        ax.plot(underdamped_ts, 0*underdamped_ts + limit, "-", c='blue', lw=3, label="Conjectured limit")
        ax.fill_between(underdamped_ts,
                        underdamped_D - nsigma*underdamped_σ/np.sqrt(nptotal),
                        underdamped_D + nsigma*underdamped_σ/np.sqrt(nptotal),
                        color='red', alpha=.2)
        ax.fill_between(underdamped_ts,
                        underdamped_D_control - nsigma*underdamped_σ_control/np.sqrt(nptotal),
                        underdamped_D_control + nsigma*underdamped_σ_control/np.sqrt(nptotal),
                        color='green', alpha=.2)
        ax.set_xlim([underdamped_ts[0], underdamped_ts[-1]])
        ax.set_ylim([.2/γ, .4/γ])
        ax.legend()
        ax.grid()
        fig.savefig("time-gle" + str(int(np.log10(γ))) + ".pdf", bbox_inches='tight')
        plt.show()

    return underdamped_D_control[-1], underdamped_σ_control[-1]/np.sqrt(nptotal)

γs = np.array([1e-3, 1e-4, 1e-5])
Ds = np.zeros(len(γs))
σs = np.zeros(len(γs))

for i, γ in enumerate(γs):
    Ds[i], σs[i]  = get_diff(γ, plot=False)
fig, ax = plt.subplots()
ax.set_xlabel('$\gamma$')
ax.semilogx(γs, γs*Ds, '.-')
ax.set_xlim([γs[0], γs[-1]])
ax.fill_between(γs, γs*Ds - 3*γs*σs, γs*Ds + 3*γs*σs, color='green', alpha=.2)

get_diff(1e-3, plot=True)
get_diff(1e-4, plot=True)
get_diff(1e-5, plot=True)
