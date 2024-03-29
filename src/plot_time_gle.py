import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', size=22)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(15, 8))
matplotlib.rc('figure', figsize=(17, 8))
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.ion()

ν = 2
# lim = .3301106464614402
lim_gle = .3295
lim_lang = 0.30626213513957773

sample_size = {
        1e-0: 5000,
        1e-1: 5000,
        1e-2: 5000,
        1e-3: 5000,
        1e-4: 5000,
        }

def get_diff(γ, plot=True):
    filename = f"data_gle/time/gle-gamma={γ}.txt"
    data = np.loadtxt(filename, skiprows=1)

    # Check julia code
    underdamped_ts  = data[:, 0]
    underdamped_D = data[:, 1]
    underdamped_σ = data[:, 2]
    underdamped_D_control = data[:, 3]
    underdamped_σ_control = data[:, 4]

    # Number of particles
    if len(data[0]) > 5:
        nptotal = data[:, 5]
    else:
        nptotal = np.zeros(len(underdamped_ts)) + sample_size[γ]

    # Limit
    limit = lim_gle/γ

    if  plot:
        nsigma = 2
        fig, ax = plt.subplots()
        ax.set_xlabel('$t$')
        ax.set_title(r"MC estimation of $\mathbf{E}[u(t)]$ and $\mathbf{E}[v(t)]$ for $\gamma = 10^{" + str(int(np.log10(γ))) +  r"},$"
                     + r" with $\pm 3 s$ confidence interval")
        ax.plot(underdamped_ts, underdamped_D, "-", c='red', label="MC/No control")
        ax.plot(underdamped_ts, underdamped_D_control, "-", c='green', label="MC/Underdamped")
        ax.plot(underdamped_ts, 0*underdamped_ts + limit, "--", c='blue', lw=3, label="$D^{\\rm und}_{\\nu}/\\gamma$")
        ax.fill_between(underdamped_ts,
                        underdamped_D - nsigma*underdamped_σ/np.sqrt(nptotal[i]),
                        underdamped_D + nsigma*underdamped_σ/np.sqrt(nptotal[i]),
                        color='red', alpha=.2)
        ax.fill_between(underdamped_ts,
                        underdamped_D_control - nsigma*underdamped_σ_control/np.sqrt(nptotal[i]),
                        underdamped_D_control + nsigma*underdamped_σ_control/np.sqrt(nptotal[i]),
                        color='green', alpha=.2)
        ax.set_xlim([underdamped_ts[0], underdamped_ts[-1]])
        ax.set_ylim([.2/γ, .4/γ])
        if γ == 1e-5:
            ax.set_ylim([28000, 36000])
        ax.legend(loc="lower right")
        ax.grid()
        fig.savefig("time-gle" + str(int(np.log10(γ))) + ".pdf", bbox_inches='tight')
        plt.show()

    return underdamped_D_control[-1], underdamped_σ_control[-1]/np.sqrt(nptotal[-1])

γs = np.array([1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1e-0])
Ds = np.zeros(len(γs))
σs = np.zeros(len(γs))

for i, γ in enumerate(γs):
    Ds[i], σs[i]  = get_diff(γ, plot=False)
fig, ax = plt.subplots()
# ax.set_title(r"\textbf{Zoom:} effective diffusion coefficient for the GLE, multiplied by $\gamma$")
ax.set_title(r"Effective diffusion coefficient for the GLE, multiplied by $\gamma$")
ax.set_xlabel('$\gamma$')
ax.set_ylabel(r"$\gamma D^{\gamma, \nu}$")
ax.semilogx(γs, γs*Ds, 'o-', color='blue', label='MC/underdamped')
ax.fill_between(γs, γs*Ds - 3*γs*σs, γs*Ds + 3*γs*σs, color='blue', alpha=.2)
ax.semilogx(γs, 0*γs + lim_gle, 'g-', lw=3, label="Conjectured asymptotic limit for GLE")
ax.semilogx(γs, 0*γs + lim_lang, 'k--', lw=3, label="Asymptotic limit for Langevin")
ax.set_xlim([γs[0], γs[-1]])
# ax.set_ylim([.3, .4])
# ax.legend(loc="center right")
ax.legend(loc="upper left")
# fig.savefig("mobility_gle_zoom.pdf", bbox_inches='tight')
fig.savefig("mobility_gle.pdf", bbox_inches='tight')

get_diff(1e-3, plot=True)
get_diff(1e-4, plot=True)
get_diff(1e-5, plot=True)
