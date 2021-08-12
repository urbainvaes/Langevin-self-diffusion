import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', size=16)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
# matplotlib.rc('figure', figsize=(15, 8))
matplotlib.rc('figure', figsize=(12, 6))

γ, ν = .0001, 2

# UNDERDAMPED {{{1
filename = f"data-gle/time/gle-gamma={γ}.txt"
data = np.loadtxt(filename, skiprows=1)

# Check julia code
underdamped_ts  = data[:, 0]
underdamped_D = data[:, 1]
underdamped_σ = data[:, 2]
underdamped_D_control = data[:, 3]
underdamped_σ_control = data[:, 4]

# Number of particles
nptotal = 5000
limit = 3301.106464614402

plt.ion()
nsigma = 3
fig, ax = plt.subplots()
ax.set_xlabel('$t$')
ax.set_title(r"Mean $\pm$ 3 standard deviations of $u(t)$ and $v(t)$ for $\gamma = 10^{-4}$")
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
ax.set_xlim([0, 100/γ])
ax.set_ylim([2000, 4000])
ax.legend()
ax.grid()
fig.savefig("time-gle.pdf", bbox_inches='tight')
plt.show()
