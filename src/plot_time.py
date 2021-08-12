import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', size=16)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('lines', lw=3)
matplotlib.rc('figure', figsize=(15, 8))

γ, δ = .001, 0

# UNDERDAMPED {{{1
control = "underdamped"
filename = f"data/time/{control}-gamma={γ}-delta={δ}.txt"
data = np.loadtxt(filename, skiprows=1)

# Check julia code
underdamped_ts  = data[:, 0]
underdamped_D11 = data[:, 1]
underdamped_D22 = data[:, 2]
underdamped_σ11 = data[:, 3]
underdamped_σ22 = data[:, 4]
underdamped_D11_control = data[:, 5]
underdamped_D22_control = data[:, 6]
underdamped_σ11_control = data[:, 7]
underdamped_σ22_control = data[:, 8]

# GALERKIN {{{1
control = "galerkin"
filename = f"data/time/{control}-gamma={γ}-delta={δ}.txt"
data = np.loadtxt(filename, skiprows=1)

# Check julia code
galerkin_ts  = data[:, 0]
galerkin_D11 = data[:, 1]
galerkin_D22 = data[:, 2]
galerkin_σ11 = data[:, 3]
galerkin_σ22 = data[:, 4]
galerkin_D11_control = data[:, 5]
galerkin_D22_control = data[:, 6]
galerkin_σ11_control = data[:, 7]
galerkin_σ22_control = data[:, 8]

plt.ion()
fig, ax = plt.subplots(1, 2)
ax[0].set_xlabel('$t$')
ax[0].set_title(r"Sample mean")
ax[0].plot(underdamped_ts, underdamped_D11, "-", label="MC/No control")
ax[0].plot(underdamped_ts, underdamped_D11_control, "--", label="MC/Underdamped")
ax[0].plot(galerkin_ts, galerkin_D11_control, "-.", label="MC/Galerkin")
ax[0].set_xlim([0, 100/γ])
ax[0].set_ylim([250, 350])
ax[0].legend()
ax[0].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title(r'Standard deviation')
ax[1].plot(underdamped_ts, underdamped_σ11, "-", label="MC/No control")
ax[1].plot(underdamped_ts, underdamped_σ11_control, "--", label="MC/Underdamped")
ax[1].plot(galerkin_ts, galerkin_σ11_control, "-.", label="MC/Galerkin")
ax[1].set_xlim([0, 100/γ])
ax[1].legend()
ax[1].grid()
fig.savefig("time.pdf", bbox_inches='tight')
plt.show()
