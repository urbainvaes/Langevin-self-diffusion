import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', size=16)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(15, 8))

control, γ, δ = "underdamped", .0001, 0
filename = f"data/time/{control}-gamma={γ}-delta={δ}.txt"
data = np.loadtxt(filename, skiprows=1)

# Check julia code
ts  = data[:, 0]
D11 = data[:, 1]
D22 = data[:, 2]
σ11 = data[:, 3]
σ22 = data[:, 4]
D11_control = data[:, 5]
D22_control = data[:, 6]
σ11_control = data[:, 7]
σ22_control = data[:, 8]

plt.ion()
fig, ax = plt.subplots(1, 2)
ax[0].set_xlabel('$t$')
ax[0].set_title(r"Sample mean")
ax[0].plot(ts, D11, ".-", label="MC/No control")
ax[0].plot(ts, D11_control, ".-", label="MC/Underdamped")
ax[0].set_ylim([2500, 3300])
ax[0].legend()
ax[0].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title(r'Standard deviation')
ax[1].plot(ts, σ11, ".-", label="MC/No control")
ax[1].plot(ts, σ11_control, ".-", label="MC/Underdamped")
ax[1].legend()
ax[1].grid()
fig.savefig("time.pdf", bbox_inches='tight')
plt.show()
