import matplotlib.pyplot as plt
import numpy as np

γs = np.loadtxt("data/data_γs.txt")
δs = np.loadtxt("data/data_δs.txt")

D11_wi = np.loadtxt("data/data_D11_wi.txt")
σ11_wi = np.loadtxt("data/data_σ11_wi.txt")
D11_wo = np.loadtxt("data/data_D11_wo.txt")
σ11_wo = np.loadtxt("data/data_σ11_wo.txt")

fig, ax = plt.subplots()
for iδ, δ in enumerate(δs):
    ax.loglog(γs, D11_wo[:, iδ], ".-")
    ax.set_prop_cycle(None)
    ax.loglog(γs, D11_wi[:, iδ], ".--")
    ax.set_xlabel("$γ$")
plt.show()

fig, ax = plt.subplots()
for iδ, δ in enumerate(δs):
    fig, ax = plt.subplots()
    ax.loglog(γs, σ11_wo[:, iδ], ".-")
    ax.set_prop_cycle(None)
    ax.loglog(γs, σ11_wi[:, iδ], ".--")
    ax.set_xlabel("$γ$")
    plt.show()
