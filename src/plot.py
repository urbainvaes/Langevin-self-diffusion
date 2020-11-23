import matplotlib.pyplot as plt
import numpy as np

γs = np.loadtxt("data/data_γs.txt")
δs = np.loadtxt("data/data_δs.txt")

D11_wi = np.loadtxt("data/data_D11_wi.txt")
σ11_wi = np.loadtxt("data/data_σ11_wi.txt")
D11_wo = np.loadtxt("data/data_D11_wo.txt")
σ11_wo = np.loadtxt("data/data_σ11_wo.txt")

fig, ax = plt.subplots()
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs):
    coeffs = np.polyfit(np.log10(γs[:7]), np.log10(D11_wo[:7, iδ]), deg=1)
    ax.loglog(γs, D11_wo[:, iδ], ".-", 
            label="$\delta = {}, D \propto \gamma^{{ {:.2f} }}$".format(δ, coeffs[0]))
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs):
    ax.loglog(γs, D11_wi[:, iδ], ".--")
ax.set_xlabel("$γ$")
plt.legend()
plt.savefig("diffusion.pdf")
plt.show()

for iδ, δ in enumerate(δs):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(None)
    coeffs = np.polyfit(np.log10(γs[:7]), np.log10(D11_wo[:7, iδ]), deg=1)
    ax.loglog(γs, D11_wo[:, iδ], ".-")
    ax.loglog(γs, D11_wi[:, iδ], ".--")
    ax.loglog(γs, 10**coeffs[1] * γs**coeffs[0], "-", label="$\gamma^{{ {} }}$".format(coeffs[0]))
    ax.set_xlabel("$γ$")
    plt.legend()
    plt.show()

fig, ax = plt.subplots()
for iδ, δ in enumerate(δs):
    fig, ax = plt.subplots()
    ax.loglog(γs, σ11_wo[:, iδ], ".-")
    ax.set_prop_cycle(None)
    ax.loglog(γs, σ11_wi[:, iδ], ".--")
    ax.set_xlabel("$γ$")
    plt.show()
