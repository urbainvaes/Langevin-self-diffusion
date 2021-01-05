import matplotlib.pyplot as plt
import numpy as np

γs = [.00001, .0000215, .0000464, .0001, .000215, .000464, .001, .00215, .00464,
      .01, .0215, .0464, .1, .215, .464, 1.]
δs = [.0, .04, .08, .16, .32, .64]

γs_galerkin = np.loadtxt("data/data-galerkin-γs.txt")
δs_galerkin = np.loadtxt("data/data-galerkin-δs.txt")
D11_wi_galerkin = np.loadtxt("data/data-galerkin-D11_wi.txt")
σ11_wi_galerkin = np.loadtxt("data/data-galerkin-σ11_wi.txt")
D11_wo_galerkin = np.loadtxt("data/data-galerkin-D11_wo.txt")
σ11_wo_galerkin = np.loadtxt("data/data-galerkin-σ11_wo.txt")

γs_underdamped = np.loadtxt("data/data-underdamped-γs.txt")
δs_underdamped = np.loadtxt("data/data-underdamped-δs.txt")
D11_wi_underdamped = np.loadtxt("data/data-underdamped-D11_wi.txt")
σ11_wi_underdamped = np.loadtxt("data/data-underdamped-σ11_wi.txt")
D11_wo_underdamped = np.loadtxt("data/data-underdamped-D11_wo.txt")
σ11_wo_underdamped = np.loadtxt("data/data-underdamped-σ11_wo.txt")

if δs_galerkin.shape == ():
    δs_galerkin.shape = (1,)
    D11_wi_galerkin.shape = (len(γs_galerkin), 1)
    σ11_wi_galerkin.shape = (len(γs_galerkin), 1)
    D11_wo_galerkin.shape = (len(γs_galerkin), 1)
    σ11_wo_galerkin.shape = (len(γs_galerkin), 1)

if δs_underdamped.shape == ():
    δs_underdamped.shape = (1,)
    D11_wi_underdamped.shape = (len(γs_underdamped), 1)
    σ11_wi_underdamped.shape = (len(γs_underdamped), 1)
    D11_wo_underdamped.shape = (len(γs_underdamped), 1)
    σ11_wo_underdamped.shape = (len(γs_underdamped), 1)

# 1D
fig, ax = plt.subplots()
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs_galerkin):
    if δ != 0:
        continue
    coeffs_galerkin = np.polyfit(np.log10(γs[:7]), np.log10(D11_wo_galerkin[:7, iδ]), deg=1)
    coeffs_underdamped = np.polyfit(np.log10(γs[:7]), np.log10(D11_wo_underdamped[:7, iδ]), deg=1)
    ax.loglog(γs, D11_wo_galerkin[:, iδ], ".-",
            label="$\delta = {}, D \propto \gamma^{{ {:.2f} }}$".format(δ, coeffs_galerkin[0]))
    ax.loglog(γs, D11_wo_underdamped[:, iδ], ".-",
            label="$\delta = {}, D \propto \gamma^{{ {:.2f} }}$".format(δ, coeffs_underdamped[0]))
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs_galerkin):
    if δ != 0:
        continue
    ax.loglog(γs, D11_wi_galerkin[:, iδ], ".--")
    ax.loglog(γs, D11_wi_underdamped[:, iδ], ".--")
ax.set_xlabel("$γ$")
plt.legend()
plt.savefig("diffusion.pdf")
plt.show()

# 1D
fig, ax = plt.subplots()
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs_galerkin):
    if δ != 0:
        continue
    ax.loglog(γs, σ11_wo_galerkin[:, iδ])
    ax.loglog(γs, σ11_wo_underdamped[:, iδ])
for iδ, δ in enumerate(δs_galerkin):
    if δ != 0:
        continue
    ax.loglog(γs, σ11_wi_galerkin[:, iδ], ".--")
    ax.loglog(γs, σ11_wi_underdamped[:, iδ], ".--")
ax.set_xlabel("$γ$")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs):
    if δ < 0:
        continue
    coeffs = np.polyfit(np.log10(γs[:7]), np.log10(D11_wo[:7, iδ]), deg=1)
    ax.loglog(γs, D11_wo[:, iδ], ".-",
            label="$\delta = {}, D \propto \gamma^{{ {:.2f} }}$".format(δ, coeffs[0]))
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs):
    if δ < 0:
        continue
    ax.loglog(γs, D11_wi[:, iδ], ".--")
ax.set_xlabel("$γ$")
plt.legend()
plt.savefig("diffusion.pdf")
plt.show()

# for iδ, δ in enumerate(δs):
#     fig, ax = plt.subplots()
#     ax.set_prop_cycle(None)
#     coeffs = np.polyfit(np.log10(γs[:7]), np.log10(D11_wo[:7, iδ]), deg=1)
#     ax.loglog(γs, D11_wo[:, iδ], ".-")
#     ax.loglog(γs, D11_wi[:, iδ], ".--")
#     ax.loglog(γs, 10**coeffs[1] * γs**coeffs[0], "-", label="$\gamma^{{ {} }}$".format(coeffs[0]))
#     ax.set_xlabel("$γ$")
#     plt.legend()
#     plt.show()

fig, ax = plt.subplots()
for iδ, δ in enumerate(δs):
    if δ < 0:
        continue
    fig, ax = plt.subplots()
    ax.set_title("$\delta = {}$".format(δ))
    ax.loglog(γs, σ11_wo[:, iδ], ".-")
    ax.set_prop_cycle(None)
    ax.loglog(γs, σ11_wi[:, iδ], ".--")
    # ax.plot(γs, σ11_wo[:, iδ]/σ11_wi[:, iδ], ".-")
    ax.set_xlabel("$γ$")
    plt.show()
