import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', size=16)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(12, 6))

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

fig, ax = plt.subplots()
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs_underdamped):
    iu = np.nonzero(D11_wo_underdamped[:, iδ])[0]
    γu = γs_underdamped[iu]
    yu = D11_wo_underdamped[:, iδ][iu]
    ifitu, cu = np.where(γu <= 1e-2)[0], [0]
    if len(ifitu) > 1:
        cu = np.polyfit(np.log10(γu[ifitu]), np.log10(yu[ifitu]), deg=1)
    ax.loglog(γu, yu, ".-",
              label="$\delta = {}, D \propto \gamma^{{ {:.2f} }}$".format(δ, cu[0]))
ax.set_prop_cycle(None)
# for iδ, δ in enumerate(δs_galerkin):
#     ig = np.nonzero(D11_wo_galerkin[:, iδ])[0]
#     γg = γs_galerkin[ig]
#     yg = D11_wo_galerkin[:, iδ][ig]
#     ax.loglog(γg, yg, ".-")
# ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs_underdamped):
    iu = np.nonzero(D11_wi_underdamped[:, iδ])[0]
    γu = γs_underdamped[iu]
    yu = D11_wo_underdamped[:, iδ][iu]
    ax.loglog(γu, yu, ".--")
ax.set_prop_cycle(None)
# for iδ, δ in enumerate(δs_galerkin):
#     ig = np.nonzero(D11_wi_galerkin[:, iδ])[0]
#     γg = γs_galerkin[ig]
#     yg = D11_wo_galerkin[:, iδ][ig]
#     ax.loglog(γg, yg, ".--")
ax.set_xlabel("$\gamma$")
plt.legend()
plt.savefig("diffusion.pdf")
plt.show()

for iδ, δ in enumerate(δs_galerkin):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(None)
    iu = np.nonzero(σ11_wo_underdamped[:, iδ])[0]
    γu = γs_underdamped[iu]
    yu = σ11_wo_underdamped[:, iδ][iu]
    ax.loglog(γu, σ11_wo_underdamped[:, iδ][iu], label="No variance reduction")
    ax.loglog(γu, σ11_wi_underdamped[:, iδ][iu], ".--", label="Underdamped control variate")
    ig = np.nonzero(σ11_wo_galerkin[:, iδ])[0]
    γg = γs_galerkin[ig]
    yg = σ11_wo_galerkin[:, iδ][ig]
    # ax.loglog(γg, σ11_wo_galerkin[:, iδ][ig])
    ax.loglog(γg, σ11_wi_galerkin[:, iδ][ig], ".--", label="Galerkin control variate")
    ax.set_xlabel("$\gamma$")
    ax.set_title("$\delta = {}$".format(δ))
    plt.legend()
    plt.savefig("var-delta={}.pdf".format(iδ))
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
