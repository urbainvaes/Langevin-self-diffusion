from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(15, 8))

# Get data {{{1
γs = [.00001, .0000215, .0000464, .0001, .000215, .000464, .001, .00215, .00464,
      .01, .0215, .0464, .1, .215, .464, 1.]
δs = [.0, .04, .08, .16, .32, .64]

diff_und = 0.30626213513957773

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
D_only_galerkin = np.loadtxt("precom_data/galerkin_approx_diff_imported_from_cluster.txt")[:, 1]

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

# Langevin dynamics in one dimension {{{1
γs = γs_galerkin
D_galerkin = D11_wi_galerkin[:, 0]
nD_galerkin = γs_galerkin*D11_wi_galerkin[:, 0]
σ_galerkin = σ11_wi_galerkin[:, 0]
D_underdamped = D11_wi_underdamped[:, 0]
σ_underdamped = σ11_wi_underdamped[:, 0]
D_nocontrol = D11_wo_underdamped[:, 0]
σ_nocontrol = σ11_wo_underdamped[:, 0]
approx_no_control = np.sqrt(2.)*abs(D_nocontrol)

plt.ion()
fig, ax = plt.subplots(1, 2)
ax[0].set_xlabel('$\gamma$')
ax[0].set_title(r"Effective diffusion coefficient: $\gamma D^{\gamma}/D_{\rm und}$")
ax[0].semilogx(γs_galerkin, γs*D_nocontrol/diff_und, ".-", label="MC/No control")
ax[0].semilogx(γs_galerkin, γs*D_galerkin/diff_und, ".-", label="MC/Galerkin")
ax[0].semilogx(γs_galerkin, γs*D_underdamped/diff_und, ".-", label="MC/Underdamped")
ax[0].semilogx(γs_galerkin, γs*D_only_galerkin/diff_und, ".-", label="Galerkin")
ax[0].legend()
ax[0].grid()
ax[0].set_xlim([1e-5, 1])
ax[1].set_xlabel('$\gamma$')
ax[1].set_title(r'Relative standard deviation')
ax[1].set_prop_cycle(None)
ax[1].loglog(γs_galerkin, σ_nocontrol/D_nocontrol, '.-', label="MC/No control")
ax[1].loglog(γs_galerkin, σ_galerkin/D_nocontrol, '.-', label="MC/Galerkin")
ax[1].loglog(γs_galerkin, σ_underdamped/D_nocontrol, '.-', label="MC/Underdamped")
ax[1].loglog(γs_galerkin, 0*γs_galerkin + np.sqrt(2), '-', lw=1, label=r"$\sqrt{2}$")
ax[1].set_ylim([.01, 2])
ax[1].set_xlim([1e-5, 1])
ax[1].set_yscale('log', base=2)
ax[1].legend()
ax[1].grid()
fig.savefig("underdamped_1d.pdf", bbox_inches='tight')
plt.show()

# Langevin dynamics in many dimensions: mean {{{1
fig, ax = plt.subplots()
ax.set_prop_cycle(None)
for iδ, δ in enumerate(δs_underdamped):
    iu = np.nonzero(D11_wo_underdamped[:, iδ])[0]
    γu = γs_underdamped[iu]
    yu = D11_wo_underdamped[:, iδ][iu]
    ifitu, cu = np.where(γu <= 1e-2)[0], [0]
    if len(ifitu) > 1:
        cu = np.polyfit(np.log10(γu[ifitu]), np.log10(yu[ifitu]), deg=1)
    line, = ax.loglog(γu, yu, ".", ms=12, label="$\delta = {}, D \propto \gamma^{{ {:.2f} }}$".format(δ, cu[0]))
    γs_interp = np.logspace(-5, 0, 10)
    ax.loglog(γs_interp, 10**cu[1]*γs_interp**cu[0], "--", c=line.get_color())
ax.set_prop_cycle(None)
# for iδ, δ in enumerate(δs_galerkin):
#     ig = np.nonzero(D11_wo_galerkin[:, iδ])[0]
#     γg = γs_galerkin[ig]
#     yg = D11_wo_galerkin[:, iδ][ig]
#     ax.loglog(γg, yg, ".-")
# ax.set_prop_cycle(None)
# for iδ, δ in enumerate(δs_galerkin):
#     ig = np.nonzero(D11_wi_galerkin[:, iδ])[0]
#     γg = γs_galerkin[ig]
#     yg = D11_wo_galerkin[:, iδ][ig]
#     ax.loglog(γg, yg, ".--")
ax.set_xlabel("$\gamma$")
ax.set_ylabel(r"$D^\gamma_{\mathbf e}$")
plt.legend()
plt.savefig("diffusion.pdf")
plt.show()

plt.ion()
fig, ax = plt.subplots()
δ = 0
ax.set_prop_cycle(None)
iu = np.nonzero(σ11_wo_underdamped[:, 0])[0]
γu = γs_underdamped[iu]
yu = σ11_wo_underdamped[:, 0][iu]
ax.loglog(γu, σ11_wo_underdamped[:, 0][iu]/D11_wo_underdamped[:, 0][iu], label="No variance reduction")
ax.loglog(γu, σ11_wi_underdamped[:, 0][iu]/D11_wo_underdamped[:, 0][iu], ".--", label="Underdamped control variate")
# ax.loglog(γs_new_galerkin, σ11_wi_new_galerkin, ".--", label="Spectral Galerkin with more modes")
ig = np.nonzero(σ11_wo_galerkin[:, 0])[0]
γg = γs_galerkin[ig]
yg = σ11_wo_galerkin[:, 0][ig]
# ax.loglog(γg, σ11_wo_galerkin[:, 0][ig])
ax.loglog(γg, σ11_wi_galerkin[:, 0][ig]/D11_wi_galerkin[:, 0][iu], ".--", label="Spectral Galerkin control variate")
ax.set_xlabel("$\gamma$")
ax.set_title("$\delta = {}$".format(δ))
plt.legend()
plt.savefig("var-delta={}.pdf".format(iδ))

fig, ax = plt.subplots()
ax.plot(γs_underdamped, np.sqrt(2) + 0*γs_underdamped, "k-", lw=3, label="No control")
# ax.plot(0, 0, "k-", label="MC/Galerkin")
# ax.plot(0, 0, "k--", label="MC/Underdamped")
for iδ, δ in enumerate(δs_galerkin):
    # ax.set_prop_cycle(None)
    iu = np.nonzero(σ11_wo_underdamped[:, iδ])[0]
    γu = γs_underdamped[iu]
    yu = σ11_wo_underdamped[:, iδ][iu]
    plot1 = ax.loglog(γu, σ11_wi_underdamped[:, iδ][iu]/D11_wo_underdamped[:, iδ][iu], "o--", label="MC/Underdamped, $\delta = {}$".format(δ))
    f = interp1d(np.log10(γu), np.log10(σ11_wi_underdamped[:, iδ][iu]/D11_wo_underdamped[:, iδ][iu]), bounds_error=False, fill_value=(5, 5))
    x = 10**fsolve(lambda x: f(x) - np.log10(np.sqrt(2)), -3)
    print(x)
    ig = np.nonzero(σ11_wo_galerkin[:, iδ])[0]
    γg = γs_galerkin[ig]
    yg = σ11_wo_galerkin[:, iδ][ig]
    # ax.loglog(γg, σ11_wo_galerkin[:, iδ][ig])
    # plot1 = ax.loglog(0, 0)
    # plot2 = ax.loglog(γg, σ11_wi_galerkin[:, iδ][ig]/D11_wo_galerkin[:, iδ][ig], "o--", color=plot1[0].get_color(), label="MC/Galerkin, $\delta = {}$".format(δ))
plt.legend()
ax.set_title("Relative standard deviation")
ax.set_xlim([1e-4, 1])
ax.set_xlabel(r"$\gamma$")
plt.savefig("var-delta.pdf", bbox_inches='tight')

# fig, ax = plt.subplots()
# ax.set_prop_cycle(None)
# for iδ, δ in enumerate(δs):
#     if δ < 0:
#         continue
#     coeffs = np.polyfit(np.log10(γs[:7]), np.log10(D11_wo[:7, iδ]), deg=1)
#     ax.loglog(γs, D11_wo[:, iδ], ".-",
#             label="$\delta = {}, D \propto \gamma^{{ {:.2f} }}$".format(δ, coeffs[0]))
# ax.set_prop_cycle(None)
# for iδ, δ in enumerate(δs):
#     if δ < 0:
#         continue
#     ax.loglog(γs, D11_wi[:, iδ], ".--")
# ax.set_xlabel("$γ$")
# plt.legend()
# plt.savefig("diffusion.pdf")
# plt.show()

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

# fig, ax = plt.subplots()
# for iδ, δ in enumerate(δs):
#     if δ < 0:
#         continue
#     fig, ax = plt.subplots()
#     ax.set_title("$\delta = {}$".format(δ))
#     ax.loglog(γs, σ11_wo[:, iδ], ".-")
#     ax.set_prop_cycle(None)
#     ax.loglog(γs, σ11_wi[:, iδ], ".--")
#     # ax.plot(γs, σ11_wo[:, iδ]/σ11_wi[:, iδ], ".-")
#     ax.set_xlabel("$γ$")
#     plt.show()
