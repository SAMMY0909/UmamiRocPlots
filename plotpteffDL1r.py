import numpy as np
import pandas as pd
from umami.configuration import logger
from umami.metrics import calc_rej
from umami.plotting import roc, roc_plot
from umami.plotting import var_vs_eff, var_vs_eff_plot

logger.info("Starting PT EFF  Rjecetion plotting process SKYNET V3.0....")
logger.info("Reading h5 files")

df = pd.read_hdf("./umami/modelDL1rlrt/results/results-59.h5")
df1 = pd.read_hdf("./umami/modelDL1rstd/results/results-59.h5")

discs_DL1r = df["disc_DL1r"]
discs_DL1r1 = df1["disc_DL1r"]

pt = df["pt"].values / 1e3
pt1 = df1["pt"].values / 1e3

# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)


is_l_lrt = df["labels"] == 0
is_c_lrt = df["labels"] == 1
is_b_lrt = df["labels"] == 2

is_l_std = df1["labels"] == 0
is_c_std = df1["labels"] == 1
is_b_std = df1["labels"] == 2

DL1r_l_lrt = var_vs_eff(
    x_var_sig=pt[is_b_lrt],
    disc_sig=discs_DL1r[is_b_lrt],
    x_var_bkg=pt[is_l_lrt],
    disc_bkg=discs_DL1r[is_l_lrt],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    wp=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DL1r R22 Epoch:59 LRT+STD",
)

DL1r_l_std = var_vs_eff(
    x_var_sig=pt1[is_b_std],
    disc_sig=discs_DL1r1[is_b_std],
    x_var_bkg=pt1[is_l_std],
    disc_bkg=discs_DL1r1[is_l_std],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    wp=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DL1r R22 Epoch:59 STD",
)

logger.info("Plotting light bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej = var_vs_eff_plot(
    mode="bkg_rej",
    ylabel="Light jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= 'R22 DL1r',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.018$\n WP77',
)
plot_bkg_rej.add(DL1r_l_std, reference=True)
plot_bkg_rej.add(DL1r_l_lrt)

plot_bkg_rej.draw()
plot_bkg_rej.savefig("pt_l_rej_DL1r.png")
######################################################################
DL1r_c_lrt = var_vs_eff(
    x_var_sig=pt[is_b_lrt],
    disc_sig=discs_DL1r[is_b_lrt],
    x_var_bkg=pt[is_c_lrt],
    disc_bkg=discs_DL1r[is_c_lrt],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    wp=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DL1r R22 Epoch:59 LRT+STD",
)

DL1r_c_std = var_vs_eff(
    x_var_sig=pt1[is_b_std],
    disc_sig=discs_DL1r1[is_b_std],
    x_var_bkg=pt1[is_c_std],
    disc_bkg=discs_DL1r1[is_c_std],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    wp=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DL1r R22 Epoch:59 STD",
)
logger.info("Plotting c bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej1 = var_vs_eff_plot(
    mode="bkg_rej",
    ylabel="c jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= 'R22 DL1r',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.018$\n WP77',
)
plot_bkg_rej1.add(DL1r_c_std, reference=True)
plot_bkg_rej1.add(DL1r_c_lrt)

plot_bkg_rej1.draw()
plot_bkg_rej1.savefig("pt_c_rej_DL1r.png")
##################################################################
plot_sig_eff = var_vs_eff_plot(
    mode="sig_eff",
    ylabel="b-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= 'R22 DL1r',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.018$',
)
plot_sig_eff.add(DL1r_l_std, reference=True)
plot_sig_eff.add(DL1r_l_lrt)

plot_sig_eff.atlas_second_tag += "\nInclusive $\\epsilon_b=77%%$"

# If you want to inverse the discriminant cut you can enable it via
# plot_sig_eff.set_inverse_cut()
plot_sig_eff.draw()
# Drawing a hline indicating inclusive efficiency
plot_sig_eff.draw_hline(0.77)
plot_sig_eff.savefig("pt_b_eff_DL1r.png")
