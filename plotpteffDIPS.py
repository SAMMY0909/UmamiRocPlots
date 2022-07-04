import numpy as np
import pandas as pd
from puma.utils import logger
from puma.metrics import calc_rej
from puma import Roc, RocPlot
from puma import VarVsEff, VarVsEffPlot

logger.info("Starting PT EFF  Rjecetion plotting process SKYNET V3.0....")
logger.info("Reading h5 files")

df = pd.read_hdf("./umami/modelDIPSlrtold/results/results-200.h5")
df1 = pd.read_hdf("./umami/modelDIPSstdold/results/results-200.h5")
df2 = pd.read_hdf("./umami/modelDIPSlrt/results/results-200.h5")
df3 = pd.read_hdf("./umami/modelDIPSstd/results/results-200.h5")

discs_dips = df["disc_dips"]
discs_dips1 = df1["disc_dips"]
discs_dips2 = df2["disc_dips"]
discs_dips3 = df3["disc_dips"]
discs_dips4 = df1["disc_dipsLoose20220314v2"]

pt = df["pt"].values / 1e3
pt1 = df1["pt"].values / 1e3
pt2 = df2["pt"].values / 1e3
pt3 = df3["pt"].values / 1e3
# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)


is_l_lrt = df["labels"] == 0
is_c_lrt = df["labels"] == 1
is_b_lrt = df["labels"] == 2

is_l_std = df1["labels"] == 0
is_c_std = df1["labels"] == 1
is_b_std = df1["labels"] == 2

is_l_lrt1 = df2["labels"] == 0
is_c_lrt1 = df2["labels"] == 1
is_b_lrt1 = df2["labels"] == 2

is_l_std1 = df3["labels"] == 0
is_c_std1 = df3["labels"] == 1
is_b_std1 = df3["labels"] == 2
####################################################################
dips_l_lrt = VarVsEff(
    x_var_sig=pt[is_b_lrt],
    disc_sig=discs_dips[is_b_lrt],
    x_var_bkg=pt[is_l_lrt],
    disc_bkg=discs_dips[is_l_lrt],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,LRT+STD cont d0<=3.5,z0<=5",
)

dips_l_std = VarVsEff(
    x_var_sig=pt1[is_b_std],
    disc_sig=discs_dips1[is_b_std],
    x_var_bkg=pt1[is_l_std],
    disc_bkg=discs_dips1[is_l_std],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,STD cont d0<=3.5,z0<=5",
)

dips_l_ref = VarVsEff(
    x_var_sig=pt1[is_b_std],
    disc_sig=discs_dips4[is_b_std],
    x_var_bkg=pt1[is_l_std],
    disc_bkg=discs_dips4[is_l_std],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS loose v2 Ref",
)
####################################################################
dips_l_lrt1 = VarVsEff(
    x_var_sig=pt2[is_b_lrt1],
    disc_sig=discs_dips2[is_b_lrt1],
    x_var_bkg=pt2[is_l_lrt1],
    disc_bkg=discs_dips2[is_l_lrt1],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,LRT+STD cont  d0<=15,z0<=15",
)

dips_l_std1 = VarVsEff(
    x_var_sig=pt3[is_b_std1],
    disc_sig=discs_dips3[is_b_std1],
    x_var_bkg=pt3[is_l_std1],
    disc_bkg=discs_dips3[is_l_std1],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,STD cont d0<=15,z0<=15",
)

#######################################################################
logger.info("Plotting light bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="Light jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= 'R22 DIPS',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.005$\n WP77',
)
plot_bkg_rej.add(dips_l_std)
plot_bkg_rej.add(dips_l_lrt)
plot_bkg_rej.add(dips_l_std1)
plot_bkg_rej.add(dips_l_lrt1)
plot_bkg_rej.add(dips_l_ref, reference=True)

plot_bkg_rej.draw()
plot_bkg_rej.savefig("dips_loose_pt_l_rej_DIPS.png")
######################################################################
dips_c_lrt = VarVsEff(
    x_var_sig=pt[is_b_lrt],
    disc_sig=discs_dips[is_b_lrt],
    x_var_bkg=pt[is_c_lrt],
    disc_bkg=discs_dips[is_c_lrt],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,LRT+STD cont d0<=3.5,z0<=5",
)

dips_c_std = VarVsEff(
    x_var_sig=pt1[is_b_std],
    disc_sig=discs_dips1[is_b_std],
    x_var_bkg=pt1[is_c_std],
    disc_bkg=discs_dips1[is_c_std],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,STD cont d0<=3.5,z0<=5",
)

dips_c_ref = VarVsEff(
    x_var_sig=pt1[is_b_std],
    disc_sig=discs_dips4[is_b_std],
    x_var_bkg=pt1[is_c_std],
    disc_bkg=discs_dips4[is_c_std],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS loose v2 Ref",
)
#####################################################################################
dips_c_lrt1 = VarVsEff(
    x_var_sig=pt2[is_b_lrt1],
    disc_sig=discs_dips2[is_b_lrt1],
    x_var_bkg=pt2[is_c_lrt1],
    disc_bkg=discs_dips2[is_c_lrt1],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,LRT+STD cont d0<=15,z0<=15",
)
dips_c_std1 = VarVsEff(
    x_var_sig=pt3[is_b_std1],
    disc_sig=discs_dips3[is_b_std1],
    x_var_bkg=pt3[is_c_std1],
    disc_bkg=discs_dips3[is_c_std1],
    bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
    working_point=0.77,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS,STD cont d0<=15,z0<=15",
)
#####################################################################################
logger.info("Plotting c bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej1 = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="c jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= 'R22 DIPS',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.005$\n WP77',
)
plot_bkg_rej1.add(dips_c_std)
plot_bkg_rej1.add(dips_c_lrt)
plot_bkg_rej1.add(dips_c_std1)
plot_bkg_rej1.add(dips_c_lrt1)
plot_bkg_rej1.add(dips_c_ref, reference=True)

plot_bkg_rej1.draw()
plot_bkg_rej1.savefig("dips_loose_pt_c_rej_DIPS.png")
##################################################################
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="b-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= 'R22 DIPS',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.005$',
)
plot_sig_eff.add(dips_l_std)
plot_sig_eff.add(dips_l_lrt)
plot_sig_eff.add(dips_l_std1)
plot_sig_eff.add(dips_l_lrt1)
plot_sig_eff.add(dips_l_ref)

plot_sig_eff.atlas_second_tag += "\n $\\epsilon_b=77%%$"

# If you want to inverse the discriminant cut you can enable it via
# plot_sig_eff.set_inverse_cut()
plot_sig_eff.draw()
# Drawing a hline indicating inclusive efficiency
plot_sig_eff.draw_hline(0.77)
plot_sig_eff.savefig("pt_b_eff_DIPS.png")

