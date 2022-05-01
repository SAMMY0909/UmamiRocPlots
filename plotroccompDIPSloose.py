import numpy as np
import pandas as pd
from umami.configuration import logger
from umami.metrics import calc_rej
from umami.plotting import roc, roc_plot

logger.info("Starting ROC plotting process SKYNET V3.0....")
logger.info("Reading h5 files")

df = pd.read_hdf("./umami/modelDIPSlrt/resultse200/results-rej_per_eff-200.h5")
df1 = pd.read_hdf("./umami/modelDIPSstd/resultse200/results-rej_per_eff-200.h5")
df2 = pd.read_hdf("./umami/modelDIPSlrtold/resultse200/results-rej_per_eff-200.h5")
df3 = pd.read_hdf("./umami/modelDIPSstdold/resultse200/results-rej_per_eff-200.h5")

dips_ujets_rej_lrt = df["dips_ujets_rej"]
dips_cjets_rej_lrt = df["dips_cjets_rej"]

dips_ujets_rej_std = df1["dips_ujets_rej"]
dips_cjets_rej_std = df1["dips_cjets_rej"]

sig_eff_lrt = df["effs"]
sig_eff_std = df1["effs"]

###########################################
dips_ujets_rej_lrt_old = df2["dips_ujets_rej"]
dips_cjets_rej_lrt_old = df2["dips_cjets_rej"]

dips_ujets_rej_std_old = df3["dips_ujets_rej"]
dips_cjets_rej_std_old = df3["dips_cjets_rej"]

sig_eff_lrt_old = df2["effs"]
sig_eff_std_old = df3["effs"]

##########################################

logger.info("Plotting ROC curves...")

plot_roc = roc_plot(
    n_ratio_panels=2, ylabel="background rejection", xlabel="b-jets efficiency", atlas_first_tag='R22 DIPS', atlas_second_tag='\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.005$\n WP77'
)
print("n_test=",df["dips_ujets_rej"].count())
plot_roc.add_roc(
    roc(
        sig_eff_lrt,
        dips_ujets_rej_lrt,
        n_test=df["dips_ujets_rej"].count(),
        rej_class="ujets",
        signal_class="bjets",
        label="e 200 DIPS STD+LRT d0-30,z0-30",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff_std,
        dips_ujets_rej_std,
        n_test=df1["dips_ujets_rej"].count(),
        rej_class="ujets",
        signal_class="bjets",
        label="e 200 DIPS STD d0-30,z0-30",
    ),
    reference=True,
)
plot_roc.add_roc(
    roc(
        sig_eff_lrt,
        dips_cjets_rej_lrt,
        n_test=df["dips_cjets_rej"].count(),
        rej_class="cjets",
        signal_class="bjets",
        label="e 200 DIPS STD+LRT d0-30,z0-30",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff_std,
        dips_cjets_rej_std,
        n_test=df1["dips_cjets_rej"].count(),
        rej_class="cjets",
        signal_class="bjets",
        label="e 200 DIPS STD d0-30,z0-30",
    ),
    reference=True,
)
####################################################################
plot_roc.add_roc(
    roc(
        sig_eff_lrt_old,
        dips_ujets_rej_lrt_old,
        n_test=df2["dips_ujets_rej"].count(),
        rej_class="ujets",
        signal_class="bjets",
        label="e 200 DIPS STD+LRT d0-3.5,z0-5",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff_std_old,
        dips_ujets_rej_std_old,
        n_test=df3["dips_ujets_rej"].count(),
        rej_class="ujets",
        signal_class="bjets",
        label="e 200 DIPS STD d0-3.5,z0-5",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff_lrt_old,
        dips_cjets_rej_lrt_old,
        n_test=df2["dips_cjets_rej"].count(),
        rej_class="cjets",
        signal_class="bjets",
        label="e 200 DIPS STD+LRT d0-3.5,z0-5",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff_std_old,
        dips_cjets_rej_std_old,
        n_test=df3["dips_cjets_rej"].count(),
        rej_class="cjets",
        signal_class="bjets",
        label="e 200 DIPS STD d0-3.5,z0-5",
    ),
)

####################################################################
# setting which flavour rejection ratio is drawn in which ratio panel
#plot_roc.set_ratio_class(1, "ujets", label="light jets ratio")
#plot_roc.set_ratio_class(2, "cjets", label="c jets ratio")
#plot_roc.set_ratio_class(1, "ujets", label="light jets ratio dips loose")
#plot_roc.set_ratio_class(2, "cjets", label="c jets ratio dips loose")

# if you want to swap the ratios just uncomment the following 2 lines
plot_roc.set_ratio_class(2, "ujets", label="light-flavour jets ratio")
plot_roc.set_ratio_class(1, "cjets", label="c-jets ratio")
plot_roc.set_leg_rej_labels("ujets", "light jets rejection")
plot_roc.set_leg_rej_labels("cjets", "c jets rejection")


plot_roc.draw()
plot_roc.savefig("roc_DIPScomparisone200.png")


