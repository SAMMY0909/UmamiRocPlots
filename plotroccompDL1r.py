import numpy as np
import pandas as pd
from umami.configuration import logger
from umami.metrics import calc_rej
from umami.plotting import roc, roc_plot

logger.info("Starting ROC plotting process SKYNET V3.0....")
logger.info("Reading h5 files")

df = pd.read_hdf("./umami/modelDL1rlrt/results/results-rej_per_eff-59.h5")
df1 = pd.read_hdf("./umami/modelDL1rstd/results/results-rej_per_eff-59.h5")

DL1r_ujets_rej_lrt = df["DL1r_ujets_rej"]
DL1r_cjets_rej_lrt = df["DL1r_cjets_rej"]

DL1r_ujets_rej_std = df1["DL1r_ujets_rej"]
DL1r_cjets_rej_std = df1["DL1r_cjets_rej"]

sig_eff_lrt = df["effs"]
sig_eff_std = df1["effs"]

logger.info("Plotting ROC curves...")

plot_roc = roc_plot(
    n_ratio_panels=2, ylabel="background rejection", xlabel="b-jets efficiency", atlas_first_tag='R22 DL1r', atlas_second_tag='\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ , $f_{c}=0.018$\n WP77'
)
plot_roc.add_roc(
    roc(
        sig_eff_lrt,
        DL1r_ujets_rej_lrt,
        n_test=df["DL1r_ujets_rej"].count(),
        rej_class="ujets",
        signal_class="bjets",
        label="Epoch 59/200 DL1r r22 STD+LRT",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff_std,
        DL1r_ujets_rej_std,
        n_test=df1["DL1r_ujets_rej"].count(),
        rej_class="ujets",
        signal_class="bjets",
        label="Epoch 59/200 DL1r r22 STD",
    ),
    reference=True,
)
plot_roc.add_roc(
    roc(
        sig_eff_lrt,
        DL1r_cjets_rej_lrt,
        n_test=df["DL1r_cjets_rej"].count(),
        rej_class="cjets",
        signal_class="bjets",
        label="Epoch 59/200 DL1r r22 STD+LRT",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff_std,
        DL1r_cjets_rej_std,
        n_test=df1["DL1r_cjets_rej"].count(),
        rej_class="cjets",
        signal_class="bjets",
        label="Epoch 59/200 DL1r r22 STD",
    ),
    reference=True,
)
# setting which flavour rejection ratio is drawn in which ratio panel
plot_roc.set_ratio_class(1, "ujets", label="light-flavour jets ratio")
plot_roc.set_ratio_class(2, "cjets", label="c-jets ratio")
# if you want to swap the ratios just uncomment the following 2 lines
# plot_roc.set_ratio_class(2, "ujets", label="light-flavour jets ratio")
# plot_roc.set_ratio_class(1, "cjets", label="c-jets ratio")
plot_roc.set_leg_rej_labels("ujets", "light-flavour jets rejection")
plot_roc.set_leg_rej_labels("cjets", "c-jets rejection")


plot_roc.draw()
plot_roc.savefig("roc_DL1r.png")


