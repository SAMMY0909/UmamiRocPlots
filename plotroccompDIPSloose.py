import numpy as np
import pandas as pd
from puma.utils import logger
from puma.metrics import calc_rej
from puma import Roc, RocPlot

logger.info("Starting ROC plotting process SKYNET V3.0....")
logger.info("Reading h5 files")

df = pd.read_hdf("./umami/modelDIPSlrtold/results/results-rej_per_eff-200.h5")
df1 = pd.read_hdf("./umami/modelDIPSstdold/results/results-rej_per_eff-200.h5")
df2 = pd.read_hdf("./umami/modelDIPSlrt/results/results-rej_per_eff-200.h5")
df3 = pd.read_hdf("./umami/modelDIPSstd/results/results-rej_per_eff-200.h5")


dips_ujets_rej_lrt = df["dips_ujets_rej"]
dips_cjets_rej_lrt = df["dips_cjets_rej"]
dips_ujets_rej_lrt1 = df2["dips_ujets_rej"]
dips_cjets_rej_lrt1 = df2["dips_cjets_rej"]

dips_ujets_rej_std = df1["dips_ujets_rej"]
dips_cjets_rej_std = df1["dips_cjets_rej"]
dips_ujets_rej_std1 = df3["dips_ujets_rej"]
dips_cjets_rej_std1 = df3["dips_cjets_rej"]

sig_eff_lrt = df["effs"]
sig_eff_std = df1["effs"]
sig_eff_lrt1 = df2["effs"]
sig_eff_std1 = df3["effs"]
###########################################
dips_ujets_rej_ref = df["dipsLoose20220314v2_ujets_rej"]
dips_cjets_rej_ref = df["dipsLoose20220314v2_cjets_rej"]

##########################################

logger.info("Plotting ROC curves...")

plot_roc = RocPlot(
    n_ratio_panels=2, 
    ylabel="background rejection", 
    xlabel="b-jets efficiency", 
    atlas_first_tag='R22 DIPS', 
    atlas_second_tag='\n$\\sqrt{s}=13$ TeV, PFlow,\n$t\\bar{t}$ , $f_{c}=0.005$\n WP77'
)
#print("n_test=", df["dips_ujets_rej"].count())
plot_roc.add_roc(
    Roc(
        sig_eff_lrt,
        dips_ujets_rej_lrt,
        n_test=1_000_000,
        rej_class="ujets",
        signal_class="bjets",
        label="DIPS,STD+LRT cont d0<=3.5,z0<=5",
    ),
)
plot_roc.add_roc(
    Roc(
        sig_eff_std,
        dips_ujets_rej_std,
        n_test=1_000_000,
        rej_class="ujets",
        signal_class="bjets",
        label="DIPS,STD cont d0<=3.5,z0<=5",
    ),

)
plot_roc.add_roc(
    Roc(
        sig_eff_lrt,
        dips_cjets_rej_lrt,
        n_test=1_000_000,
        rej_class="cjets",
        signal_class="bjets",
        label="DIPS,STD+LRT cont d0<=3.5,z0<=5",
    ),
)
plot_roc.add_roc(
    Roc(
        sig_eff_std,
        dips_cjets_rej_std,
        n_test=1_000_000,
        rej_class="cjets",
        signal_class="bjets",
        label="DIPS,STD cont d0<=3.5,z0<=5",
    ),
)
####################################################################
plot_roc.add_roc(
    Roc(
        sig_eff_lrt1,
        dips_ujets_rej_lrt1,
        n_test=1_000_000,
        rej_class="ujets",
        signal_class="bjets",
        label="DIPS,STD+LRT cont d0<=15,z0<=15",
    ),
)
plot_roc.add_roc(
    Roc(
        sig_eff_std1,
        dips_ujets_rej_std1,
        n_test=1_000_000,
        rej_class="ujets",
        signal_class="bjets",
        label="DIPS,STD cont d0<=15,z0<=15",
    ),
)
plot_roc.add_roc(
    Roc(
        sig_eff_lrt1,
        dips_cjets_rej_lrt1,
        n_test=1_000_000,
        rej_class="cjets",
        signal_class="bjets",
        label="DIPS,STD+LRT cont d0<=15,z0<=15",
    ),
)
plot_roc.add_roc(
    Roc(
        sig_eff_std1,
        dips_cjets_rej_std1,
        n_test=1_000_000,
        rej_class="cjets",
        signal_class="bjets",
        label="DIPS,STD cont d0<=15,z0<=15",
    ),
)
####################################################################
plot_roc.add_roc(
    Roc(
        sig_eff_lrt,
        dips_ujets_rej_ref,
        n_test=1_000_000,
        rej_class="ujets",
        signal_class="bjets",
        label="DIPS Loose v2 Ref",
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff_std,
        dips_cjets_rej_ref,
        n_test=1_000_000,
        rej_class="cjets",
        signal_class="bjets",
        label="DIPS Loose v2 Ref",
    ),
    reference=True,
)
####################################################################
plot_roc.set_ratio_class(2, "ujets", label="light-flavour jets ratio")
plot_roc.set_ratio_class(1, "cjets", label="c-jets ratio")
plot_roc.set_leg_rej_labels("ujets", "light jets rejection")
plot_roc.set_leg_rej_labels("cjets", "c jets rejection")
plot_roc.draw()
plot_roc.savefig("roc_DIPSloosecomparison.png")
