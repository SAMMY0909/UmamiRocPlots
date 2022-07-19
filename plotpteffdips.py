import numpy as np
import pandas as pd
import random
from puma.utils import logger
from puma.metrics import calc_rej
from puma import Roc, RocPlot
from puma import VarVsEff, VarVsEffPlot


logger.info("Starting pT vs EFF & Rejection plotting process ....")
logger.info("Reading h5 files")

df=[]
flist=[
       #"./umami/modelDIPSlrtold/results/results-200.h5",
       #"./umami/modelDIPSlrtold/results/results-200.h5",
       #"./umami/modelDIPSstdold/results/results-200.h5",
       #"./umami/modelDIPSlrt/results/results-200.h5",
       #"./umami/modelDIPSstd/results/results-200.h5",
       #"./Eval_results_lrt/results/results-0.h5",
       "./Eval_results_std/results/results-0.h5",
       ]
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(flist))]
for ind in flist:
    df.append(pd.read_hdf(ind))

disc_dips=[]
disc_name=['disc_dipsLoose20220314v2','disc_dips']

for ctr in range(len(df)):
    disc_dips.append(df[ctr][disc_name[0]])
    #if ctr==(len(df)-1) or ctr==(len(df)-2) or ctr==0:
    #if ctr==0:
    #    disc_dips.append(df[ctr][disc_name[0]])
    #else:
    #    disc_dips.append(df[ctr][disc_name[1]])

pt=[]

for ptind in range(len(df)):
    pt.append(df[ptind]["pt"]/1e3)

sig_eff = np.linspace(0.49, 1, 20)

is_l=[]
is_c=[]
is_b=[]

for qrkctr in range(len(df)):
    is_l.append(df[qrkctr]["labels"]==0)
    is_c.append(df[qrkctr]["labels"]==1)
    is_b.append(df[qrkctr]["labels"]==2)

####################################################################

plot_labels=[
    #"DIPS loose v2 Ref",
    #"DIPS,LRT+STD cont $d_0$<=3.5,$z_0$<=5",
    #"DIPS,STD cont $d_0$<=3.5,$z_0$<=5",
    #"DIPS,LRT+STD cont $d_0$<=15,$z_0$<=15",
    #"DIPS,STD cont $d_0$<=15,$z_0$<=15",
    #"DIPS v2 STD+LRT LLP $d_0$<=15,$z_0$<=15",
    "DIPS v2 LLP $d_0$<=15,$z_0$<=15",
    ]
plots_l=[]
plots_c=[]
###################################################################
logger.info("Initializing plots as a function of pt.")
for loopctr in range(len(disc_dips)):
    plots_l.append(
        VarVsEff(
            x_var_sig=pt[loopctr][is_b[loopctr]],
            disc_sig=disc_dips[loopctr][is_b[loopctr]],
            x_var_bkg=pt[loopctr][is_l[loopctr]],
            disc_bkg=disc_dips[loopctr][is_l[loopctr]],
            bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
            working_point=0.77,
            disc_cut=None,
            fixed_eff_bin=False,
            #colour=color[loopctr],
            label=plot_labels[loopctr],
        )
    )
for loopctr in range(len(disc_dips)):
    plots_c.append(
        VarVsEff(
            x_var_sig=pt[loopctr][is_b[loopctr]],
            disc_sig=disc_dips[loopctr][is_b[loopctr]],
            x_var_bkg=pt[loopctr][is_c[loopctr]],
            disc_bkg=disc_dips[loopctr][is_c[loopctr]],
            bins=[20, 30, 40, 60, 85, 110, 140, 200, 250],
            working_point=0.77,
            disc_cut=None,
            fixed_eff_bin=False,
            label=plot_labels[loopctr],
        )
    )
####################################################################
logger.info("Plotting light bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej_l = VarVsEffPlot(
    n_ratio_panels=1,
    mode="bkg_rej",
    ylabel="Light jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= '\nR22',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV \n$f_{c}=0.005$\n WP77',
    #y_scale=1.4,
    #figsize=[7, 5]
)
for ctr in range(len(plots_l)):
    plot_bkg_rej_l.add(plots_l[ctr],reference=not bool(ctr))

plot_bkg_rej_l.draw()
plot_bkg_rej_l.savefig("dips_loose_pt_l_rej_DIPS_llp.png")
######################################################################
logger.info("Plotting c bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej_c = VarVsEffPlot(
    n_ratio_panels=1,
    mode="bkg_rej",
    ylabel="c jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= '\nR22',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV \n$f_{c}=0.005$\n WP77',
    #y_scale=1.4,
    #figsize=[7, 5]

)

for ctr in range(len(plots_c)):
    plot_bkg_rej_c.add(plots_c[ctr],reference=not bool(ctr))

plot_bkg_rej_c.draw()
plot_bkg_rej_c.savefig("dips_loose_pt_c_rej_DIPS_llp.png")
##################################################################
plot_sig_eff = VarVsEffPlot(
    n_ratio_panels=1,
    mode="sig_eff",
    ylabel="b-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_first_tag= '\nR22',
    atlas_second_tag= '\n$\\sqrt{s}=13$ TeV \n$f_{c}=0.005$\n WP77',
    #y_scale=1.4,
    #figsize=[7, 5]
)

for ctr in range(len(plots_c)):
    plot_sig_eff.add(plots_l[ctr],reference=not bool(ctr))

# If you want to inverse the discriminant cut you can enable it via
# plot_sig_eff.set_inverse_cut()
plot_sig_eff.draw()
# Drawing a hline indicating inclusive efficiency
plot_sig_eff.draw_hline(0.77)
plot_sig_eff.savefig("pt_b_eff_DIPS_llp.png")
