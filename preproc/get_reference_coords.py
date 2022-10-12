import pathlib as pl
import os
cwd = os.getcwd()
if "preproc" not in cwd:
    os.chdir("preproc")
reference_plot_dir = pl.Path("../Reference_plots")

from get_coords import text_coords
scale_factor = 8

ref_plots = list(reference_plot_dir.glob(f"*scale_factor_{scale_factor}.png"))

ref_plot = ref_plots[1]

df = text_coords(ref_plot,True,True,True,False,False)