import pathlib as pl
import os
cwd = os.getcwd()
if "preproc" not in cwd:
    os.chdir("preproc")
reference_plot_dir = pl.Path("../Reference_plots")

from get_coords import text_coords
scale_factor = 4
space_text = ["with_spaces","no_spaces"][0]

ref_plots = list(reference_plot_dir.glob(f"*scale_factor_{scale_factor}_{space_text}.png"))

for ref_plot in ref_plots:
    out = text_coords(
        ref_plot,
        use_image_to_data=False,
        use_reference_widths=False,
        plot_examples=True,
        binarise=True,
        add_border=False,
        upscale_im=False,
        return_letter_dict=True
    )

    if isinstance(out,tuple):
        boxes_df,letter_widths = out
    else:
        boxes_df = out
    boxes_df.to_excel(reference_plot_dir.joinpath(f"{ref_plot.stem}_from_tessarect.xlsx"),engine="openpyxl")