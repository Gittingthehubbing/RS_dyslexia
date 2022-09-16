# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:11:06 2021

@author: Martin Vasilev
"""

from functools import partial
from multiprocessing import Pool
import pathlib as pl

from rewrite_asc import rewrite_asc


def main():
    use_multiprocessing = True #very slow otherwise
    data_base_path = "D:/pydata/Eye_Tracking/Dyslexia/Dyslexia_Leon"
    file_dir= f'{data_base_path}/Dyslexia/'
    img_dir= f'{data_base_path}/Info/Stimuli Texts Questions/'
    files = [x.stem for x in pl.Path(file_dir).glob("*.asc")]

    if use_multiprocessing:
        with Pool() as pool:
            results = pool.map(partial(
                rewrite_asc, file_dir=file_dir, img_dir=img_dir
            ),files)
    else:
        for file_name in files:
            rewrite_asc(file_dir, file_name, img_dir)

if __name__=="__main__":
    main()

