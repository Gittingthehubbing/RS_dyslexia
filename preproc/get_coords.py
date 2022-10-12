# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 00:24:06 2021

@author: Martin R. Vasilev
"""

import pytesseract

from PIL import Image, ImageOps
import os
if os.path.exists("C:/Program Files/Tesseract-OCR/tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

import pandas as pd
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from https://www.nature.com/articles/s41598-021-84945-9
SCREEN_RES = (1024,768)
SCREEN_SIZE = 21 # inches diag
ratio = SCREEN_RES[0]/SCREEN_RES[1]
diag_in_px = np.sqrt(SCREEN_RES[0]**2 + SCREEN_RES[1]**2)
height_in_in = SCREEN_SIZE/np.sqrt(1+ratio**2)
width_in_in = ratio * height_in_in
dpi = SCREEN_RES[0]/width_in_in
dpi_from_diag = diag_in_px/SCREEN_SIZE

FONT_PROPS = dict(
    TNR = dict(font_size = 20),
    OD = dict(font_size = 18),
)

def plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replotted_",dpi=300):
    fig, ax = plt.subplots(figsize=(SCREEN_RES[0]/dpi,SCREEN_RES[1]/dpi),dpi=dpi)

    # Display the image
    ax.imshow(img)

    # # Create a Rectangle patch
    ymin = np.min(y1_n)
    xmin = np.min(x1_n)
    # rect = patches.Rectangle((xmin, ymin), 5, line_span, linewidth=1, edgecolor='r', facecolor='none')

    for idx in range(len(x1_n)):
        xdiff = x2_n[idx]-x1_n[idx]
        ydiff = y2_n[idx]-y1_n[idx]
        rect = patches.Rectangle((x1_n[idx], y1_n[idx]), xdiff, ydiff, linewidth=0.25, edgecolor='r', facecolor='none')

        # # Add the patch to the Axes
        ax.add_patch(rect)
    filepath =  pl.Path(filename)
    plot_dir = filepath.parent.joinpath("plots")
    plot_dir.mkdir(exist_ok=True)
    save_name = plot_dir.joinpath(f"{extra_text}{filepath.stem}.png")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close("all")

def text_coords(filename:str,use_image_to_data=False,plot_examples=False,binarise=False,add_border=True,upscale_im=True):
    """Function that extracts letter coordinates from stimulus image
    filename should be path to image with text."""
        

    ### Font settings:
    # TNR:
    # y_offset= 66
    # line_span= 18
    # dist_lines= 5 

    # smallest_letter_pixels_x = 3
    # empty_space_width = 8
    y_offset= 4

    """From Paper: OD (Open Dyslexia) 18pt
    TNR (Times New Roman) 20pt
    1024 X 768 pixel resolution"""

    #TODO try:
    #cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            # cv.THRESH_BINARY,11,2)

    img_path = filename

    img = Image.open(img_path)

    if binarise:
        threshold = 225
        img = img.convert('L')
        # Threshold
        img = img.point( lambda p: 255 if p > threshold else 0 )
        # To mono
        img = img.convert('1')

    if add_border:
        img = ImageOps.expand(img,border=10,fill='black') # add border that tessarect might benefit from

    im_width, im_height = img.size
    if upscale_im:
        im_resize_factor = 4
        if hasattr(Image,"Resampling"):
            resample = Image.Resampling.BICUBIC
        elif hasattr(Image,"BICUBIC"):
            resample = Image.BICUBIC
        img = img.resize((im_width*im_resize_factor, im_height*im_resize_factor),resample=resample)
    else:
        im_resize_factor = 1

    #https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc

    # if use_image_to_data:
    word_data_df=pytesseract.image_to_data(img, config='--psm 3 --oem 1',output_type=pytesseract.Output.DATAFRAME)
    text = pytesseract.image_to_string(img, config='--psm 3')
    bool_filter = [False if ' ' == row.text else True for _,row in word_data_df.iterrows()]
    word_data_df = word_data_df.copy().iloc[bool_filter]
    if ' ' in word_data_df.iloc[-1].text:
        word_data_df = word_data_df.copy().iloc[:-1]
    line_nums = []
    # is_last_word_in_line = []
    current_line = -1
    for idx,row in word_data_df.iterrows():
        # is_last_word_in_line.append(False)
        if idx==0:
            line_nums.append(-1)
            # if idx == word_data_df.shape[0]:
                # is_last_word_in_line[0]=True
            continue
        else:
            if not row.isna().text and not isinstance(word_data_df.loc[idx-1,"text"],str): #np.isnan(word_data_df.loc[idx-1,"text"])
                current_line += 1
                line_nums.append(current_line)
                # if idx == 1:
                    # is_last_word_in_line[0]=False
                # is_last_word_in_line[idx-1]=True
            else:
                line_nums.append(current_line)

    word_data_df["line_numer"] = line_nums
    # word_data_df["is_last_word_in_line"] = is_last_word_in_line
    word_data_df = word_data_df.dropna(axis=0,how="any")    
    word_data_df.reset_index(inplace=True,drop=True)
    word_data_df.loc[:,["left","top","width","height"]] = word_data_df.loc[:,["left","top","width","height"]].copy().applymap(lambda x: x/im_resize_factor)

    line_height_from_ocr = word_data_df.height.mean()

        
    # else:
    data=pytesseract.image_to_boxes(img, config='--psm 11',output_type=pytesseract.Output.STRING)
    text_psm_11 = pytesseract.image_to_string(img, config='--psm 11')
    

    if hasattr(Image,"Resampling"):
        resample = Image.Resampling.NEAREST
    elif hasattr(Image,"NEAREST"):
        resample = Image.NEAREST
    img = img.resize((im_width, im_height),resample=resample)


    lines= data.split('\n')
    lines= list(filter(None, lines))

    yRes= im_height # y dimension of screen
    # here y coords are recorded relative to bottom of image, so we need to reverse them

    # recode values from tesseract string into num coords:
    letter= []
    x1= []
    y1= []
    x2= []
    y2= []

    for i in lines:
        break_line= i.split(' ')
        letter.append(break_line[0]) # add letters
        x1.append(int(break_line[1])/im_resize_factor) # x1 left end of box
        x2.append(int(break_line[3])/im_resize_factor) # x2 right end of box
        
        y1.append(yRes- int(break_line[4])/im_resize_factor)
        y2.append(yRes- int(break_line[2])/im_resize_factor)
        


    # plt.show()
    unique_letters = np.unique(letter)
    letter_widths = dict()
    for l in unique_letters:
        indices = [i for i in range(len(letter)) if letter[i] == l]
        letter_widths_found = [x2[letter_idx] - x1[letter_idx] for letter_idx in indices]
        # letter_idx = letter.index(l)
        # letter_width = x2[letter_idx] - x1[letter_idx]
        letter_width = np.median(letter_widths_found)
        letter_widths[l] = letter_width
    max_letter_width = np.max([v for k,v in letter_widths.items()])
    min_letter_width = np.min([v for k,v in letter_widths.items()])
    letter_widths['.'] = min_letter_width
    letter_widths[','] = min_letter_width

    if use_image_to_data:
        
        boxes = []

        for df_idx,line_df in word_data_df.groupby("line_numer"):
            line_height = line_df.height.max()
            line_top = line_df.top.min()
            line_bottom = line_top+line_height
            for idx, word_df in line_df.reset_index().iterrows():
                letter_start_x = word_df.left
                
                for lidx,l in enumerate(word_df.text):
                    letter_end_x = letter_start_x + letter_widths[l]
                    boxes.append(dict(
                        x1 = letter_start_x,
                        x2 = letter_end_x,
                        y1 = line_top,
                        y2 = line_bottom,
                        letter =l,
                        line = df_idx
                    ))
                    letter_start_x = letter_start_x + letter_widths[l]
                # width_assigned_to_word = np.sum([x["x2"]-x["x1"] for x in boxes[-(lidx+1):]])
                # width_adjust_factor = word_df.width / width_assigned_to_word
                if idx == line_df.shape[0]-1:
                    continue
                if line_df.shape[0] < 2:
                    space_width = word_df.width
                else:
                    space_width = np.min([line_df.loc[:,"left"].iloc[x_idx] - (line_df.loc[:,"left"].iloc[x_idx-1] + line_df.loc[:,"width"].iloc[x_idx-1]) for x_idx in range(1,line_df.shape[0])])
            
                boxes.append(dict(
                    x1 = letter_start_x,
                    x2 = letter_start_x+space_width,
                    y1 = line_top,
                    y2 = line_bottom,
                    letter =' ',
                    line = df_idx
                ))
        boxes_df = pd.DataFrame(boxes)
        for idx in range(boxes_df.shape[0]-1):
            if boxes_df.loc[idx].letter == ' ':
                boxes_df.loc[idx,"x2"] = boxes_df.copy().iloc[idx+1].x1
        if plot_examples: plot_text_and_boxes(img,boxes_df.x1,boxes_df.y1,boxes_df.x2,boxes_df.y2,filename,extra_text="from_im_to_data")
        if use_image_to_data:
            return boxes_df

    #plt.imsave(fname='my_image.png', arr=img, cmap='gray_r', format='png')
    # now we need to go over the coords and add the empty spaces:
    letter_n= [letter[0]]
    x1_n= [x1[0]]
    x2_n= [x2[0]]
    y1_n= [y1[0]]
    y2_n= [y2[0]]
    maximum_letter_width =np.max(np.array(x2)- np.array(x1))
    for i in range(1,len(letter)):
        pixel_difference = x1[i]- x2[i-1]
        if  pixel_difference >= maximum_letter_width/6: #i>0 and
            letter_n.append(' ')
            x1_n.append(x2[i-1]+1) # start of empty space
            x2_n.append(x1[i]-1) # end of empty space
            y1_n.append(y1[i])
            y2_n.append(y2[i])
            
            ## now we need to append actual character at current iteration:
            letter_n.append(letter[i])
            x1_n.append(x1[i])
            x2_n.append(x2[i])
            y1_n.append(y1[i])
            y2_n.append(y2[i])
                            
        else:
            letter_n.append(letter[i])
            x1_n.append(x1[i])
            x2_n.append(x2[i])
            y1_n.append(y1[i])
            y2_n.append(y2[i])

        current_text = "".join(letter_n)
            
    # df = pd.DataFrame(list(zip(letter_n, x1_n, x2_n, y1_n, y2_n)),
    #                columns =['letter', 'x1', 'x2', 'y1', 'y2'] )


    xdiff = np.diff(x1_n) # differences between successive x1 numbers
    # Return-sweeps are going to show (large) negative differences
    neg_index= np.where((xdiff < 0)&(abs(xdiff)>xdiff[0]*4))# find position of line breaks
    breaks= np.append(neg_index[0], len(x1_n))

    ### start at beginning and use a fixed offset and between line-height
    #how to fix extreme values affecting min/ max:
    #1) keep track of prev lines- make sure the current line is bigger than the end of the previous one. 
    #2) If not, take the average of line spacing in the previous
    if plot_examples: plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replot_before_boxChange")
    y_starts = []
    y_ends = []
    for i in range(len(breaks)):
        if i==0:
            start= 0
            end= breaks[0]+1 # +1 bc we count from 0
            # y_start= y_offset # y offset of 1st line
            # y_end= y_start+ line_span 
        else:
            start= breaks[i-1]+1
            end= breaks[i]+1
            # y_start= y_end +dist_lines # y offset of 1st line
            # y_end= y_start+ line_span 
        
        y_start= min(y1_n[start:end])-line_height_from_ocr/4 # y offset of 1st line
        if len(y_ends)>0 and y_start < y_ends[-1]:
            y_start =  y_ends[-1]
        # y_end= max(y2_n[start:end])+y_offset
        y_end= y_start + line_height_from_ocr + line_height_from_ocr/4
        y_starts.append(y_start)
        y_ends.append(y_end)
        # replace existing y positions with the box bounds:
        y1_n[start:end]= [y_start]* len(y1_n.copy()[start:end])
        y2_n[start:end]= [y_end]* len(y2_n.copy()[start:end])
    

    if plot_examples: plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replot_after_boxChange")

    for x_idx in range(1,len(x1_n)):
        if x1_n[x_idx] > x2_n[x_idx-1]:
            x1_n[x_idx] = x2_n[x_idx-1]
    # y1_n_diff_all = np.diff(y1_n)
    # y1_n_diff = np.unique(y1_n_diff_all)
    # lines_span_from_ocr = y1_n_diff[y1_n_diff>0]
    # # assert len(line_span_from_ocr) == 1, "Line spans not equal"
    # line_span_from_ocr = lines_span_from_ocr[0]
    # for i in range(len(breaks)):
    #     if i==0:
    #         start= 0
    #         end= breaks[0]+1 # +1 bc we count from 0
    #         y_start= min(y1_n) # y offset of 1st line
    #     else:
    #         start= breaks[i-1]+1
    #         end= breaks[i]+1
    #         y_start= y_end # y offset of 1st line


    #     y_end= y_start+ line_span_from_ocr 
        
    #     y1_n[start:end]= [y_start]* len(y1_n.copy()[start:end])
    #     y2_n[start:end]= [y_end]* len(y2_n.copy()[start:end])
    #x_diff= [x2_n - x1_n]
    if plot_examples: plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replot_after_boxChange_touchingBoxes")
    df2 = pd.DataFrame(list(zip(letter_n, x1_n, x2_n, y1_n, y2_n)),
                columns =['letter', 'x1', 'x2', 'y1', 'y2'] )   

    
    return df2

if __name__ == '__main__':
    print("not for executing")