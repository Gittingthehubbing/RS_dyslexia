# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 00:24:06 2021

@author: Martin R. Vasilev
"""

import pytesseract

from PIL import Image, ImageOps
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
import pandas as pd
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replotted_"):
    
    fig, ax = plt.subplots(figsize=(15,15),dpi=300)

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

def text_coords(filename:str):
    """Function that extracts letter coordinates from stimulus image
    filename should be path to image with text."""
        

    ### Font settings:
    # TNR:
    # y_offset= 66
    # line_span= 18
    # dist_lines= 5 

    # smallest_letter_pixels_x = 3
    empty_space_width = 8
    y_offset= 4

    """From Paper: OD (Open Dyslexia) 18pt
    TNR (Times New Roman) 20pt
    1024 X 768 pixel resolution"""


    img_path = filename

    threshold = 225
    img = Image.open(img_path)
    img = img.convert('L')
    # Threshold
    img = img.point( lambda p: 255 if p > threshold else 0 )
    # To mono
    img = img.convert('1')

    # img = ImageOps.expand(img,border=10,fill='black') # add border that tessarect might benefit from

    im_width, im_height = img.size
    im_resize_factor = 4
    img = img.resize((im_width*im_resize_factor, im_height*im_resize_factor),resample=Image.Resampling.BICUBIC)

    data=pytesseract.image_to_boxes(img, config='--psm 11',output_type=pytesseract.Output.STRING)
    word_data_df=pytesseract.image_to_data(img, config='--psm 11',output_type=pytesseract.Output.DATAFRAME)
    osd=pytesseract.image_to_osd(img)
    xml=pytesseract.image_to_alto_xml(img)
    # pytesseract.run_and_get_output()

    text = pytesseract.image_to_string(img, config='--psm 11')

    line_nums = []
    current_line = -1
    for idx,row in word_data_df.iterrows():
        if idx==0:
            line_nums.append(0)
            continue
        else:
            if not row.isna().text and not isinstance(word_data_df.loc[idx-1,"text"],str): #np.isnan(word_data_df.loc[idx-1,"text"])
                current_line += 1
                line_nums.append(current_line)
            else:
                line_nums.append(current_line)
    word_data_df["line_numer"] = line_nums
    word_data_df = word_data_df.dropna(axis=0,how="any")    
    word_data_df.loc[:,["left","top","width","height"]] = word_data_df.loc[:,["left","top","width","height"]].copy().applymap(lambda x: x/im_resize_factor)

    line_height_from_ocr = word_data_df.height.mean()

    img = img.resize((im_width, im_height),resample=Image.Resampling.NEAREST)

    # with open('coords1.txt', 'w') as f:
    #     f.write(data)
        
    # with open('text1.txt', 'w') as f:
    #     f.write(text)



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
        letter_idx = letter.index(l)
        letter_width = x2[letter_idx] - x1[letter_idx]
        letter_widths[l] = letter_width
    max_letter_width = np.max([v for k,v in letter_widths.items()])

    
    boxes = []

    for df_idx,line_df in word_data_df.groupby("line_numer"):
        line_height = line_df.height.max()
        line_top = line_df.top.min()
        line_bottom = line_top+line_height
        for idx, row in line_df.iterrows():
            leter_start_x = row.left
            space_width = np.min([line_df.loc[:,"left"].iloc[x_idx] - (line_df.loc[:,"left"].iloc[x_idx-1] + line_df.loc[:,"width"].iloc[x_idx-1]) for x_idx in range(1,line_df.shape[0])])
            
            for lidx,l in enumerate(row.text):
                letter_end_x = leter_start_x + letter_widths[l]
                boxes.append(dict(
                    x1 = leter_start_x,
                    x2 = letter_end_x,
                    y1 = line_top,
                    y2 = line_bottom,
                    letter =l,
                    line = df_idx
                ))
                leter_start_x = leter_start_x + letter_widths[l]
        
            boxes.append(dict(
                x1 = leter_start_x,
                x2 = leter_start_x+space_width,
                y1 = line_top,
                y2 = line_bottom,
                letter =' ',
                line = df_idx
            ))
    boxes_df = pd.DataFrame(boxes)
    plot_text_and_boxes(img,boxes_df.x1,boxes_df.y1,boxes_df.x2,boxes_df.y2,filename,extra_text="from_im_to_data")

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
    plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replot_before_boxChange")
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
        
        y_start= min(y1_n[start:end])-y_offset # y offset of 1st line
        if len(y_ends)>0 and y_start < y_ends[-1]:
            y_start =  y_ends[-1]
        # y_end= max(y2_n[start:end])+y_offset
        y_end= y_start + line_height_from_ocr
        y_starts.append(y_start)
        y_ends.append(y_end)
        # replace existing y positions with the box bounds:
        y1_n[start:end]= [y_start]* len(y1_n.copy()[start:end])
        y2_n[start:end]= [y_end]* len(y2_n.copy()[start:end])
    

    plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replot_after_boxChange")

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
    # #x_diff= [x2_n - x1_n]
    # plot_text_and_boxes(img,x1_n,y1_n,x2_n,y2_n,filename,extra_text="replot_after_boxChange_touchingBoxes")
    df2 = pd.DataFrame(list(zip(letter_n, x1_n, x2_n, y1_n, y2_n)),
                columns =['letter', 'x1', 'x2', 'y1', 'y2'] )   
    return df2
