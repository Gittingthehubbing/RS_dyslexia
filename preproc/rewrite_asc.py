

import os
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import pathlib as pl



def text_coords(filename:str, yRes:int= 768):
    """Function that extracts letter coordinates from stimulus image
    filename should be path to image with text."""
        

    ### Font settings:
    # TNR:
    y_offset= 66
    line_span= 18
    dist_lines= 5 


    img_path = filename
    img = Image.open(img_path)
    data=pytesseract.image_to_boxes(img)

    text = pytesseract.image_to_string(img, config='--psm 11')

    # with open('coords1.txt', 'w') as f:
    #     f.write(data)
        
    # with open('text1.txt', 'w') as f:
    #     f.write(text)

    # fig, ax = plt.subplots()

    # # Display the image
    # ax.imshow(img)

    # # Create a Rectangle patch
    # rect = patches.Rectangle((113, 768-688), 5, 13, linewidth=1, edgecolor='r', facecolor='none')

    # # Add the patch to the Axes
    # ax.add_patch(rect)

    # plt.show()

    #plt.imsave(fname='my_image.png', arr=img, cmap='gray_r', format='png')


    lines= data.split('\n')
    lines= list(filter(None, lines))

    yRes= img.size[1] # y dimension of screen
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
        x1.append(int(break_line[1])) # x1 left end of box
        x2.append(int(break_line[3])) # x2 right end of box
        
        y1.append(yRes- int(break_line[4]))
        y2.append(yRes- int(break_line[2]))
        

    # now we need to go over the coords and add the empty spaces:
    letter_n= []
    x1_n= []
    x2_n= []
    y1_n= []
    y2_n= []
        
    for i in range(len(letter)):
        
        if i>0 and x1[i]- x2[i-1] >= 3:
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
    neg_index= np.where(xdiff < 0)# find position of line breaks
    breaks= np.append(neg_index[0], len(x1_n))

    ### start at beginning and use a fixed offset and between line-height
    #how to fix extreme values affecting min/ max:
    #1) keep track of prev lines- make sure the current line is bigger than the end of the previous one. 
    #2) If not, take the average of line spacing in the previous

    for i in range(len(breaks)):
        if i==0:
            start= 0
            end= breaks[0]+1 # +1 bc we count from 0
            y_start= y_offset # y offset of 1st line
            y_end= y_start+ line_span 
        else:
            start= breaks[i-1]+1
            end= breaks[i]+1
            y_start= y_end +dist_lines # y offset of 1st line
            y_end= y_start+ line_span 
            
        # replace existing y positions with the box bounds:
        y1_n[start:end]= [y_start]* len(y1_n.copy()[start:end])
        y2_n[start:end]= [y_end]* len(y2_n.copy()[start:end])
            

    #x_diff= [x2_n - x1_n]
    df2 = pd.DataFrame(list(zip(letter_n, x1_n, x2_n, y1_n, y2_n)),
                columns =['letter', 'x1', 'x2', 'y1', 'y2'] )   
    return df2
def rewrite_asc( file_name:str, file_dir:str, img_dir:str):
    """Goes through asc file from eye-tracker and fixes issues.
    Creates new asc file in subfolder ./new"""

    # open the .asc file
    with open(file_dir + file_name + '.asc',"r") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        
    # now we go line by line to modify/ add stuff as needed:
    new_file= []
    img_file= ''
    D= 0
    cond= 0
    item= 0
    ET_flag= ''
    wait_flag=0 
    trial_seq= 0


    for i in range(len(lines)):
        
        curr_line= lines[i]
        
        if '!V TARGET_POS TARG1' in curr_line:
            continue # get rid of annoying EB flags
        
        # fix display coords issue:
        if 'DISPLAY_COORDS' in curr_line:
            curr_line= curr_line.replace('DISPLAY_COORDS', 'DISPLAY COORDS')
        
        if 'IMGLOAD CENTER' in curr_line: # image loaded for current trial
            img_file= curr_line.split(' ')[4]
            wait_flag= 1
            trial_seq= trial_seq+1
            
            if 'text' in img_file: # check if it is item
                D= 0                
            elif 'question' in img_file: # check if it is item
                D= 1       
                
            if 'TNR' in img_file: # times new roman font
                cond= 1                
            elif 'OD' in img_file: # open dyslexia font
                cond= 2
                
            if D== 0: # if not question...
                
                # get numbers, second number is always item number:
                item= int(re.findall(r'\d+', img_file)[1])
            
            # if item is a question, take last item number (since item is not updated above)
            ET_flag= 'TRIALID ' + 'E' + str(cond)+'I' +str(item)+ 'D' +str(D)
            
            ## extract text coordinates from image:
            im_path = list(pl.Path(img_dir).rglob(f"*{img_file}"))
            assert len(im_path) >0, "no matching images found"

            df= text_coords(str(im_path[0]))            

        if 'TRIALID' in curr_line:
            if wait_flag==0:
                
                if trial_seq>0:
                    msg_flag= curr_line.split(' ')[0]
                    
                    ## add trial end flags
                    new_file.append(msg_flag + ' ENDBUTTON 5')
                    new_file.append(msg_flag + ' DISPLAY OFF')
                    new_file.append(msg_flag + ' TRIAL_RESULT 5')
                    new_file.append(msg_flag + ' TRIAL OK')
                
                continue # don't add the current flag so that we have one flag per trial
            else:
                # if 'TRIALID' in curr_line and wait_flag==1:
            
                # replace flag with umass convention, so we can open it in EyeDoctor
                msg_flag= curr_line.split(' ')[0]
                curr_line= msg_flag+ ' ' + ET_flag
                wait_flag= 0 # reset so that it doesn't get triggered in repetition
                
                # print current line:
                new_file.append(curr_line)
                
                ### Print text coordinates:
                new_file.append(msg_flag + ' DISPLAY TEXT 1')
            
                for i in range(len(df)):
                    new_file.append(msg_flag + 'REGION CHAR %d 1 %s %d %d %d %d' % (i, df.letter[i], df.x1[i], df.y1[i], df.x2[i], df.y2[i]))
                    new_file.append(msg_flag + ' DELAY 1 MS')

                
                
                # print start flags:
                msg_flag= curr_line.split(' ')[0]
                new_file.append(msg_flag + ' GAZE TARGET ON')
                new_file.append(msg_flag + ' GAZE TARGET OFF')
                new_file.append(msg_flag + ' DISPLAY ON')
                new_file.append(msg_flag + ' SYNCTIME')
                
                continue
        
        if 'SYNCTIME' in curr_line:
            continue
        #    msg_flag= curr_line.split(' ')[0]
        #    new_file.append(msg_flag + ' GAZE TARGET ON')
        #    new_file.append(msg_flag + ' GAZE TARGET OFF')
        #    new_file.append(msg_flag + ' DISPLAY ON')
        
        # append current line to new file once all changes have been done:
        new_file.append(curr_line)

    with open(file_dir + 'new/'+ file_name + "_new.asc", 'w') as f:
        f.write('\n'.join(new_file))