import pytesseract
import os

if os.path.exists("C:/Program Files/Tesseract-OCR/tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


import re
import pathlib as pl

from get_coords import text_coords


def rewrite_asc( file_name:str, file_dir:str, img_dir:str):
    """Goes through asc file from eye-tracker and fixes issues.
    Creates new asc file in folder {oldname}_new"""

    # open the .asc file
    print(f"Reading in file: {file_dir + file_name + '.asc'}")
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

            df= text_coords(str(im_path[0]),use_image_to_data=True,use_reference_widths=False,plot_examples=False,binarise=True,add_border=False,upscale_im=True)            
            # print(f"Created letter bounding boxes for {str(im_path[0])}")
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
    file_dir_path = pl.Path(file_dir).parent.joinpath(f"{pl.Path(file_dir).stem}_new")
    file_dir_path.mkdir(exist_ok=True,parents=False)
    # print(f'Writing file: {file_dir_path.joinpath(file_name + "_new.asc")}')
    with open(file_dir_path.joinpath(file_name + "_new.asc") , 'w') as f:
        f.write('\n'.join(new_file))
    return f'Written file: {file_dir_path.joinpath(file_name + "_new.asc")}'

if __name__ == '__main__':
    print("not for executing")