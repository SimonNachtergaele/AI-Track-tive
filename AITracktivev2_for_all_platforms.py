"""Spyder Editor

This is a Python 3 script written by PhD student Simon Nachtergaele in 2020-2021 
This script is (intelectual) property of FWO Flanders and Ghent University.

Contact: Simon.Nachtergaele@UGent.be

requirements for .exe export:
    microscope.ico in same folder as .exe file
    icon-01.png also in same folder as .exe file
    logo-01.png also in the same folder
    
v1.6 has a logger which makes it much easier to debug. 
v1.7 incorporates the comments of editor Pieter Vermeesch: polygon drawing visually enabled and space in stead of escape
v1.7 also has a few parameters changed in order to gain ~90% succes rate in most cases 
v1.8 is succesfully tested on Windows, Linux and MAC-OS (thanks to Sharmaine Verhaert!)
v1.8 also has the live function which enables live identification of the fission tracks on your screen (when using two screens)
v1.9 has the functionality to add tracks and produce yolo-compatible .txt files 
v1.9 has also the possibility to use something else than middle mouse button, so it can be run on a laptop
v1.9 has live track recognition for apatite and mica DNN


Development ideas for this script:  
 - give an error when the circle (from LAFT) is plotted next to the screen 
    # EDIT: work with 4 points and check if these four coordinates are inside the frame
 - start with erroneously recognized fission tracks in stead of the other 
    # EDIT: use space first if no tracks needed to be added. Fill this in the user manual or instructions. 
 - Drag the polygonal mask to somewhere else (Ana Fonseca asked this)
    
 Future improvements:
 - find the confined tracks (make another class and train a new model)
   # EDIT: could be used for finding live confined tracks when implemented in some software 

 Based on: https://www.youtube.com/watch?v=_FNfRtXEbr4

 Make an .exe using this link https://wasi0013.com/2017/10/03/making-a-stand-alone-executable-from-a-python-script-using-pyinstaller/
 
 very well explained here: 
 https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e#:~:text=Anchor%20box%20is%20a%20prior,means%20on%20the%20entire%20dataset.&text=In%20our%20later%20training%2C%20instead,offsets%20to%20these%20bounding%20boxes.
"""


version_programme = str('v2')
print('AI-Track-tive '+str(version_programme)+' starting')
print('If you want to display console in your Python IDE: comment line with "print = logger.info" ')
security_version = "unsecured"
import cv2
import numpy as np
import glob
import random 
import os
import sys
import platform
import pandas as pdd
import pickle #needed for storing the defaults
import webbrowser

from tkinter import *
from tkinter import font
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilenames
from inspect import currentframe, getframeinfo
from string import ascii_letters, digits
# Logger
import logging
# create logger with 'spam_application'
logger = logging.getLogger('AI-Track-tive '+str(version_programme))
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('info.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(fh)
logger.addHandler(ch)

logger.info('\n starting the application')

# change print statement to logger 
print = logger.info  #comment this if you want to print the console here 

error=''

# Get working directory
path_script=os.path.realpath(__file__)
path_script_basename=os.path.basename(path_script)
path_script_dirname=os.path.dirname(path_script)

# Print console elsewhere in the log file 
cwd=os.getcwd() 
print('current working directory is '+str(cwd))

if platform.system()=='Linux':
    print('Linux found')

elif platform.system()=='Windows':
    print('Windows found')
    
     #Get Windows version
    windows_version = int(platform.release())
     
    if windows_version!=int(10):
        print('windows version '+str(windows_version))
        print('Windows 10 required for full compatibility')
    else:
        print('Windows 10 detected')
    
elif platform.system()=='Darwin':
    print('Mac found')

# Define step i.e. systematic error of the Laplacian filter
step=0 # maybe take one or two at maximum

# Make a callable variable 
fail='no'

sys.setrecursionlimit(10000000)   #otherwise python doesn't want to make an .exe

# Get operating system, source: https://www.programcreek.com/python/example/2044/sys.getwindowsversion
def get_winver():
    if not WINDOWS:
        raise NotImplementedError("not WINDOWS")
    wv = sys.getwindowsversion()
    if hasattr(wv, 'service_pack_major'):  # python >= 2.7
        sp = wv.service_pack_major or 0
    else:
        r = re.search(r"\s\d$", wv[4])
        if r:
            sp = int(r.group(0))
        else:
            sp = 0
    return (wv[0], wv[1], sp)

#frameinfo = getframeinfo(currentframe())
#print(frameinfo.filename, frameinfo.lineno)

#==============================================================================
# SET SOME DEFAULT VALUES
#==============================================================================
# If it's the first time that the programme is used: just pick some defaults 
emailaddress=str("computercounttracksbetter@notimetolose.com")
# Default key
key_default="default" 
# Default emailaddress
emailaddress_default="lazyfissiontracker@gmail.com"
# Defaults for Simon's pc: 
location_model_apatite=str(path_script_dirname)+r"\INPUT\yolov3_training_last_apatite_804px_2000it_10DUR5BC.weights"    
# Choose which model to use for mica
location_model_mica=str(path_script_dirname)+r"\INPUT\yolov3_training_last_25mica_804px_4000it_13082020.weights"
# Choose which testing model 
location_model_testing=str(path_script_dirname)+r"\INPUT\yolov3_testing.cfg"
# Choose output directory
output_directory=str(path_script_dirname)+r"\OUTPUT"
# Choose pix
pix_def = 1608
# Choose dist_enty
dist_def = 117.5
# Resolution
resolution_def=str("1920x1080")
# Shutdown script value 
shutdown_script = 0 
# Default spot diameter (will be changed later)
spot_diameter=float(5)

# Pickle default settings document
# source = https://stackoverflow.com/questions/26835477/pickle-load-variable-if-exists-or-create-and-save-it
try:
    imported_pickle = pickle.load(open("savedpathlocations.pkl","rb"))
    print('pickle found')
except (OSError, IOError, EOFError) as e:
    print('new pickle made')
    if security_version == "secured":
        print('exception found in the pickle')
        imported_pickle = [key_default, emailaddress_default, location_model_apatite, location_model_mica,location_model_testing,output_directory, pix_def, dist_def,resolution_def]
        pickle.dump(imported_pickle,open("savedpathlocations.pkl","wb"))
    else:
        imported_pickle = [location_model_apatite, location_model_mica,location_model_testing,output_directory, pix_def, dist_def,resolution_def]
        pickle.dump(imported_pickle,open("savedpathlocations.pkl","wb"))
    
with open('savedpathlocations.pkl','rb') as f:
    print('pickle openened')
    if security_version == "secured":
        [key_default, emailaddress_default, location_model_apatite, location_model_mica,location_model_testing,output_directory, pix_def, dist_def, resolution_def] = pickle.load(f)
    else:
        [location_model_apatite, location_model_mica,location_model_testing,output_directory, pix_def, dist_def, resolution_def] = pickle.load(f)
        
#==============================================================================
# CREATE A INITIATOR WINDOW
#==============================================================================

root_intro = Tk()

def on_closing_root_intro():
    if messagebox.askokcancel("Quit","Do you want to quit?"):
        root_intro.destroy()
        os._exit(00)   # restart kernel 
root_intro.protocol("WM_DELETE_WINDOW",on_closing_root_intro)

# FIND THE SCREEN PROPERTIES
screen_width_computer = root_intro.winfo_screenwidth()
screen_height_computer = root_intro.winfo_screenheight()

#place intro window somewhere 
root_intro.geometry('+500+150')

#Get scale factor
if platform.system()=='Windows':
        
    if windows_version==int(10):
        import ctypes
        scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100  # apparently only works for windows 8 and higher... 
        
    else: # Windows 7 for example
        scaleFactor = 1.0
        
elif platform.system()=='Linux':
    print('Linux detected')
    
    
elif platform.system()=='Darwin':
    print('Apple detected')
    # Source: https://gist.github.com/justvanrossum/9843bf52d93cbe1c7a3f37420bea8d34
    from AppKit import NSScreen, NSDeviceSize, NSDeviceResolution
    from Quartz import CGDisplayScreenSize
    
    for i, screen in enumerate(NSScreen.screens(), 1):
        description = screen.deviceDescription()
        pw, ph = description[NSDeviceSize].sizeValue()
        rx, ry = description[NSDeviceResolution].sizeValue()
        mmw, mmh = CGDisplayScreenSize(description["NSScreenNumber"])
        scaleFactor = screen.backingScaleFactor()
        pw *= scaleFactor
        ph *= scaleFactor
        print(f"display #{i}: {mmw:.1f}×{mmh:.1f} mm; {pw:.0f}×{ph:.0f} pixels; {rx:.0f}×{ry:.0f} dpi")

else:
    print('no operating system detected') 
    scaleFactor = 1 #best guess

print('scalefactor is '+str(scaleFactor))

#root_intro.eval('tk::PlaceWindow . center') # command trying to center the window
root_intro.title('AI-Track-tive')

# Change icon
from PIL import Image
#root_intro.iconbitmap('microscope.ico')  #LINUX

# Change font
myFont = font.Font(size=11,weight="bold")

# Change background
root_intro.configure(bg='white')

# Insert logo from AI-Track-tive
import PIL.Image
import PIL.ImageTk
im = PIL.Image.open("logo-01.png")
photo = PIL.ImageTk.PhotoImage(im)

# Design lables
label = Label(root_intro, image=photo, bg="white")
label.image = photo
label.pack()

label_welcome = Label(root_intro, bg="white", text = "\n Welcome to AI-track-tive " +str(version_programme)+ "\n It is developed by Simon Nachtergaele")
label_welcome['font'] = myFont
label_welcome.pack()

# Make a function that opens a browser
def callback(url):
    webbrowser.open_new(url)
    
# link1 = Label(root_intro, bg="white", text="Click here for tutorial videos (under construction)", fg="blue", cursor="hand2")
# link1.pack()
# link1.bind("<Button-1>", lambda e: callback("https://www.youtube.com/watch?v=fSfit87vkrA"))

link3 = Label(root_intro, bg="white", text="Click here to contact Simon", fg="blue", cursor="hand2")
link3.pack()
link3.bind("<Button-1>", lambda e: callback("https://telefoonboek.ugent.be/en/people/802001843844"))

link4 = Label(root_intro, bg="white", text="Check the tutorials", fg="blue", cursor="hand2")
link4.pack()
link4.bind("<Button-1>", lambda e: callback("https://users.ugent.be/~smanacht/download.html"))

link5 = Label(root_intro, bg="white", text="Click here to see updates on GitHub", fg="blue", cursor="hand2")
link5.pack()
link5.bind("<Button-1>", lambda e: callback("https://github.com/SimonNachtergaele/AI-Track-tive"))

link2 = Label(root_intro, bg="white", text="Click here to read the terms of use", fg="blue", cursor="hand2")
link2.pack()
link2.bind("<Button-1>", lambda e: callback("https://creativecommons.org/licenses/by-nc-sa/4.0/"))


# License, emailaddress
if security_version == "secured":
    address_intro = Label(root_intro, text="Please enter your email address", bg='white', anchor='w').pack(fill='both')
    address_intro_entry = StringVar(root_intro,value=emailaddress_default)
    address_intro_entry = Entry(root_intro, width = round(70), textvariable=address_intro_entry) 
    address_intro_entry.pack()

# Key
if security_version == "secured":
    #key_default='fill in your key here'
    key_intro = Label(root_intro, text="Enter the software key Simon gave you", bg='white', anchor='w').pack(fill='both')
    key_intro_entry = StringVar(root_intro,value=key_default)
    key_intro_entry = Entry(root_intro, width = round(70), textvariable=key_intro_entry) 
    key_intro_entry.pack()
    
if security_version =="unsecured":
    unsecured_version_information = Label(root_intro, text="Please enjoy the unsecured version of this software", bg='white').pack(fill='both')
    license_button_intvar = IntVar()
    license_button = Checkbutton(root_intro, text="I agree to the terms of use and accept to use this under a CC BY-NC-SA 4.0 license", bg='white', variable = license_button_intvar, cursor="hand2").pack(fill='both')
    
    
space = Label(root_intro, text=" ",bg='white').pack()

# Function for closing the window
def close_intro_window2():
    # if it's the secured version, than we need to store the password and emailadress to use it later
    if security_version == "secured":
        # Make attribute for inserted key
        close_intro_window2.key=key_intro_entry.get()
        # Make attribute for emailaddress
        close_intro_window2.emailaddress=address_intro_entry.get()
    # Close the window
    if license_button_intvar.get()==0:
        error_not_agreeing=Label(root_intro,text="please agree to the terms of use",bg='white').pack(fill='both')
    if license_button_intvar.get()==1:
        root_intro.destroy() 

#Continue button
quit_button = Button(root_intro,text = "Continue", command = close_intro_window2, cursor="hand2").pack()
# Add a space
space = Label(root_intro, text=" ",bg='white').pack()

root_intro.mainloop()        

#==============================================================================
#  SPECIFY DIRECTORIES WINDOW
#==============================================================================

# Build a new root for specifying the directories
root_directories = Tk()
root_directories.title('Insert all the necessary information below')
root_directories.geometry('+400+150')

# Function that starts when you close the window 
def on_closing_root_directories():
    if messagebox.askokcancel("Quit","Do you want to quit?"):
        root_directories.destroy()
        os._exit(00)   # restart kernel 
root_directories.protocol("WM_DELETE_WINDOW",on_closing_root_directories)

# Icon
#root_directories.iconbitmap('microscope.ico') #LINUX 

# Build new label
label_intro2 = Label(root_directories, text="Enter the location of the deep neural network for apatite. For yolov3 DNN's, this is a file ending on .weights", anchor='w').pack(fill='both')

# Apatite model 
location_model_apatite = StringVar(root_directories, value=location_model_apatite)
location_model_apatite = Entry(root_directories, width = round(150), textvariable=location_model_apatite) 
location_model_apatite.pack()

# Mica model
label_intro2 = Label(root_directories, text="Enter the location of the deep neural network for mica. For yolov3 DNN's, this is a file ending on .weights", anchor='w').pack(fill='both')
location_model_mica = StringVar(root_directories, value=location_model_mica)
location_model_mica = Entry(root_directories, width = round(150), textvariable=location_model_mica) 
location_model_mica.pack()

# Testing model
label_intro3 = Label(root_directories, text="Enter the location of the configuration file from the neural network. For yolov3 DNN's, this is a file ending on .cfg",anchor='w').pack(fill='both')
location_model_testing= StringVar(root_directories, value=location_model_testing)
location_model_testing = Entry(root_directories, width = round(150),textvariable=location_model_testing) 
location_model_testing.pack()

# Output directory
label_intro4 = Label(root_directories, text="Enter the output directory below",anchor='w').pack(fill='both')
output_directory = StringVar(root_directories, value=output_directory)
output_directory = Entry(root_directories, width = round(150), textvariable = output_directory)
output_directory.pack()

# Image size 
label_intro5 = Label(root_directories, text="How wide is your image in pixels? (use dots and no commas as a decimal separator)",anchor='w').pack(fill='both')
pix = StringVar(root_directories, value=pix_def)
pix_entry = Entry(root_directories, width = round(150), textvariable = pix)
pix_entry.pack()

# Real image size 
label_intro6 = Label(root_directories, text="How wide is your image in micrometer? (use dots and no commas as a decimal separator)",anchor='w').pack(fill='both')
dist = StringVar(root_directories, value=dist_def)
dist_entry = Entry(root_directories, width = round(150), textvariable = dist)
dist_entry.pack()

# Real image size 
label_intro7 = Label(root_directories, text="What is your screen resolution? (e.g. 1200x900)",anchor='w').pack(fill='both')
resolution = StringVar(root_directories, value=resolution_def)
resolution_entry = Entry(root_directories, width = round(150), textvariable = resolution)
resolution_entry.pack()

# If count is chosen:
def inactivator_buttons_dpar_live_txt():
    if var_manually_review.get()==1:
        annotate_button.deselect()
        live_button.deselect()
        dpar_button.deselect()
        live_button_mica.deselect()
        
        review_manually_button.select()
    
# If dpar button has been chosen
def inactivator_buttons_count_live_txt():
    if dpar_review.get()==1:
        review_manually_button.deselect()
        annotate_button.deselect()
        live_button.deselect()
        live_button_mica.deselect()
        
        dpar_button.select()

# If live button has been chosen
def inactivator_buttons_dpar_count_txt():
    if live.get()==1:
        review_manually_button.deselect()
        annotate_button.deselect()
        dpar_button.deselect()
        live_button_mica.deselect()
        
        live_button.select()
        
# If live button has been chosen
def inactivator_buttons_dpar_count_live():
    if annotate.get()==1:
        dpar_button.deselect()
        review_manually_button.deselect()
        live_button.deselect()
        live_button_mica.deselect()
        
        annotate_button.select()
        
        
# If live button has been chosen
def inactivator_buttons_dpar_count_for_mica_txt():
    if live_mica.get()==1:
        review_manually_button.deselect()
        annotate_button.deselect()
        dpar_button.deselect()
        live_button.deselect()
        
        live_button_mica.select()
        
# Create default settings file 
def set_settings_as_default():
    
    #One example how: https://stackoverflow.com/questions/3128673/how-do-i-make-python-remember-settings
    with open('savedpathlocations.pkl','wb') as f:
        if security_version == "secured":
                
            # Check if everything is filled in 
            if location_model_apatite.get()!='' and location_model_mica.get()!='' and location_model_testing.get()!='' and output_directory.get()!='' and pix_entry.get()!='' and dist_entry.get()!='':
                # apatite model
                close_intro_window.location_model_apatite=str(location_model_apatite.get())
                # mica model
                close_intro_window.location_model_mica=str(location_model_mica.get())
                # testing model
                close_intro_window.location_model_testing=str(location_model_testing.get())
                # output directory
                close_intro_window.output_directory=str(output_directory.get())
                # pixels
                close_intro_window.pix_entry = float(pix_entry.get())
                # distance 
                close_intro_window.dist_entry = float(dist_entry.get()) 
                #resolution
                close_intro_window.resolution = str(resolution_entry.get())
            
            # If not everything is filled in, then there needs to be written that the program requires more information 
            else:
                error_message_directories = Label (root_directories, text='please specify all necessary information')
                error_message_directories.pack()
            
            pickle.dump([close_intro_window2.key,
                     close_intro_window2.emailaddress,
                     close_intro_window.location_model_apatite, 
                     close_intro_window.location_model_mica,
                     close_intro_window.location_model_testing,
                     close_intro_window.output_directory, 
                     close_intro_window.pix_entry, 
                     close_intro_window.dist_entry,
                     close_intro_window.resolution],f) 
        
        # If it's the unsecured version
        else: 
            
            # Check if everything is filled in 
            if location_model_apatite.get()!='' and location_model_mica.get()!='' and location_model_testing.get()!='' and output_directory.get()!='' and pix_entry.get()!='' and dist_entry.get()!='':
                # apatite model
                close_intro_window.location_model_apatite=str(location_model_apatite.get())
                # mica model
                close_intro_window.location_model_mica=str(location_model_mica.get())
                # testing model
                close_intro_window.location_model_testing=str(location_model_testing.get())
                # output directory
                close_intro_window.output_directory=str(output_directory.get())
                # pixels
                close_intro_window.pix_entry = float(pix_entry.get())
                # distance 
                close_intro_window.dist_entry = float(dist_entry.get()) 
                #resolution
                close_intro_window.resolution = str(resolution_entry.get())
            
            # If not everything is filled in, then there needs to be written that the program requires more information 
            else:
                error_message_directories = Label (root_directories, text='please specify all necessary information')
                error_message_directories.pack()

            pickle.dump([close_intro_window.location_model_apatite, 
                     close_intro_window.location_model_mica,
                     close_intro_window.location_model_testing,
                     close_intro_window.output_directory, 
                     close_intro_window.pix_entry, 
                     close_intro_window.dist_entry,
                     close_intro_window.resolution],f) 
# Button to click when the settings are saved 
default_button = Button(root_directories,text = "Click here if you want to save the above information", command = set_settings_as_default, cursor="hand2").pack()

# Review tracks manually button
var_manually_review = IntVar()
review_manually_button = Checkbutton(root_directories, text="count tracks and review manually",command = inactivator_buttons_dpar_live_txt, variable=var_manually_review, cursor="hand2")
review_manually_button.pack()

# Create button to start dpar measurements
dpar_review = IntVar()
dpar_button = Checkbutton(root_directories, text="perform dpar measurement",command=inactivator_buttons_count_live_txt, variable=dpar_review, cursor="hand2")
dpar_button.pack()

# Live tracks: apatite DNN
live = IntVar()
live_button = Checkbutton(root_directories, text="let the apatite DNN find tracks on my live screen", command=inactivator_buttons_dpar_count_txt, variable=live, cursor="hand2")
live_button.pack()

# Live tracks: mica DNN
live_mica = IntVar()
live_button_mica = Checkbutton(root_directories, text="let the mica DNN find tracks on my live screen", command=inactivator_buttons_dpar_count_for_mica_txt, variable=live_mica, cursor="hand2")
live_button_mica.pack()

# Annotate the tracks manually
annotate = IntVar()
annotate_button = Checkbutton(root_directories, text="make (yolov3) .txt files by annotating the tracks", command=inactivator_buttons_dpar_count_live, variable=annotate, cursor="hand2")
annotate_button.pack()

# If the user wants to read the instructions
instructions_window = IntVar()
instructions_window_button = Checkbutton(root_directories, text="show instruction windows", variable=instructions_window, cursor="hand2")
instructions_window_button.select()
instructions_window_button.pack()

def close_intro_window():
    # Check if everything is filled in 
    if location_model_apatite.get()!='' and location_model_mica.get()!='' and location_model_testing.get()!='' and output_directory.get()!='' and pix_entry.get()!='' and dist_entry.get()!='':
        
        if dpar_review.get()==1 and var_manually_review.get()==0 and live.get()==0 or var_manually_review.get()==1 and dpar_review.get()==0 and live.get()==0 or live.get()==1 and var_manually_review.get()==0 and dpar_review.get()==0 or live_mica.get()==1 and var_manually_review.get()==0 and dpar_review.get()==0 or annotate.get()==1 and live.get()==0 and var_manually_review.get()==0 and dpar_review.get()==0:
        
            # apatite model
            close_intro_window.location_model_apatite=str(location_model_apatite.get())
            # mica model
            close_intro_window.location_model_mica=str(location_model_mica.get())
            # testing model
            close_intro_window.location_model_testing=str(location_model_testing.get())
            # output directory
            close_intro_window.output_directory=str(output_directory.get())
            # pixels
            close_intro_window.pix_entry = float(pix_entry.get())
            # distance 
            close_intro_window.dist_entry = float(dist_entry.get()) 
            # resolution
            close_intro_window.resolution = str(resolution_entry.get())
            # Close the window
            root_directories.destroy() 
            
        # If not everything is filled in, then there needs to be written that the program requires more information 
        else:
            error_message_directories = Label (root_directories, text='please specify all necessary information')
            error_message_directories.pack()
    
    # If not everything is filled in, then there needs to be written that the program requires more information 
    else:
        error_message_directories = Label (root_directories, text='please specify all necessary information')
        error_message_directories.pack()
    

            

space = Label(root_directories, text=" ")
space.pack()

quit_button_intro = Button(root_directories,text = "Continue", command = close_intro_window, cursor="hand2")
quit_button_intro.pack()

root_directories.mainloop()
   
    
#==============================================================================
# CREATE FUNCTION TO CONVERT LIST OF LISTS WITH TRACK RECTANGLES TO TXT FILE FOR LABELIMG
#==============================================================================

def labelImgformatter(rectangles, name):
    # Every .txt starts with a 15 and space
    s = ''
    for r in rectangles:
        r_converted = str('') # make an empty to string to append the values to
        for value in r:
            r_converted += str('15 ')+str(value)
        s+=str(r_converted)+str('\n')

    # Change the working directory
    os.chdir(close_intro_window.output_directory)

    # Create a text file with all the annotations     
    text_file = open(name[0][:-4]+".txt", "w", encoding='utf-8')
    text_file.write(s)
    text_file.close()
    
    print('Created a txt file with the track locations using labelimgformatter')
    print('name is '+str(name[0][:-4])+str('.txt'))
    
    # Class file 
    text_file_classes = open('classes.txt','w', encoding='utf-8')
    text_file_classes.write('Track')
    text_file_classes.close()
    print('created class file')
    
    return s

#==============================================================================
# CREATE CLASS MISSINGTRACKSAP 
#==============================================================================

class MissingTracksAp(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our rectangle
        self.missed_tracks = [] # list of coordinates for rectangles
        self.latest_mouse_event = 2 # default value  
        self.points_fpt = []
        self.false_positives_tracks = [] 
        self.stop = False

    def on_mouse(self, event, x, y, flags, *user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        #k = cv2.waitKey(1) # changed it from one to zero              
            
        if flags == 8: # NEW
            print('flag 8: control pressed and left mouse button ') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)

        elif flags == 9: # NEW
            print('flag 9: control and left mouse button pressed ') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Start dragging a rectangle on a missed track
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            self.stop = False 
            
        elif event == cv2.EVENT_LBUTTONUP:
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            self.latest_mouse_event=1
            print("Completing polygon with %d points." % len(self.points))
            
            if len(self.points) == 2 and not self.points[0]==self.points[1]:
                self.missed_tracks.append(self.points)
                self.points=[]
                self.done = True
            else:
                pass
            print("the current list of " + str(len(self.missed_tracks)) + " missed tracks is "+ str(self.missed_tracks))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            print('right button down')
            self.points_fpt.append((x, y))
            
        elif event == cv2.EVENT_RBUTTONUP:
            print('right button up')
            self.points_fpt.append((x, y))
            self.latest_mouse_event=0
            print(self.points_fpt[0])
            print(self.points_fpt[1])
            if not self.points_fpt[0]==self.points_fpt[1]:
                print('points are not the same')
                self.false_positives_tracks.append(self.points_fpt)
            print('appended erroneously recognised track')
            self.points_fpt=[]
            self.done=True
    
    
        elif event == cv2.EVENT_MOUSEWHEEL: 
            
            print ("mousewheel detected")
            bar = cv2.getTrackbarPos('RL/TL', title_window)
            
            if bar == 1:  
                cv2.setTrackbarPos('RL/TL', title_window,0)
                dst = cv2.addWeighted(croppedimage, 0, croppedimage_epi, 1, 0.0)
                cv2.imshow(title_window, dst)
                print('trackbar changed to 0')
            
            else:
                cv2.setTrackbarPos('RL/TL', title_window,1)
                dst = cv2.addWeighted(croppedimage, 1, croppedimage_epi, 0, 0.0)
                cv2.imshow(title_window, dst) 
                print('trackbar changed to 1')

    def findtracksmanually(self):
        print("findtracksmanually function starts")
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)  # This comment creates a window but loads an image later
        #cv2.imshow("should",self.window_name)
        #cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)       
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        cv2.waitKey(1)    # efkes uit gezet  

        while(not self.done):          
            #Here you find code to change the RL/TL with space bar    

            # If there are tracks missed and added manually
            if (len(self.missed_tracks) >= 1): 
                 # Draw all the current polygon segments  
                 cv2.polylines(croppedimage, np.array([self.points]), False, FINAL_LINE_COLOR, 1) 
                 # And  also show what the current segment would look like 
                 # ... skipped this because extra lines are added now to the image
                 cv2.line(croppedimage, self.points[-1], self.current, WORKING_LINE_COLOR)   
            # Update the window
            cv2.imshow("Manual Review process", croppedimage) 
            
            # If space is pressed
            if cv2.waitKey(0)== 32:
                print('space pressed in line 750')
                self.done = True   
        print('function findtrackmanually ended')        
        
        return self.window_name  #should be an object, not a list             

    def list_manually_found_tracks(self):
        print('manually indicated tracks list')
        print(self.missed_tracks)
        return self.missed_tracks 
    
    def list_manually_found_tracks_mistaken(self):
        return self.false_positives_tracks  
    
    def latest_track_found(self):
        # If a track was added
        if self.latest_mouse_event==1: 
            return 1
        
        # If a mistaken track was added
        elif self.latest_mouse_event==0:
            return 0
        
        # If space was hit or something else
        elif self.latest_mouse_event==10:
            return 2
        
        else:
            self.latest_mouse_event=2
            return 1
        
    
#==============================================================================
# CREATE CLASS MISSINGTRACKSANNOTATE 
#==============================================================================


# Copied the MissingTracksAp class but removed the right mouse button functionalities 
class MissingTracksAnnotate(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our rectangle
        self.missed_tracks = [] # list of coordinates for rectangles
        self.latest_mouse_event = 2 # default value  
        self.points_fpt = []
        self.false_positives_tracks = [] # 
        self.stop = False
        

    def on_mouse(self, event, x, y, flags, *user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        #k = cv2.waitKey(1) # changed it from one to zero        
        
        if flags == 8: # NEW
            print('flag 8: control pressed and left mouse button') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            print('middlemousebutton clicked')
            print('press space to continue')
            self.stop = True
            self.done = True
            

        elif flags == 9: # NEW
            print('flag 9: control and left mouse button pressed ') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Start dragging a rectangle on a missed track
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            self.stop = False 
            
        elif event == cv2.EVENT_LBUTTONUP:
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            self.latest_mouse_event=1
            print("Completing polygon with %d points." % len(self.points))
            
            if len(self.points) == 2 and not self.points[0]==self.points[1]:
                self.missed_tracks.append(self.points)
                self.points=[]
                self.done = True
            else:
                pass
            print("the current list of " + str(len(self.missed_tracks)) + " missed tracks is "+ str(self.missed_tracks))
        

    def findtracksmanually(self):
        print("findtracksmanually function starts")
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)  # This comment creates a window but loads an image later
        #cv2.imshow("should",self.window_name)
        #cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)       
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        cv2.waitKey(1)      

        while(not self.done):          
            #Here you find code to change the RL/TL with space bar    

            # If there are tracks missed and added manually
            if (len(self.missed_tracks) >= 1): 
                 # Draw all the current polygon segments  
                 cv2.polylines(croppedimage, np.array([self.points]), False, FINAL_LINE_COLOR, 1) 
                 # And  also show what the current segment would look like 
                 # ... skipped this because extra lines are added now to the image
                 cv2.line(croppedimage, self.points[-1], self.current, WORKING_LINE_COLOR)   
            # Update the window
            cv2.imshow("Manual Review process", croppedimage) 
            # If space is pressed
            if cv2.waitKey(0)== 32: 
                self.done = True   
        print('function findtrackmanually ended')        
        
        return self.window_name  #should be an object, not a list             

    def list_manually_found_tracks(self):
        print('manually indicated tracks list')
        print(self.missed_tracks)
        return self.missed_tracks 
    
    def list_manually_found_tracks_mistaken(self):
        return self.false_positives_tracks  
    
    def latest_track_found(self):
        # If a track was added
        if self.latest_mouse_event==1: 
            return 1
        
        # If a mistaken track was added
        elif self.latest_mouse_event==0:
            return 0
        
        # If space was hit or something else
        elif self.latest_mouse_event==10:
            return 2
        
        else:
            self.latest_mouse_event=2
            return 1
 
#==============================================================================
# CREATE CLASS MISSINGTRACKSMICA
#==============================================================================
      
class MissingTracksMica(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our rectangle
        self.missed_tracks = [] # list of polygons        
        self.latest_mouse_event = 2 
        self.points_fpt = []
        self.false_positives_tracks = []
        self.stop = False

    def on_mouse(self, event, x, y, flags, user_param):
 
        if flags == 8: # NEW
            print('flag 8: control pressed and left mouse button') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            print('middlemousebutton clicked')
            print('press space to continue')
            self.stop = True
            self.done = True

        elif flags == 9: # NEW
            print('flag 9: control and left mouse button pressed ') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            self.stop = False
            
        elif event == cv2.EVENT_LBUTTONUP:
            # Right click means we're done
            self.points.append((x, y))
            print("Completing polygon with %d points." % len(self.points))
            self.latest_mouse_event = 1 
            print("the current list of " + str(len(self.missed_tracks)) + " missed tracks is "+ str(self.missed_tracks))
            
            if len(self.points) == 2 and not self.points[0]==self.points[1]:
                self.missed_tracks.append(self.points)
                self.points=[]
                self.done = True
            else:
                pass
            
            print("the current list of " +str(len(self.missed_tracks))+" missed tracks is "+str(self.missed_tracks))
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            print('right button down')
            self.points_fpt.append((x, y))
            
        elif event == cv2.EVENT_RBUTTONUP:
            print('right button up')
            self.points_fpt.append((x, y))
            self.latest_mouse_event = 0
            print(self.points_fpt[0])
            print(self.points_fpt[1])
            if not self.points_fpt[0] == self.points_fpt[1]:
                print('points are not the same')
                self.false_positives_tracks.append(self.points_fpt)
            print('appended erroneously recognised track')
            self.points_fpt = []
            self.done = True
                       
        elif event == cv2.EVENT_MOUSEWHEEL:
            print ("mousewheel detected")
            
            bar = cv2.getTrackbarPos('RL/TL', title_window)
            
            if bar == 1:  
                cv2.setTrackbarPos('RL/TL', title_window,0)
                dst = cv2.addWeighted(croppedimage_mica, 0, croppedimage_mica_epi, 1, 0.0)
                cv2.imshow(title_window, dst)
                print('trackbar changed to 0')
    
            else:
                cv2.setTrackbarPos('RL/TL', title_window,1)
                dst = cv2.addWeighted(croppedimage_mica, 1, croppedimage_mica_epi, 0, 0.0)
                cv2.imshow(title_window, dst) 
                print('trackbar changed to 1')
    
    def findtracksmanually(self):
        print("findtracksmanually function starts")
        # Let's create our working window and set a mouse callback to handle events
        # This comment below creates a window but loads an image later
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)  
        #cv2.imshow("should",self.window_name)
        #cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))      
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        #cv2.waitKey(1)


        while (self.done == False):  
            # If there are tracks missed and added manually
            if (len(self.missed_tracks) >= 1): 
                # Draw all the current polygon segments
                cv2.polylines(croppedimage_mica, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(croppedimage_mica, self.points[-1], self.current, WORKING_LINE_COLOR)   #skipped this because extra lines are added now to the image
            # Update the window
            cv2.imshow("Manual Review process", croppedimage_mica)   # this is the main difference compared to the MissingTracksAp program
            # If space is pressed
            if cv2.waitKey(0)== 32:
                self.done = True
                
        return self.window_name  #should be an object, not a list             

    def list_manually_found_tracks(self):
        return self.missed_tracks
    
    def list_manually_false_tracks (self):
        return self.false_positives_tracks
    
    def latest_track_found(self):
        #If a track was added
        if self.latest_mouse_event==1:
            return 1
        elif self.latest_mouse_event==0:
            return 0
        else:
            self.latest_mouse_event == 2
            return 1
    
#==============================================================================
# CREATE CLASS MISSINGTRACKSMICAGLASS
#==============================================================================
    
class MissingTracksMicaGlass(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our rectangle
        self.missed_tracks = [] # list of rectangle coordinates
        self.latest_mouse_event = 2
        self.points_fpt = []
        self.false_positives_tracks = []
        self.stop = False

    def on_mouse(self, event, x, y, flags, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)       
        if flags == 8: # NEW
            print('flag 8: control pressed and left mouse button') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)

        elif flags == 9: # NEW
            print('flag 9: control and left mouse button pressed ') # NEW
            print('press space to continue')
            self.stop = True
            self.done = True
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            self.stop=False
            
        elif event == cv2.EVENT_LBUTTONUP:
            print("adding point #%d with position(%d,%d)" % (len(self.points),x,y))
            self.points.append((x, y))
            self.latest_mouse_event = 1
            print("Completing polygon with %d points." % len(self.points))
            
            if len(self.points) == 2 and not self.points[0]==self.points[1]:
                self.missed_tracks.append(self.points)
                self.points=[]
                self.done = True
            else:
                pass
            
            print("the current list of " + str(len(self.missed_tracks))+ " missed tracks is " + str(self.missed_tracks))
 
        elif event == cv2.EVENT_RBUTTONDOWN:
            print('right button down')
            self.points_fpt.append((x, y))
            
        elif event == cv2.EVENT_RBUTTONUP:
            print('right button up')
            self.points_fpt.append((x,y))
            self.latest_mouse_event = 0
            print(self.points_fpt[0])
            print(self.points_fpt[1])
            
            if not self.points_fpt[0]==self.points_fpt[1]:
                print('points are not the same')
                self.false_positives_tracks.append(self.points_fpt)
            print('appended erroneously recognised track in mica')
            self.points_fpt=[]
            self.done = True

        elif event == cv2.EVENT_MBUTTONDOWN:
            print('middlemousebutton clicked')
            print('press space to continue')
            self.stop=True
            self.done=True
            
        elif event == cv2.EVENT_MOUSEWHEEL: 
            
            print ("mousewheel detected")
            
            bar = cv2.getTrackbarPos('RL/TL', title_window)
            
            if bar == 1:  
                cv2.setTrackbarPos('RL/TL', title_window,0)
                dst = cv2.addWeighted(img_mica, 0, img_mica_epi, 1, 0.0)
                cv2.imshow(title_window, dst)
                print('trackbar changed to 0')
            
            else:
                cv2.setTrackbarPos('RL/TL', title_window,1)
                dst = cv2.addWeighted(img_mica, 1, img_mica_epi, 0, 0.0)
                cv2.imshow(title_window, dst) 
                print('trackbar changed to 1')
    
    def findtracksmanually(self):
        print("findtracksmanually function starts")
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)  
        cv2.waitKey(1)        
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        #cv2.waitKey(1)
        
        while(self.done == False):  
            if (len(self.missed_tracks) >= 1): #if there are tracks missed and added manually
                # Draw all the current polygon segments
                cv2.polylines(img_mica, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(croppedimage_mica, self.points[-1], self.current, WORKING_LINE_COLOR)   #skipped this because extra lines are added now to the image
            # Update the window
            cv2.imshow("Manual Review process", img_mica)
            # space button
            if cv2.waitKey(0)== 32:  
                self.done = True
       
        return self.window_name  #should be an object, not a list             

    def list_manually_found_tracks(self):
        return self.missed_tracks
    
    def list_manually_false_tracks(self):
        return self.false_positives_tracks
    
    def latest_track_found(self):
        # If a tracks was added
        if self.latest_mouse_event == 1:
            return 1
        # If a mistaken track was added
        elif self.latest_mouse_event == 0:
            return 0
        else:
            self.latest_mouse_event=1
            return 1

#==============================================================================
# CREATE SCALING VARIABLES AND FUNCTION POLYGONAREA 
#==============================================================================

print('screen_width from wrong tkinter function is ' + str(screen_width_computer))
print('screen_height from wrong tkinter function is ' + str(screen_height_computer))

resolution=close_intro_window.resolution
#screen_height_manual_entry = resolution
#(resolution)

# Determine the screen width and height based on the entry 
resolution_split=resolution.split('x')
screen_height=float(resolution_split[1])/float(scaleFactor)
print('manually inserted screen height is '+str(screen_height))
screen_width=float(resolution_split[0])/float(scaleFactor)
print('manually inserted screen width is '+str(screen_width))

# Below you can change the lay-out to a custom lay out
place_height=round(screen_height/5)
place_width=round(screen_width/3)

# Change to a 100 µm on 100 µm window and slice the photos
px = close_intro_window.pix_entry
print('original pixel size is '+str(px))
dist = close_intro_window.dist_entry
s1 = round(0.5*(px-(px*100)/dist))
print('s1 is ' + str(s1))
s2 = round(int(px) - int(s1)) 
print('s2 is ' + str(s2))
s3 = s2-s1
print('s3 is '+str(s3))
       
# Create a function that calculates the area of the polygon 
def PolygonArea(corners,width):
    n = len(corners) # of corners
    print('number of corners is '+str(n))
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
        
    area_pix = abs(area) / 2.0
    print('area_pix is ' + str(area_pix))
    print('width is ' +str(width))
    
    converter=float(width/100) # pix/µm
    print('converter is ' + str(converter))
    
    area_µm=area_pix/(converter*converter)
    print(str(area_µm) + 'µm²')
    
    return area_µm

#==============================================================================
# CREATE SCALING VARIABLES AND FUNCTION POLYGONAREA
#==============================================================================

class PolygonDrawerMica(object):

    def __init__(self, window_name):
        self.window_name = "press space to advance" # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
    
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        if self.done: # Nothing more to do
            return
        
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True
       
    def runmica(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        #cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        canvas_mica = img_mica 
        cv2.imshow(self.window_name, canvas_mica)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        while(not self.done):           
            if (len(self.points) > 0):
                   # Draw all the current polygon segments
                   cv2.polylines(canvas_mica, np.array([self.points]), False, (0,0,0), 1)
                   # And  also show what the current segment would look like
                   #cv2.line(canvas_mica, self.points[-1], self.current, FINAL_LINE_COLOR) #Used to be working_line_color
            
            # Update the window
            cv2.imshow(self.window_name, canvas_mica)
            # Wait 100 ms before next iteration
            if cv2.waitKey(100) == 32: # space hit
                self.done = True
                
        print('mask is being made now')
        mask_mica=np.zeros_like(img_mica)
        
        # of a filled polygon (custom)
        if (len(close_window.t) > 0):
            print('len close window t higher than 1')
            cv2.fillPoly(mask_mica, np.array([close_window.t]), FINAL_LINE_COLOR)
        # of a custom dranw polygon    
        elif len(close_window.t) == 0:
            print('een custom defined polygoon')
            cv2.fillPoly(mask_mica, np.int32([polygon_points]),FINAL_LINE_COLOR)
        
        # extra step: see https://pythonprogramming.net/lane-region-of-interest-python-plays-gta-v/
        masked_mica = cv2.bitwise_and(img_mica, mask_mica)   
        cv2.imshow(self.window_name, masked_mica)
        
        # Waiting for the user to press any key
        #cv2.waitKey()

        cv2.destroyAllWindows()
        
        return masked_mica


#==============================================================================
# CREATE CLASS CIRCULARROI
#==============================================================================

class CircularROI(object):
    def __init__(self, window_name, radius):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        #self.radius= int(10)
        #self.radius = int(round(width*radius/100))
   
    def on_mouse(self, event, x, y, buttons, user_param):       
       if self.done: # Nothing more to do
           return
   
       if event == cv2.EVENT_MOUSEMOVE:
           # We want to be able to draw the line-in-progress, so update current mouse position
           self.current = (x, y)
           
       elif event == cv2.EVENT_LBUTTONDOWN:
           # Left click means adding a point at current position to the list of points
           print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
           self.points.append((x, y))
           self.done = True      
           
           # Failed attempt to display a circle 
           #r=self.radius
           #cv2.circle(img,(x,y),20,(0,0,0),1)
   
    def mask(self,spot_diameter_chosen):
        print('mask function started')
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name,self.on_mouse)
        cv2.waitKey(1) 
        while(not self.done):
            canvas_ap = img 
            
            # if (len(self.points) > 0):
            #     # Draw all the current polygon segments
            #     cv2.polylines(canvas_ap, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
            #     # And  also show what the current segment would look like
            #     cv2.line(canvas_ap, self.points[-1], self.current, WORKING_LINE_COLOR)
                
            # Update the window
            cv2.imshow(self.window_name, canvas_ap)
        
            if cv2.waitKey(0) == 32: # space hit
                self.done = True
     
        mask_ap=np.zeros_like(img)
        
        print('self.points')
        print(self.points)
        center_coord=self.points[0]
        #radius=100
        print('spot diameter is'+str(spot_diameter_chosen))
        radius_µm=int(round((0.5*spot_diameter_chosen)))
        print('radius in µm is '+str(radius_µm)+str('µm'))

        thickness = int(-1)
        print(thickness)
        
        radius_pix = int(round(width*(radius_µm)/100))
        
        # of a filled polygon
        if (len(self.points) == 1):            
            cv2.circle(mask_ap, center_coord, radius_pix, (255,255,255), thickness)
       
        # extra step: see https://pythonprogramming.net/lane-region-of-interest-python-plays-gta-v/
        masked_ap = cv2.bitwise_and(img, mask_ap)   
        print('masked_ap created')
        cv2.imshow(self.window_name, masked_ap)
        
        # Waiting for the user to press any key
        cv2.waitKey(0)
       
        return masked_ap
    
    def coordinate_center_circle(self):
        co=str(self.points[0])
        return co
    
#==============================================================================
# CREATE CLASS CIRCULARROI Mica
#==============================================================================   
    
class CircularROIMica(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = True # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        #self.points = [] # List of points defining our polygon
   
    def on_mouse(self, event, x, y, buttons, user_param):
       # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
       if self.done: # Nothing more to do
           return
   
       if event == cv2.EVENT_MOUSEMOVE:
           # We want to be able to draw the line-in-progress, so update current mouse position
           self.current = (x, y)
           #print('mouse moves')
           
       elif event == cv2.EVENT_LBUTTONDOWN:
           # Left click means adding a point at current position to the list of points
           print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
           #self.points.append((x, y))
           self.done = True
   
    def mask(self,spot_diameter_chosen,coord):
        self.coord=tuple(coord)
        x=self.coord[0]
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        
        # Update the window
        cv2.imshow(self.window_name, img_mica)
        
        if cv2.waitKey(0) == 32: # space hit
            self.done = True
     
        mask_mica=np.zeros_like(img_mica)
        
        print('spot diameter is'+str(spot_diameter_chosen))
        radius_µm=int(round((0.5*spot_diameter_chosen)))
        print('radius in µm is '+str(radius_µm)+str('µm'))

        thickness = int(-1)
        radius_pix = int(round(width*(radius_µm)/100))
        color=(255,255,255)
        radius_pix=int(radius_pix)

        coord_split=coord.split(',')
        print('coord split')
        print(coord_split)
        x=str(coord_split[0])
        x=int(x[1:])
        print('x is '+str(x))
        
        y=str(coord_split[1])
        y=int(y[:4])
        print('y is '+str(y))
        
        radius_pix=150
        color = (255,255,255)

        cv2.circle(mask_mica, (x,y), radius_pix, color, thickness)
        
        masked_mica = cv2.bitwise_and(img_mica, mask_mica)   
        cv2.imshow(self.window_name, masked_mica)
        
        cv2.waitKey(0)

        return masked_mica
    
#==============================================================================
# CREATE CLASS POLYGONDRAWERAP
#==============================================================================

class PolygonDrawerAp(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
 
        if self.done: # Nothing more to do
            return
 
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
            
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True
   
    def run(self):
        print('run')
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        canvas = img #np.zeros(CANVAS_SIZE, np.uint8)  
        # Update the window
        cv2.imshow(self.window_name, canvas)  
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)  
        
        while not self.done:
            #print('in the loop!')
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, (0,0,0), 1)
                # And  also show what the current segment would look like
                #cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)   #skipped this because extra lines are added now to the image
            
            cv2.imshow(self.window_name, canvas)
            
            # Wait 50 ms before next iteration 
            if cv2.waitKey(100) == 32:
                self.done = True
        
        print('mask is being made now')
        mask=np.full_like(img,(0,0,0))

        print(np.array([self.points]))
        # if there is a polygon drawn, draw a filled polygon
        if (len(self.points) > 0):
           cv2.fillPoly(mask, np.array([self.points]), FINAL_LINE_COLOR)  

        # extra step: see https://pythonprogramming.net/lane-region-of-interest-python-plays-gta-v/
        mask_flipped = cv2.bitwise_and(img, mask)   #this operation flips the bits
        cv2.imshow(self.window_name, mask_flipped)
        cv2.destroyWindow(self.window_name)
       
        return mask_flipped    
    
#==============================================================================
# CREATE CLASS POLYGONDRAWERAP_PREDEFINED
#==============================================================================

class PolygonDrawerApPredefined(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False                                                                                                
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        
        if self.done: # Nothing more to do
            return
 
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
            self.done = True
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            self.done = True
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True
   
    def run(self):       
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        print('line1094')
        while(not self.done):
            img #np.zeros(CANVAS_SIZE, np.uint8)
            if (len(self.points) > 0):
                pass
            # Update the window
            cv2.imshow(self.window_name, img)
            # 
            if cv2.waitKey(0) == 32: # space hit
                self.done = True
        
        print('np array t')
        print(np.array(close_window.t))
        
        mask=np.full_like(img,(0,0,0))        
        
        # if there is a polygon drawn, draw a filled polygon
        if (len(close_window.t) > 0):
            cv2.fillPoly(mask, np.array([close_window.t]), FINAL_LINE_COLOR)  

        # extra step: see https://pythonprogramming.net/lane-region-of-interest-python-plays-gta-v/
        mask_flipped = cv2.bitwise_and(img, mask)   #this operation flips the bits
        cv2.imshow(self.window_name, mask_flipped)
   
        cv2.destroyWindow(self.window_name)
       
        return mask_flipped    


#==============================================================================
# CREATE CLASS DPAR: MEASURES DPAR
#==============================================================================
class dpar(object):  
    
    # Create a function that doesn't do anything at all
    def nothing(x):
        #cv2.destroyAllWindows()
        pass
    
    # Create a function that calculates the median from a given list 
    def median(lst):
        sortedLst = sorted(lst)
        lstLen = len(lst)
        index = (lstLen - 1) // 2
           
        if len(lst)==0 or len(lst)==1:
            return 0
            
        elif (lstLen % 2):
            return sortedLst[index]
        else:
            return (sortedLst[index] + sortedLst[index + 1])/2.0

    # Initializer function 
    def __init__(self,sample,window_name): 
        # Store the info in the self object
        self.sample = sample
        self.window_name = window_name     

        
    def run(self,img_grayscale,img_original,img_adjusted):
        conv = float(close_intro_window.dist_entry)/(float(close_intro_window.pix_entry)*scale_dpar*(1-close_window_dpar.fraction))
        # Set parameters for thresholding 
        th = 160  
        window_name='dpar measurement'   
        val=7
        alpha_slider_max=10
        
        # Show windows
        cv2.imshow(window_name,img_adjusted)  #tresholded one
        
        # ==================================================================================================
        ret, thresh = cv2.threshold(img_adjusted, th, 255, cv2.THRESH_BINARY)
        thresh_inv = cv2.bitwise_not(thresh)
        
        # First while loop for color segmentation
        while(1):

            def on_trackbar_segmentation(th):
                alpha = val / alpha_slider_max
                beta = 1.0 - alpha
                ret, thresh = cv2.threshold(img_grayscale,th,255,cv2.THRESH_BINARY)
                dst = cv2.addWeighted(img_grayscale, alpha, thresh, beta, 0.0)           
                cv2.imshow(window_name, dst)  
                
                
            def on_trackbar(val):
                alpha = val / alpha_slider_max
                beta =  1.0 - alpha
                dst = cv2.addWeighted(img_grayscale, alpha, thresh, beta, 0.0)           
                cv2.imshow(window_name, dst)  
                
            cv2.createTrackbar('th',window_name,th,255, on_trackbar_segmentation) 
            cv2.createTrackbar('grayscale/binary', window_name, val, alpha_slider_max, on_trackbar)
                    
            # wait until space is pressed
            k = cv2.waitKey(1) 
            
            if k == 32:
                break
            
            # get current positions of our trackbars
            th = cv2.getTrackbarPos('th',window_name) 
            val = cv2.getTrackbarPos('grayscale/binary',window_name)
           
            # Do thresholding
            ret, thresh = cv2.threshold(img_grayscale, th, 255, cv2.THRESH_BINARY)
        
        # Invert afterwards    
        thresh_inv = cv2.bitwise_not(thresh)
        
        # Do threshold
        #ret, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        
        # FILTERS
        kernel = np.ones((3,3),np.uint8)
        # denoise it using opening filter
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
        # erosion
        erosion = cv2.erode(opening,kernel)
        # dilation
        dilation = cv2.dilate(erosion,kernel)
        
        # Invert the image (black needs to be white in opencv)
        thresh_inv = cv2.bitwise_not(dilation)
        
        if k==32:
            cv2.destroyAllWindows()
        
        # Contours
        min_size = 40
        max_size = 45

        
        # ==================================================================================================
        
        print('segmentation stopped here')
        print('part 2+3 of dpar starts now ')
        
        # new to version 4
        # Second while loop for minimum and maximum size discrimination
        while(1):
            
            # Show image
            cv2.imshow(window_name,thresh_inv)
            
            # Create scrollbar
            cv2.createTrackbar('min_size',window_name,min_size,int(round(50*scale_dpar)),dpar.nothing)
            cv2.createTrackbar('max_size',window_name,max_size,int(round(600*scale_dpar)),dpar.nothing)
            
            # Wait until an space key is pressed
            k = cv2.waitKey(1)
            
            # Get current positions of our trackbars
            min_size = cv2.getTrackbarPos('min_size',window_name)
            max_size = cv2.getTrackbarPos('max_size',window_name)
        
            # Calculate the contours from binary image
            contours,hierarchy = cv2.findContours(thresh_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            
            # Show colored image in order to add the contours there later 
            cv2.imshow('test',img_original)
            
            # Make a list with areas for each found contour 
            areas = []
            i=0
            contours_5=[]
            for i in range(0, len(contours)):
                #print('i is '+str(i))
                if contours[i].shape[0] > 5:
                    areas.append(cv2.contourArea(contours[i]))
                    contours_5.append(contours[i])
            
            # Particle size filtered
            areas_good_size=[]
            j=0
            for j in range(0, len(areas)):
                #print('j is '+str(j))
                if contours_5[j].shape[0] > 5:
                    if min_size<areas[j]<max_size:
                        areas_good_size.append(1)
                    else:
                        areas_good_size.append(0)
            
            # Make binary list with particle size
            l=len(areas_good_size)
            indexes_filt=[]
            contours_filt=[]
            
            for i in range(l):
                if areas_good_size[i]==0:
                    pass
                elif areas_good_size[i]==1:
                    # indexes of the filtered contours
                    indexes_filt.append(i)
                    # contours of the filtered contours
                    contours_filt.append(contours_5[i])
                else:
                    pass
                    
            # Draw rotated rectangles around (source: https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html)
            minRect = [None]*len(contours_filt)
            minEllipse = [None]*len(contours_filt)
            widths_list=[]
            for i, c in enumerate(contours_filt):
                # Add the ellipse to the minEllipse list
                if c.shape[0] > 5:
                    
                    # Add the contours of the bounding rectangle to the list minRect
                    minRect[i] = cv2.minAreaRect(c)
                    
                    # Add the maximal length of the bounding rectangle to the list width_lists 
                    widths_list.append(max(minRect[i][1]))
                    minEllipse[i] = cv2.fitEllipse(c)
                    
            # Draw the ellipses on the colored image (https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html)
            for i, c in enumerate(contours_filt):
                color = (0,255,0)
                # ellipse
                if c.shape[0] > 5:
                    cv2.ellipse(img_original, minEllipse[i], color, 1)
            
            # Draw rotated rectangles around (source: https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html)
            minRect = [None]*len(contours_filt)
            minEllipse = [None]*len(contours_filt)
            widths_list = []
            list_angles = []
            elongation_factor=[]
            
            for i, c in enumerate(contours_filt):
                if c.shape[0] > 5:
                    minRect[i] = cv2.minAreaRect(c)
                    minEllipse[i] = cv2.fitEllipse(c)
                    widths_list.append(round(max(minEllipse[i][1]),2))
                    list_angles.append(round(minEllipse[i][2],1))
                    max_width = max(minEllipse[i][1])
                    min_width = min(minEllipse[i][1])
                    elongation_factor.append(round(max_width/min_width,1))
            
            # Drawing (https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html)
            
            for i, c in enumerate(contours_filt):
                color = (0,255,0)
                # contour
                #cv2.drawContours(drawing, contours_filt, i, color)

                # ellipse
                if c.shape[0] > 5:
                    cv2.ellipse(img_original, minEllipse[i], color, 1)

                #cv2.imshow('results segmentation overdrawn on grain',img_original)   
                
                #cv2.destroyAllWindows()
            if k == 32:
                break
            
                
        if k==32:
            cv2.destroyAllWindows()
        
        # Contours
        cv2.imshow(window_name,thresh_inv)
        
        # Set trackbar max_size
        max_size = min_size+1
        print('minEllipse')      
        
        # ==================================================================================================

        # Fourth part for angle discrimination
        print('max size filtering has been done now')
        print('part 4')
        
        # Filter on angle (orientation feature)
        sortedLst_median = sorted(list_angles)
        stdev_angles=np.std(list_angles)

        lstLen = len(list_angles)
        index = (lstLen - 1) // 2
        #print(list_angles)
        
        if lstLen == 0:
            median_angle=''
        elif (lstLen % 2):
            median_angle=(sortedLst_median[index])
        else:
            median_angle=(sortedLst_median[index] + sortedLst_median[index + 1])/2.0
                
        list_filt_angle=[]
        print('list angles')
        print(list_angles)

        m=median_angle
        
        if lstLen != 0:
            print('m is '+str(m))
            ma=float(m+close_window_dpar.standard_deviation)
            print('ma is '+str(ma))
            mb=float(m-close_window_dpar.standard_deviation)
            print('mb is '+str(mb))
    
        contours_filt_angle=[]
        widths_list_angle_filtered=[]
        
        font2 = cv2.FONT_HERSHEY_PLAIN 
        fontScale=int(1)
        thickness=int(1)
        color_text = (255,0,0)
        color_width=(0,0,0)
        color_elongation=(0,0,255)
        
        
        for i, c in enumerate(contours_filt):
            an = list_angles[i]
            e = elongation_factor[i]
            
            # If the angle is right
            if an<ma and an>mb and float(e)>float(close_window_dpar.elongation):
                # Add angle
                print('angle and elongation factor is ok: '+str(an))
                list_filt_angle.append(an)
                
                # Add contour
                contours_filt_angle.append(c)
                
                # Add width
                w=max(minEllipse[i][1])
                widths_list_angle_filtered.append(w)
                
                # Add it on the pictures
                coord=minEllipse[i][0]               
                x=round(coord[0])
                y=round(coord[1])
                
                # Add text 
                cv2.putText(img_original,str(round(e)),(x-5,y),font2,fontScale,color_elongation,thickness)
                cv2.putText(img_original,str(round(an)),(x,y+8),font2,fontScale,color_text,thickness)
                cv2.putText(img_original,str(round((w*conv),1)),(x,y-10),font2,fontScale,color_width,thickness)
            
            else:
                
                # Add angle
                print('angle is '+str(an) + ' and elongation is ' + str(round(e,1)))
                pass
            
        print('list angles')
        #print(list_angles)
        #print(contours_filt_angle)
        print('list filt angle')
        #print(list_filt_angle)

        cv2.imshow('angles',img_original)            
          
        # Calculate median for unfiltered and filtered list
        median_unf = dpar.median(widths_list)  
        median_filt = dpar.median(widths_list_angle_filtered)  
                  
        # end with storing the image somewhere
        print(str(median_unf*conv)+str(' µm'))
        print(str(median_filt*conv)+str(' µm'))
        
        cv2.imwrite(str(sample)+"segmented.png",thresh_inv)
        # end of the while loop
        
        name=str(close_window_dpar.name)
        print('name is ...')
        print(name)
        
        cv2.waitKey(0)
        cv2.imwrite(str(name)+str('dpar')+'.png',img_original)
        cv2.destroyAllWindows()
        
        d={'Sample':[name], 'Median_filtered dpar':[median_filt*conv]}
                
        # Build a panda dataframe to add the counting data
        d_pd=pdd.DataFrame()
    
        # Append data to the 
        d_pd=d_pd.append((pdd.DataFrame(d)),ignore_index=True)  
        
        # Export to csv file  
        d_pd.to_csv(str(name)+"dpar.csv", sep=';', encoding='utf-8',index=False)


#==============================================================================
# SET INACTIVATORS FOR BUTTONS 
#==============================================================================

# Function to apply when ED glass is selected 
def inactivator_buttons_glass():
    # Other options must be deselected
    apatitebutton.deselect()
    laftbutton.deselect()
    
    # DIS- and ENABLE other entries
    button_mica_names.config(state=NORMAL)
    button_mica_names_epi.config(state=NORMAL)
    button_apatite_names.config(state=DISABLED)
    button_apatite_names_epi.config(state=DISABLED)
    
    # all specifications for the graticule needs to be greyed out
    spot_diameter_label.config(state=DISABLED)
    circular_graticule_button.config(state=DISABLED)
    polygon_graticule_button.config(state=DISABLED)
    label_polygon_list.config(state=DISABLED)
    label_polygon_list_2.config(state=DISABLED)
    no_graticule_button.config(state=DISABLED)
    
    # because only one option is possible, the rest should be deselected and only one should be selected
    no_graticule_button.select()
    polygon_graticule_button.deselect()
    circular_graticule_button.deselect()
    
# Function to apply when LAFT is selected    
def inactivator_buttons_laft():
    # Other options must be deselected
    glassbutton.deselect()
    apatitebutton.deselect()
    
    # Gray out some buttons below 
    button_mica_names.config(state=DISABLED)
    button_mica_names_epi.config(state=DISABLED)
    button_apatite_names.config(state=NORMAL)
    button_apatite_names_epi.config(state=NORMAL)
    
    # Some buttons must be "normal" again 
    spot_diameter_label.config(state=NORMAL)
    circular_graticule_button.config(state=NORMAL)
    polygon_graticule_button.config(state=NORMAL)
    label_polygon_list.config(state=NORMAL)
    label_polygon_list_2.config(state=NORMAL)
    no_graticule_button.config(state=NORMAL)
    
# Function to apply when AP/ED is selected   
def inactivator_buttons_ap_ed():
    # Other options must be deselected
    glassbutton.deselect()
    laftbutton.deselect()
    
    # Normal the buttons below again 
    button_mica_names.config(state=NORMAL)
    button_mica_names_epi.config(state=NORMAL)
    button_apatite_names.config(state=NORMAL)
    button_apatite_names_epi.config(state=NORMAL)

    # all specifications for the graticule needs to be NORMAL
    spot_diameter_label.config(state=NORMAL)
    circular_graticule_button.config(state=NORMAL)
    polygon_graticule_button.config(state=NORMAL)
    label_polygon_list.config(state=NORMAL)
    label_polygon_list_2.config(state=NORMAL)
    no_graticule_button.deselect()
    no_graticule_button.config(state=NORMAL)

# Function to apply when no graticule is selected                
def inactivator_buttons_no_graticule():
    # Polygon
    if no_graticule.get()==1:
        print('no graticule')
        spot_diameter_entry.config(state=DISABLED)
        circular_graticule_button.deselect()
        polygon_graticule_button.deselect()
        polygon_list_entry.config(state=DISABLED)
        
    elif polygon_graticule.get()==0 and circular_graticule.get()==0 or polygon_graticule.get()==1 and circular_graticule.get()==1:
        print('anders')
        spot_diameter_entry.config(state=DISABLED)
        polygon_list_entry.config(state=DISABLED)
    else:
        print('else')
        
# Inactivator function for when circular graticule is selected         
def inactivator_buttons_spot():
    # Polygon
    if polygon_graticule.get()==1:
        print('polygon')
        spot_diameter_entry.config(state=DISABLED)
        circular_graticule_button.deselect()
        no_graticule_button.deselect()
        polygon_list_entry.config(state=NORMAL)
    elif polygon_graticule.get()==0 and circular_graticule.get()==0 or polygon_graticule.get()==1 and circular_graticule.get()==1:
        print('other')
        spot_diameter_entry.config(state=DISABLED)
        polygon_list_entry.config(state=DISABLED)
    else:
        print('else')
        
# Inactivator function for when a custom polygon is selected 
def inactivator_buttons_polygon():
    # Spot
    if circular_graticule.get()==1:
        print('spot')
        spot_diameter_entry.config(state=NORMAL)
        polygon_graticule_button.deselect()
        polygon_list_entry.config(state=DISABLED)
        no_graticule_button.deselect()
    elif polygon_graticule.get()==0 and circular_graticule.get()==0 or polygon_graticule.get()==1 and circular_graticule.get()==1:
        print('other')
        spot_diameter_entry.config(state=DISABLED)
        polygon_list_entry.config(state=DISABLED)
    else:
        print('else')
        
    
#==============================================================================
# MAKE A FUNCTION WHICH RETURNS A GAMMA-CORRECTED IMAGE (corrects for irregular illumination)
#==============================================================================

# source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/#pyi-pyimagesearch-plus-optin-modal
def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

#==============================================================================
# CHECK IF THE KEY IS RIGHT
#==============================================================================

# I am not using a secured version anymore
# .... so the code is quite useless
if security_version =="secured":
    email=str(close_intro_window2.emailaddress)
    # lower case
    email=email.lower()
    # specify salt for encrypting
    salt="XxoOo.oOxXIIY"
    # add the salt to the emailaddress
    x_salted=email+salt
    import hashlib
    key = hashlib.md5(x_salted.encode())
    key_digest=key.hexdigest()
    
    if key_digest == close_intro_window2.key:
        # Right key, program will proceed
        right_key='yes'
    else:
        # The program will give an error later
        right_key='no'
        
elif security_version =="unsecured":
    right_key='yes'
    pass

#==============================================================================
# GIVE AN ERROR MESSAGE
#==============================================================================
# I am not using a secured version anymore
# .... so the code is quite useless

if right_key=='no':
    root_fail_key = Tk()
    #root_fail_key.iconbitmap('microscope.ico')
    root_fail_key.title('Error')

    h_1 = Label(root_fail_key, text="You filled in a wrong software key").pack()
    h_2 = Label(root_fail_key, text="Ask Simon.Nachtergaele@UGent.be for a personal software key").pack()
        
    def close_wrong_key():
        root_fail_key.destroy()
    
    quit_button_wrong_key = Button(root_fail_key,text = "Quit", command = close_wrong_key, cursor="hand2")
    quit_button_wrong_key.pack()
    shutdown_script = 1
    root_fail_key.mainloop() 
    
#==============================================================================
# START THE LOOP FOR DPAR MEASUREMENT
#==============================================================================

if dpar_review.get()==1:
    print('dpar needs to be measured')
        
    # Make tkinter window
    root_dpar = Tk()
    #root_dpar.iconbitmap('microscope.ico')
    root_dpar.title('AI-track-tive: dpar module '+str(version_programme))
    root_dpar.geometry('+'+str(place_width)+'+'+str(place_height))
    #root_dpar.geometry("+700+400")
    
    # Make function that gives an error message on closing the window
    def on_closing():
        if messagebox.askokcancel("Quit","Do you want to quit?"):
            root_dpar.destroy()
            os._exit(00)   # restart kernel 
            
    root_dpar.protocol("WM_DELETE_WINDOW",on_closing)
    
    list_samples=list()
    
    # Labels
    font_bold=font.Font(size=9,weight="bold")    
    
    name = Label(root_dpar, text="Enter the name for your sample:")
    name['font']= font_bold
    name.grid(row=0, column=0)
    name_entry = Entry(root_dpar, width = 25) 
    name_entry.grid(row=1, column=0)
    name = name_entry.get()
    
    space = Label(root_dpar, text=" ")
    space.grid(row=2, column=0)
      
    loc_label = Label(root_dpar, text="Select the microscopy images:")
    loc_label['font']=font_bold
    loc_label.grid(row=3, column=0)
        
    is_there_an_ap_chosen='0'
    counter=0
    
    # Create function that gets the location of the apatite photo files
    def choose_samplenames_apatite_epi():
        name_apatites_epi = askopenfilenames() 
        choose_samplenames_apatite_epi.name_apatites=list(name_apatites_epi)
        
        global is_there_an_ap_chosen
        is_there_an_ap_chosen='1'
        
        global counter
        counter=len(choose_samplenames_apatite_epi.name_apatites)

        print('is there an ap chosen occurred')
            
    button_apatite_names_epi = Button(text="Select 1 apatite in reflected light", width = 40, command = choose_samplenames_apatite_epi, cursor="hand2")
    button_apatite_names_epi.grid(row=5, column=0)
    
    space = Label(root_dpar, text=" ")
    space.grid(row=6, column=0)
    
    more_info = Label(root_dpar, text="Specify the following:")
    more_info['font']=font_bold
    more_info.grid(row=7, column=0)
    
    # Number of grains 
    grains_label = Label(root_dpar, text="# grains:")
    grains_label.grid(row=8, column=0)
    default_n_ap = StringVar(root_dpar,value=int(1))
    grains_label_entry = Entry(root_dpar, width = 3, textvariable = default_n_ap) 
    grains_label_str=str(grains_label_entry.get())
    grains_label_entry.grid(row=9, column=0)
    
    # Fraction of the image used 
    fraction_label = Label(root_dpar, text="I do not want to use ...% of the image \n (increase to zoom in, decrease to zoom out)")
    fraction_label.grid(row=10, column=0)
    fraction = StringVar(root_dpar,value=int(50))
    fraction_label_entry = Entry(root_dpar, width = 5, textvariable = fraction) 
    fraction_label_str=str(fraction_label_entry.get())
    fraction_label_entry.grid(row=11, column=0)
        
    # Number for gamma correction
    gamma_label = Label(root_dpar, text="gamma value for unequivally exposure: (between 1 and 3)")
    gamma_label.grid(row=12, column=0)
    default_gamma = StringVar(root_dpar,value=int(2))
    gamma_label_entry = Entry(root_dpar, width = 3, textvariable = default_gamma) 
    gamma_label_float=float(grains_label_entry.get())
    gamma_label_entry.grid(row=13, column=0)
    
    # Number for orientation correction
    std_label = Label(root_dpar, text="maximal angle between median and measured etch pit (°) ")
    std_label.grid(row=15, column=0)
    default_std = StringVar(root_dpar,value=int(30))
    std_label_entry = Entry(root_dpar, width = 3, textvariable = default_std) 
    std_label_float=float(grains_label_entry.get())
    std_label_entry.grid(row=16, column=0)
    
    # Number for elongation correction
    elong_label = Label(root_dpar, text="minimal elongation")
    elong_label.grid(row=17, column=0)
    elong_std = StringVar(root_dpar,value=int(2))
    elong_label_entry = Entry(root_dpar, width = 3, textvariable = elong_std) 
    elong_label_float=float(elong_label_entry.get())
    elong_label_entry.grid(row=18, column=0)
       
    print('name entry: ')
    print(name_entry.get())

    
    # Function for quit button
    def close_window_dpar():        
        # Retrieve the number of grains needed to analyze
        if name_entry.get()!='' and grains_label_entry.get()!='' and is_there_an_ap_chosen=='1' and counter==1 and elong_label_entry.get()!='' and fraction_label_entry.get()!='':
            close_window_dpar.name_apatites_epi = choose_samplenames_apatite_epi.name_apatites
            close_window_dpar.name = name_entry.get()
            close_window_dpar.grains_label_entry=int(grains_label_entry.get())
            close_window_dpar.gamma = float(gamma_label_entry.get())
            close_window_dpar.standard_deviation = float(std_label_entry.get())
            close_window_dpar.elongation=float(elong_label_entry.get())
            close_window_dpar.fraction = float(fraction_label_entry.get())/200
            
            # Close the window
            root_dpar.destroy()
        else:            
            error_message_dpar = Label(root_dpar, text="please give all necessary information")
            error_message_dpar.grid(row=22,column=0)
    
    # Some more labels to end this tkinter window
    space = Label(root_dpar, text=" ")
    space.grid(row=20, column=0)
    
    quit_button = Button(root_dpar,text = "Continue", command = close_window_dpar, cursor="hand2")
    quit_button.grid(row=21, column=0)
       
    
    root_dpar.mainloop()
    print('gamma')
    print(close_window_dpar.gamma)
    print('standard deviation')
    print(close_window_dpar.standard_deviation)
    
    # Build a panda dataframe to add the dpar data
    dpar_pd=pdd.DataFrame()
    
    # Change the working directory
    os.chdir(close_intro_window.output_directory)
        
    grains_number = close_window_dpar.grains_label_entry
    apatite_paths_epi=list(close_window_dpar.name_apatites_epi)
    
    # List of lists apatite 
    if dpar_review.get()==1:
        list_of_lists_ap_epi=[]
        l=list()
        i=1    
        print(i)
        while (i-1) < grains_number:
            j=1
            add=apatite_paths_epi[:j]
            list_of_lists_ap_epi.append(list(add))
            apatite_paths_epi=apatite_paths_epi[j:]
            i+=1
        print('list of lists apatite epi' +str(list_of_lists_ap_epi))
    
    for i in range(len(list_of_lists_ap_epi)):
        
        sample = list_of_lists_ap_epi[i][0]
        print('sample is '+str(sample))
        img = cv2.imread(sample)
        
        # Set fraction of image
        fraction=close_window_dpar.fraction
        f_a=fraction
        f_b=1-fraction
        
        # Take the center of each image
        a=int(round((float(f_a)*close_intro_window.pix_entry)))
        print('a is '+str(a))
        b=int(round((float(f_b)*close_intro_window.pix_entry)))
        print('b is '+str(b))
        img = img[a:b,a:b]
        
        # Make grayscale image
        img_ap_gray = cv2.imread(sample, 0)   
        # Cut a window of 100 µm on 100 µm Resize image 
        img_ap_gray=img_ap_gray[a:b,a:b]
        img_ap_gray=adjust_gamma(img_ap_gray,gamma=float(close_window_dpar.gamma))
        
        # Set scale because we'll need it later in the resizing paragraph
        scale_dpar=1
            
        # Resize depending on screen width and height
        height, width, channels = img.shape
        print('width before resizing' + str(width))
        
        # # Check if the images have all the right size
        # if px == width and px == height:
        #     print('right size')
        # else:
        #     error = 'the size of the images is not right'
        #     logger.error('ERROR: the size of the images is not right')
        #     print('ERROR: the size of the image is not right')
        #     frameinfo = getframeinfo(currentframe())
        #     print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
        #     fail='yes'
        #     break
        

        # Do a while loop in order to resize and yield an image that fills the screen for +-80% 
        while width < 0.80*screen_height:
            img = cv2.resize(img, None, fx=1.05, fy=1.05)
            img_ap_gray = cv2.resize(img_ap_gray, None, fx=1.05, fy=1.05)
            height, width, channels = img.shape
            scale_dpar=scale_dpar*1.05
            print(scale_dpar)
            print(height)
            
        # Do a while loop in order to resize and yield an image that fills the screen for +-80% 
        while width > 0.80*screen_height:
            img = cv2.resize(img, None, fx=0.95, fy=0.95)
            img_ap_gray = cv2.resize(img_ap_gray, None, fx=0.95, fy=0.95)
            height, width, channels = img.shape
            scale_dpar=scale_dpar*0.95
            print(scale_dpar)
            print(height)
        
        print('resizing is done')
        
        # Gamma filter
        img_adjusted=adjust_gamma(img,gamma=float(close_window_dpar.gamma))
        
        # Save the grayscale image
        sample_black_and_white=sample[:-4]+str("blackandwhite")+str(".jpg")
        cv2.imwrite(sample_black_and_white,img_ap_gray)
        
        # Start Dpar measurement
        ap_picture = dpar(sample,"image1")
        dpar_ap_picture = dpar.run(img_ap_gray,img_ap_gray,img, img_adjusted)
        
#==============================================================================
# PRODUCE TEXT FILES FROM IMAGES WITH TRACK ANNOTATED
#==============================================================================


# First step is to create another window that asks for the image that you want to annotate
if annotate.get()==1:

    
    # Make a new tkinter root
    root_an = Tk()
    root_an.title('AI-Track-tive track annotations')

    # Quit button
    def close_window_root_an():
        root_an.destroy()
            
    # Warning message when closing 
    def on_closing():
        if messagebox.askokcancel("Quit","Do you want to quit?"):
            root_an.destroy()
            os._exit(00)   # restart kernel 
    root_an.protocol("WM_DELETE_WINDOW",on_closing)
    
    # Define size of the window
    root_an.geometry('+'+str(place_width)+'+'+str(place_height))
            
    # Location label
    loc_an = Label(root_an, text="Select the microscopy images:")
    loc_an.grid(row=1, column=0)
    
    # Specify the width of the entries 
    width_int=int(45)
    
    # Get location apatite photo files
    def choose_samplenames_annotate():
        name_an = askopenfilenames() 
        
        # Select only one image at the time
        # One image
        if len(name_an)==1:
            choose_samplenames_annotate.name_an=list(name_an)
            button_names = Button (text="Select 1 image", width = width_int, command = choose_samplenames_annotate, cursor="hand2", bg='pale green')
            button_names.grid(row=5, column=0)
        
        # Zero images
        elif len(name_an)==0:
            assert "error"
   
                
    button_names = Button(root_an,text="Select 1 image", width = width_int, command = choose_samplenames_annotate, cursor="hand2", bg='tomato')
    button_names.grid(row=5, column=0)

    quit_button = Button(root_an,text = "Start annotating tracks", command = close_window_root_an, cursor="hand2")
    quit_button.grid(row=28, column=0)

    # End the root 
    root_an.mainloop()   
    
    # Make instructions window
    if instructions_window.get()==1:
        instructions_an = np.zeros((200,600,1), np.uint8)
        instructions_an.fill(255)
        cv2.namedWindow('AI-Track-tive instructions for annotating labels')
        cv2.putText(instructions_an,' - You now need to manually indicate all tracks', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
        cv2.putText(instructions_an,' - Do this by clicking in the left upper corner of a track', (10,55) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.putText(instructions_an,' - Move your mouse to the other corner', (10,80) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.putText(instructions_an,' - And then release the left mouse button and press space', (10,105) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
        cv2.putText(instructions_an,' - Do this for all the tracks. Finish with middle mouse button (or CTRL+mouse move)', (10,130) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
    
        cv2.imshow('AI-Track-tive instructions for annotating labels',instructions_an)
    
    # Loading image
    img = cv2.imread(choose_samplenames_annotate.name_an[0])
    
    # Resize depending on screen width and height
    height, width, channels = img.shape
    print('width after slicing but before resizing' + str(width))       
    
    # If image is too small, make it larger
    while width < 0.8 * screen_height:
        img = cv2.resize(img, None, fx=1.05, fy=1.05)
        height, width, channels = img.shape
        print('resized to'+str(height))
        
    # If image is too large, make it smaller
    while width > 0.8 * screen_height:
        img = cv2.resize(img, None, fx=0.95, fy=0.95)
        height, width, channels = img.shape
        print('resized to'+str(height))
        
    # Get the image shape properties again after resizing 
    height, width, channels = img.shape
    
    # Show the image
    #cv2.imshow(title_an, img)
    
    # Call the class
    pdt = MissingTracksAnnotate("Manual Review process")
    count_ap_loop=0
    
    rect_txt_list=list()
    
    # Create a loop (while the middle mouse button or CTRL is not clicked)
    while pdt.stop == False:
        print(' ')
        print('pdt.latest_track_found is '+str(pdt.latest_track_found()))
        
                           
        if pdt.latest_track_found()==1:
            print('line 2266')
            croppedimage = img
            
            # Now a new loop will start for the z- and RL/TL functionality 
            croppedimage_manually_added_tracks = pdt.findtracksmanually()   
            
            # If it's the first track that's added
            if count_ap_loop==0:
                mft_ap=pdt.list_manually_found_tracks()
            # If it's not
            else:
                mft_ap = pdt.list_manually_found_tracks()
            
            print('Draw rectangles now')

            if mft_ap==[]:
                print('line 2283')
                pass
            else:               
                print('mft_ap is '+str(mft_ap))
                
                try:
                    print('count ap loop is')
                    print(count_ap_loop)
                    l=mft_ap[count_ap_loop]  # new coordinate to be added
                except IndexError:
                    error = 'You added the tracks too fast. Please follow the exact instructions.'
                    logger.error('ERROR: you added the tracks too fast. Slow down please.')
                    #print('IndexError')
                    frameinfo = getframeinfo(currentframe())
                    #print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                    
                print('l is '+str(l))
                
                if len(l)==2:
                    print('length is alright')
                    corn_one=l[0]
                    corn_two=l[1]

                    if int(corn_one[0])<int(corn_two[0]):
                        x = int(abs(corn_one[0]))
                        
                        if int(corn_one[1])<int(corn_two[1]):
                            y = int(abs(corn_one[1]))
                        else:
                            y = int(abs(corn_two[1]))
                            
                    else: 
                        x = int(abs(corn_two[0]))
                        if int(corn_one[1])<int(corn_two[1]):
                            y = int(abs(corn_one[1]))
                        else:
                            y = int(abs(corn_two[1]))
                    
                    # Find w
                    w = int(abs(corn_one[0]-corn_two[0]))
                    
                    # Find h
                    h = int(abs(corn_one[1]-corn_two[1]))              
                    
                    # Add text to the image
                    cv2.rectangle(croppedimage, (x, y), (x + w, y + h), (0, 200, 0), 1)
                    
                    # Txt file needs fractions
                    rect_txt = list()
                    x_txt = str((0.5*w + x)/(width))
                    rect_txt.append(x_txt[:8])
                    y_txt = str((0.5*w + y)/width)
                    rect_txt.append(y_txt[:8])
                    w_txt = str(w/width)
                    rect_txt.append(w_txt[:8])
                    h_txt = str(h/width)
                    rect_txt.append((h_txt[:8]))
                    
                    # Append the self-identified rectangle to the list 
                    rect_txt_list.append(rect_txt)
                    
                    # produce the .txt file 
                    print(labelImgformatter(rect_txt_list,choose_samplenames_annotate.name_an))               
                    
                    # Modify count function
                    count_ap_loop= count_ap_loop+1
                
                else: 
                    print('space was hit')
                
            
            cv2.imshow("Manual Review process", croppedimage)
            cv2.waitKey(0)
            base_name=os.path.basename(choose_samplenames_annotate.name_an[0])
            cv2.imwrite(str(base_name[:-4])+str("_annotated.png"),croppedimage)                               
            cv2.imshow("Manual Review process", croppedimage)

    
    # Produce a .txt file from which an example code has been made in GL part of this code 
    cv2.destroyAllWindows()
    
        
#==============================================================================
# START THE LOOP FOR THE FISSION TRACK COUNTING
#==============================================================================

while var_manually_review.get()==1 and shutdown_script == 0 and annotate.get()==0:
    root = Tk()
    #root.iconbitmap('microscope.ico')
    root.title('AI-track-tive '+str(version_programme))
    
    def on_closing():
        if messagebox.askokcancel("Quit","Do you want to quit?"):
            root.destroy()
            os._exit(00)   # restart kernel 
            
    root.protocol("WM_DELETE_WINDOW",on_closing)

    # Set window in the middle of the screen (more or less)
    root.geometry('+'+str(place_width)+'+'+str(place_height))
    
    list_samples=list()
    # Set font
    font_bold=font.Font(size=9,weight="bold")    
    
    # Add some labels and entries 
    name = Label(root, text="Enter the name for your sample/grain:")
    name['font']= font_bold
    name.grid(row=0, column=0)
    name_entry = Entry(root, width = 25) 
    name_entry.grid(row=1, column=0)
    name = name_entry.get()
      
    Header_second = Label(root, text="Select the type of sample:")
    Header_second['font']= font_bold
    Header_second.grid(row=3, column=0)
    
    # Apatite + external detector 
    var2 = IntVar()
    apatitebutton = Checkbutton(root,text="apatite + external detector (EDM) ",variable=var2, cursor="hand2", command = inactivator_buttons_ap_ed)
    apatitebutton.grid(row=4, column=0)
    
    # Only external detector
    var1 = IntVar()
    glassbutton = Checkbutton(root, text="only external detector (EDM)", variable=var1, cursor="hand2", command=inactivator_buttons_glass)
    glassbutton.grid(row=5, column=0)
    
    # LAFT
    var3 = IntVar()
    laftbutton = Checkbutton(root, text="only apatite (LA-ICP-MS based FT dating)", variable=var3, cursor="hand2", command = inactivator_buttons_laft)
    laftbutton.grid(row=6, column=0)
        
    # ROI label
    graticule_shape = Label(root, text="Specify your region of interest:")
    graticule_shape['font']=font_bold
    graticule_shape.grid(row=8, column=0)
    
    # No region of interest and just slice it to a square of 100µm on 100µm 
    no_graticule = IntVar()
    no_graticule_button = Checkbutton(root, text="square of 100µm on 100µm", variable=no_graticule, cursor="hand2", command = inactivator_buttons_no_graticule)
    no_graticule_button.grid(row=9, column=0)
    
    # Region of Interest is a custom drawn polygon 
    polygon_graticule = IntVar()
    polygon_graticule_button = Checkbutton(root, text="custom polygon", variable=polygon_graticule, cursor="hand2", command = inactivator_buttons_spot)
    polygon_graticule_button.grid(row=10, column=0)
    
    # Label to enter the custom polygon using a list
    label_polygon_list = Label(root, text="Enter the coordinates for your custom polygon here:")
    label_polygon_list.grid(row=11, column=0)
    label_polygon_list_2 = Label(root, text="e.g. (0,0,680,0,680,680,0,680)")
    label_polygon_list_2.grid(row=12, column=0)
    polygon_list_entry = Entry(root, width=30)
    polygon_list_entry.grid(row=13, column=0)    
    
    # Label for circular 
    circular_graticule = IntVar()
    circular_graticule_button = Checkbutton(root, text="circular", variable=circular_graticule, cursor="hand2",command = inactivator_buttons_polygon)
    circular_graticule_button.grid(row=14, column=0)

    # Specify size of the circle 
    spot_diameter_label = Label(root,text='Circle diameter in µm: (max 100µm)')
    spot_diameter_label.grid(row=15, column=0)
    spot_diameter_entry = Entry(root, width=5)
    spot_diameter_entry.grid(row=16, column=0)
    
    # Location label
    loc_label = Label(root, text="Select the microscopy images:")
    loc_label['font']=font_bold
    loc_label.grid(row=20, column=0)
    
    # Specify the width of the entries 
    width_int=int(45)
    
    # Get location apatite photo files
    def choose_samplenames_apatite():

        # Glass
        if var2.get()==0 and var1.get()==1 and var3.get()==0:
            pass
        
        # Apatite only (LAFT)
        elif var1.get()==0 and var2.get()==0 and var3.get()==1:
            name_apatites = askopenfilenames() 
            
            # If two apatite images are given 
            if len(name_apatites)==2:
                choose_samplenames_apatite.name_apatites=list(name_apatites)
                button_apatite_names = Button (text="Select 1 apatite transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_apatite, cursor="hand2", bg='pale green')
                button_apatite_names.grid(row=21, column=0)
                
            # If one apatite image is given
            elif len(name_apatites)==1:
                name_apatites_list=list(name_apatites)
                name_apatites_list.extend(name_apatites_list)
                name_apatites = name_apatites_list
                choose_samplenames_apatite.name_apatites=list(name_apatites)
                button_apatite_names = Button (text="Select 1 apatite transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_apatite, cursor="hand2", bg='pale green')
                button_apatite_names.grid(row=21, column=0)
                print(' ')
                print(name_apatites)
                print(' ')
            
            elif len(name_apatites)==0:
                assert "error"
                
        # Apatite + ED location gathereres:
        else:
            name_apatites = askopenfilenames() 
            
            # If two apatite images are given
            if len(name_apatites)==2:
                choose_samplenames_apatite.name_apatites=list(name_apatites)
                button_apatite_names = Button (text="Select 1 apatite transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_apatite, cursor="hand2", bg='pale green')
                button_apatite_names.grid(row=21, column=0)
                
            # If one apatite image is given 
            elif len(name_apatites)==1:
                name_apatites_list=list(name_apatites)
                name_apatites_list.extend(name_apatites_list)
                name_apatites = name_apatites_list
                
                choose_samplenames_apatite.name_apatites=list(name_apatites)
                button_apatite_names = Button (text="Select 1 apatite transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_apatite, cursor="hand2", bg='pale green')
                button_apatite_names.grid(row=21, column=0)
            
            elif len(name_apatites)==0:
                assert "error"
                
                
    button_apatite_names = Button(text="Select 1 apatite transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_apatite, cursor="hand2", bg='tomato')
    button_apatite_names.grid(row=21, column=0)
    
    # Get location apatite photo files
    def choose_samplenames_apatite_epi():
        
        # Glass
        if var2.get()==0 and var1.get()==1 and var3.get()==0:
            pass
        
        # Apatite only (LAFT)
        elif var1.get()==0 and var2.get()==0 and var3.get()==1:
            name_apatites_epi = askopenfilenames() 
            choose_samplenames_apatite_epi.name_apatites=list(name_apatites_epi)
            
            #Change to green when there is one filled in 
            if len(name_apatites_epi)==1:
                button_apatite_names = Button (text="Select 1 apatite reflected light image", width = width_int, command = choose_samplenames_apatite_epi, cursor="hand2", bg='pale green')
                button_apatite_names.grid(row=22, column=0)
                
        # Apatite + ED:
        else:
            name_apatites_epi = askopenfilenames() 
            choose_samplenames_apatite_epi.name_apatites=list(name_apatites_epi)
            
            if len(name_apatites_epi)==1:
                button_apatite_names = Button (text="Select 1 apatite reflected light image", width = width_int, command = choose_samplenames_apatite_epi, cursor="hand2", bg='pale green')
                button_apatite_names.grid(row=22, column=0)
            
    button_apatite_names_epi = Button(text="Select 1 apatite reflected light image", width = width_int, command = choose_samplenames_apatite_epi, cursor="hand2", bg='tomato')
    button_apatite_names_epi.grid(row=22, column=0)
    
    # Get location mica photo files
    def choose_samplenames_mica():
        
        # Glass
        if var2.get()==0 and var1.get()==1 and var3.get()==0: 
            name_micas = askopenfilenames()
            
            # If two mica pictures is given
            if len(name_micas)==2:
                button_mica_names = Button(text="Select 1 mica transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_mica, cursor="hand2", bg='pale green')
                button_mica_names.grid(row=24, column=0)
                choose_samplenames_mica.name_micas=list(name_micas)
                
            # If only one mica picture is given
            elif len(name_micas)==1:
                button_mica_names = Button(text="Select 1 mica transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_mica, cursor="hand2", bg='pale green')
                button_mica_names.grid(row=24, column=0)
                
                name_micas_list=list(name_micas)
                name_micas_list.extend(name_micas_list)
                name_micas=name_micas_list
                
                choose_samplenames_mica.name_micas=list(name_micas)
            
        # Apatite only (LAFT)
        elif var1.get()==0 and var2.get()==0 and var3.get()==1: 
            pass
            
        # Apatite + ED     
        elif var2.get()==1 and var1.get()==0 and var3.get()==0:  
            name_micas = askopenfilenames()

            # If two mica pictures is given
            if len(name_micas)==2:
                button_mica_names = Button(text="Select 1 mica transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_mica, cursor="hand2", bg='pale green')
                button_mica_names.grid(row=24, column=0)
                choose_samplenames_mica.name_micas=list(name_micas)
            
            # If one mica picture is given
            elif len(name_micas)==1:
                button_mica_names = Button(text="Select 1 mica transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_mica, cursor="hand2", bg='pale green')
                button_mica_names.grid(row=24, column=0)
                
                name_micas_list=list(name_micas)
                name_micas_list.extend(name_micas_list)
                name_micas=name_micas_list
                
                choose_samplenames_mica.name_micas=list(name_micas)
        
        
    button_mica_names = Button(text="Select 1 mica transmitted light image (or 2 for z-stack)", width = width_int, command = choose_samplenames_mica, cursor="hand2", bg='tomato')
    button_mica_names.grid(row=24, column=0)
    
    # Get location mica photo files for reflected (epi) light
    def choose_samplenames_mica_epi():

        # Glass
        if var2.get()==0 and var1.get()==1 and var3.get()==0: 
            name_micas_epi = askopenfilenames()
            choose_samplenames_mica_epi.name_micas=list(name_micas_epi)
            
            if len(name_micas_epi)==1:
                button_mica_names_epi = Button(text="Select 1 mica reflected light image", width = width_int, command = choose_samplenames_mica_epi, cursor="hand2", bg='pale green')
                button_mica_names_epi.grid(row=25, column=0)
            
        # Apatite only (LAFT)
        elif var1.get()==0 and var2.get()==0 and var3.get()==1: 
            button_mica_names_epi = Button(text="Select 1 mica reflected light image", width = width_int, command = choose_samplenames_mica_epi, cursor="hand2")
            pass
            
        # Apatite + ED     
        elif var2.get()==1 and var1.get()==0 and var3.get()==0:  
            name_micas_epi = askopenfilenames()
            choose_samplenames_mica_epi.name_micas=list(name_micas_epi)
            
            if len(name_micas_epi)==1:
                button_mica_names_epi = Button(text="Select 1 mica reflected light image", width = width_int, command = choose_samplenames_mica_epi, cursor="hand2", bg='pale green')
                button_mica_names_epi.grid(row=25, column=0)
            
        
    button_mica_names_epi = Button(text="Select 1 mica reflected light image", width = width_int, command = choose_samplenames_mica_epi, cursor="hand2", bg='tomato')
    button_mica_names_epi.grid(row=25, column=0)
    
    # define values because they might be necessary     
    ap_z_entry_str=str(2)   
    mica_z_entry_str=str(2)   
    grains_label_str = str(1)
 
    # Quit button
    def close_window():
        text = name_entry.get()
        
        # Check for special characters 
        if set(text).difference(ascii_letters + digits):       
            error_message = Label(root, text='Please give a name without special characters')
            error_message.grid(row=30, column= 0)
            
        # If the name entry is filled in
        elif name_entry.get()!='': 
            
            # Retrieve name
            close_window.name = name_entry.get()
                        
            # If a circular graticule is chosen, get the diameter
            if circular_graticule.get() == 1:
                close_window.spot_diameter=int(spot_diameter_entry.get())
        
            if len(polygon_list_entry.get())!=0:
                # Make list of tuples from entry
                p = polygon_list_entry.get()
                len_p=len(p)
                print(len_p)
                if p.endswith(','):
                    p=p[:-1]
                else:
                    pass
                p=list(map(int, p.strip().split(',')))
                print('p split '+str(p))
                l=int(len(p))
                print('l is '+str(l))
                t=[]
                # if the list of coordinates is even
                if l%2==0:
                    print('even')
                    for i in range (int(l/2)):
                        print(i)
                        c = p[:2]
                        print('c is '+ str(c))
                        p=p[2:]
                        t.append(tuple(c))
                    print('t is below')
                    print(t)
                    close_window.t = t
                #if the list of coordinates is odd, no polygon will be drawn 
                else:
                    print('odd')
                    close_window.t = [(0,0),(804,0),(804,804),(0,804)]
                    shutdown_script == 0
            else:
                close_window.t=[] 
            
            # Close the window
            root.destroy()
        
        # If it's not filled in correctly    
        else:
            error_message = Label(root, text='Please give all necessary information')
            error_message.grid(row=30, column=0)
     
    # Add some more labels
    space = Label(root, text=" ")
    space.grid(row=26, column=0)
    
    space = Label(root, text=" ")
    space.grid(row=27, column=0)
    
    quit_button = Button(root,text = "Continue", command = close_window, cursor="hand2")
    quit_button.grid(row=28, column=0)
    
    space = Label(root, text=" ")
    space.grid(row=29, column=0)
    
    # End the root 
    root.mainloop()
    
    # Build a panda dataframe to add the counting data
    d_pd=pdd.DataFrame()
    
    # Change the working directory
    os.chdir(close_intro_window.output_directory)
    
    # Assign the pathways for the photos to variables. 
    if var2.get()==1 and var1.get()==0 and var3.get()==0: 
        print('input apatite + ed (EDM)')
        try:
            apatite_paths = choose_samplenames_apatite.name_apatites
        except AttributeError:
            error = 'no TL pictures were given for apatite'
            logger.error('ERROR: no pictures were given to the apatite button')
            print('ERROR: no pictures were given to the apatite button')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
            
        try: 
            apatite_paths_epi=list(choose_samplenames_apatite_epi.name_apatites)
        except AttributeError:
            error = 'no RL pictures were given for apatite'
            logger.error('ERROR: no pictures were given to the apatite epi button')
            print('ERROR: no pictures were given to the apatite epi button')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
            
        try:
            mica_paths=list(choose_samplenames_mica.name_micas) 
        except AttributeError:
            error = 'no TL pictures were given for mica'
            logger.error('ERROR: no TL pictures were given for the mica')
            print('ERROR: no TL pictures were given for the mica')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
            
        try: 
            mica_paths_epi=list(choose_samplenames_mica_epi.name_micas)
        except AttributeError:
            error = 'no RL pictures were given for mica'
            logger.error('ERROR: no RL pictures were given for the mica')
            print('ERROR: no RL pictures were given for the mica')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
            
        grains_number=int(1)
        z_stacks_apatite_number=int(2)
        z_stacks_mica_number=int(2)
        
    elif var1.get()==0 and var2.get()==0 and var3.get()==1: 
        print('input apatite LAFT')
       
        try: 
            apatite_paths=list(choose_samplenames_apatite.name_apatites)
        except AttributeError:
            error = 'no TL pictures were given for apatite'
            logger.error('ERROR: no TL pictures were given for apatite')
            print('ERROR: no TL pictures were given for apatite') 
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
       
        try: 
            apatite_paths_epi=list(choose_samplenames_apatite_epi.name_apatites)
        except AttributeError:
            error = 'no RL pictures were given for apatite'
            logger.error('ERROR: no RL pictures were given for apatite')
            print('ERROR: no RL pictures were given for apatite')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
        
        
        grains_number=int(1)
        z_stacks_apatite_number=int(2)
        
    elif var1.get()==1 and var2.get()==0 and var3.get()==0: 
        print('input glass')
        #if it's a glass
        # just clone the paths in order not to have an error
        try:
            apatite_paths=list(choose_samplenames_mica.name_micas)
        except AttributeError:
            error = 'no TL pictures were given for mica'
            logger.error('ERROR: no TL pictures were given for mica ')
            print('ERROR: no TL pictures were given for mica')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
        try:
            apatite_epi_paths=list(choose_samplenames_mica_epi.name_micas)
        except AttributeError:
            error = 'no RL pictures were given for apatite'
            logger.error('ERROR: no RL pictures were given for mica')
            print('ERROR: no RL pictures were given for mica')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
            
        try:
            mica_paths=list(choose_samplenames_mica.name_micas)
        except AttributeError:
            error = 'no TL pictures were given for mica'
            logger.error('ERROR: no TL pictures were given for mica')
            print('ERROR: no TL pictures were given for mica')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
            
        try:    
            mica_epi_paths=list(choose_samplenames_mica_epi.name_micas)
        except AttributeError:
            error = 'no RL pictures were given for mica'
            logger.error('ERROR: no RL pictures were given for mica')
            print('ERROR: no RL pictures were given for mica')
            frameinfo = getframeinfo(currentframe())
            print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
            
        
        grains_number=int(1)
        z_stacks_apatite_number=int(2) 
        z_stacks_mica_number=int(2)
    else:
        print('input pathways is not as expected')
        
    # Create list of lists apatite when ap/ed or only ap is chosen
    if var2.get()==1 and var1.get()==0 and var3.get()==0 or var1.get()==0 and var2.get()==0 and var3.get()==1:
        list_of_lists_ap=[]
        l=list()
        i=1    
        
        while (i-1) < grains_number:
            j=z_stacks_apatite_number
            add=apatite_paths[:j]
            list_of_lists_ap.append(list(add))
            apatite_paths=apatite_paths[j:]
            i+=1
        print('list of lists apatite' +str(list_of_lists_ap))
    
    # Create list of lists mica when ap/ed or only ed is chosen
    if  var2.get()==1 and var1.get()==0 and var3.get()==0 or var2.get()==0 and var1.get()==1 and var3.get()==0:
        list_of_lists_mica=[]
        l=list()
        i=1
        while (i-1) < grains_number:
            j=z_stacks_mica_number
            add=mica_paths[:j]
            list_of_lists_mica.append(list(add))
            mica_paths=mica_paths[j:]
            i+=1
        print("list_of_lists_mica" + str(list_of_lists_mica))
          
    # Make indexes which I'll use later
    count_mica=0  
    count_glass=0
    
    # If no graticule is chosen
    if no_graticule.get() == 0 and polygon_graticule.get() == 0 and circular_graticule.get() == 0:
        fail='yes'
        root_fail_no_graticule = Tk()
        root_fail_no_graticule.title('Error')
        root_fail_no_graticule.geometry('+'+str(place_width)+'+'+str(place_height))
        space = Label(root_fail_no_graticule, text=" ")
        space.pack() 
        
        Header1= Label(root_fail_no_graticule, text="Please select a graticule")
        Header1.pack()
            
        def close_no_graticule_window_and_quit():
            # Close the window
            root_fail_no_graticule.destroy()  
        
        quit_button_quit = Button(root_fail_no_graticule,text = "Quit", command = close_no_graticule_window_and_quit, cursor="hand2")
        quit_button_quit.pack()
           
        root_fail_no_graticule.mainloop()  
        
    #==========================================================================
    # START HERE WITH COUNTING IF IT IS AN APATITE + ED COUPLE OR LAFT
    #==========================================================================   
    
    # Start here if it's an apatite/ed couple or only apatite (LAFT)
    elif var2.get()==1 and var1.get()==0 or var3.get()==1 and var2.get()==0 and var1.get()==0:
        fail='no'
        if instructions_window.get()==1:
            instructions_ap_ft = np.zeros((200,500,1), np.uint8)
            instructions_ap_ft.fill(255)
            cv2.namedWindow('AI-Track-tive instructions apatite fission track recognition')
        
            # Polygonal graticule instructions
            if circular_graticule.get() == 0 and polygon_graticule.get() == 1:
                cv2.putText(instructions_ap_ft,'Draw a polygon using left mouse button', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft,'Finish your polygon by pressing space ', (10,60) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
            
            # Circular graticule instructions
            elif circular_graticule.get() == 1 and polygon_graticule.get() == 0:
                cv2.putText(instructions_ap_ft,'Press left mouse button ', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft,'... where you want to place your circular graticule ', (10,60) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft,'Finish your region of interest by pressing space ', (10,90) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                
            # No graticule instructions
            elif circular_graticule.get() == 0 and polygon_graticule.get() == 0 and no_graticule.get() == 1:
                cv2.putText(instructions_ap_ft,'Press space to continue ', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                print('no graticule = no instructions needed')
                
            cv2.putText(instructions_ap_ft,'Press space to continue', (10,120) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)    
            cv2.imshow('AI-Track-tive instructions apatite fission track recognition',instructions_ap_ft)
            
        print('apatite fission tracks will be counted in the following code!'+str(apatite_paths))
        print('list of lists apatite'+str(list_of_lists_ap))

        for apatite_paths in list_of_lists_ap: 
            print('ANOTHER ROUND IN THE APATITE LIST')
            print(list_of_lists_ap)
            
            # source: 
            # https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python
                
            CANVAS_SIZE = (800,800)
            FINAL_LINE_COLOR = (255, 255, 255) 
            WORKING_LINE_COLOR = (1,1,1)
            
            # Fission track recognition in apatite

            print("Start of the apatite fission track recognition")

            
            # Make an empty list for the laplacian values (quantification for focus of the images)
            list_laplacians=list()
            
            # Name custom object
            classes = ["Track"]   
            
            # Perform a loop to calculate the laplacian values (with the goal of finding the best focussed image)
            for z in apatite_paths:
                img_path = glob.glob(z) 
                img = cv2.imread(z,cv2.IMREAD_GRAYSCALE)
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                list_laplacians.append(laplacian_var)
                    
            # Create a dictionary with the laplacian values and paths
            dict_laplacians=dict(zip(apatite_paths, list_laplacians))
            print('dict laplacians'+str(dict_laplacians))

            #Now I get the best focussed image according to the Laplacian filter
            z_focussed = max(dict_laplacians, key=dict_laplacians.get)
            print('z_focussed before changing'+str(z_focussed))
                        
            # Actually, I prefer to take one image above the most focussed image (because I was working with z-stacks before)
            # So... if there is more than one image given
            # but now step = 0 I guess? 
            if z_stacks_apatite_number>int(step):
                index_most_focussed=list(dict_laplacians.keys()).index(z_focussed)
                print('index most focussed')
                print(index_most_focussed)
                index_most_focussed_adapted=index_most_focussed-step
                # Now change it
                z_focussed = list(dict_laplacians)[index_most_focussed_adapted]
                
            # If only one is given
            else: 
                pass

            print('z_focussed after changing'+str(z_focussed))
            
            # Read deep neural network and configuration file for apatite fission track recognition
            try:
                net = cv2.dnn.readNet(close_intro_window.location_model_apatite, close_intro_window.location_model_testing)
            except:
                error = 'the location for the apatite DNN or .cfg file is not right'
                print('the location for the deep neural network for apatite or the .cfg file is not right')
                logger.error('ERROR: the location for the deep neural network for apatite or the .cfg file is not right')
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                fail='yes'
                break
            
            # Get layer names
            layer_names = net.getLayerNames() 
            
            # Name custom object
            classes = ["Track"]         
            
            # Images path
            ap_path = glob.glob(z_focussed)  
            imageslayer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
            # Loading image
            img = cv2.imread(ap_path[0])
            height, width, channels = img.shape
            
            # Check if the images have all the right size
            if px == width and px == height:
                print('right size')
            else:
                error = 'the TL image of apatite is not the right size'
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                fail='yes'
                break
            
            # Cut a window of 100 µm on 100 µm Resize image 
            img = img [int(s1):int(s2), int(s1):int(s2)] 
            s=1
            
            # Resize depending on screen width and height
            height, width, channels = img.shape
            print('width after slicing but before resizing' + str(width))       
            
            # If image is too small, make it larger
            while width < 0.80 * screen_height:
                img = cv2.resize(img, None, fx=1.05, fy=1.05)
                height, width, channels = img.shape
                
            # If image is too large, make it smaller
            while width > 0.80 * screen_height:
                img = cv2.resize(img, None, fx=0.95, fy=0.95)
                height, width, channels = img.shape
                
            # Get the image shape properties again after resizing 
            height, width, channels = img.shape
            
            # REPEAT BUT NOW FOR EPISCOPIC APATITE IMAGES 
            print('episcopic light image resizing starts here') 
            print(choose_samplenames_apatite_epi.name_apatites)
            croppedimage_epi = cv2.imread(choose_samplenames_apatite_epi.name_apatites[0])
            
            height_epi, width_epi, channels_epi = croppedimage_epi.shape
            
            # Check if the images have all the right size
            if px == width_epi and px == height_epi:
                print('right size')
            else:
                error = 'the RL image of apatite is not the right size'
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                fail='yes'
                break
            
            # Window of 100 µm on 100 µm
            print('s1 epi is '+str(s1))
            print('s2 epi is '+str(s2)) 
            croppedimage_epi = croppedimage_epi[int(s1):int(s2), int(s1):int(s2)]

            # Half the size 
            width_epi = s3
            print('width apatite epi s3')
            print(width_epi)
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            # If image is too small, enlarge it 
            while width_epi < 0.80*screen_height:
               croppedimage_epi = cv2.resize(croppedimage_epi, None, fx = 1.05, fy = 1.05)
               height_epi, width_epi, channels_mica = croppedimage_epi.shape
               print('height apatite epi is '+str(height_epi))
                
            # If image is too large, make it smaller 
            while width_epi > 0.80*screen_height:
                croppedimage_epi = cv2.resize(croppedimage_epi, None, fx= 0.95, fy=0.95)
                height_epi, width_epi, channels_mica = croppedimage_epi.shape
                print('width apatite epi is '+str(width_epi))
            
            # Get new size 
            height_epi, width_epi, channels_epi = croppedimage_epi.shape
            print('width epi ' +str(width_epi)) 
            print('height epi ' +str(height_epi))
            
            # Repeat for diascopic apatites images
            print('diascopic unfocussed light image resizing starts here') 
            index_focussed=index_most_focussed
            index_unfocussed=index_focussed-1 # setting for our microscope
            croppedimage_unfocussed = cv2.imread(list_of_lists_ap[0][int(index_unfocussed)])
            
            # Get sizes for diascopic apatite images
            croppedimage_unfocussed_height, croppedimage_unfocussed_width, croppedimage_unfocussed_channels = croppedimage_unfocussed.shape
            
            # Check if the images have all the right size
            if px == croppedimage_unfocussed_width and px == croppedimage_unfocussed_height:
                print('right size')
            else:
                error = 'an image is not the right size'
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                fail='yes'
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                break
            
            # Window of 100 µm on 100 µm
            print('s1 unfocussed is '+str(s1))
            print('s2 unfocussed is '+str(s2)) 
            croppedimage_unfocussed = croppedimage_unfocussed[int(s1):int(s2), int(s1):int(s2)]
            
            # Half the size 
            width_unfocussed = s3
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            while width_unfocussed < 0.80 * screen_height:
                croppedimage_unfocussed = cv2.resize(croppedimage_unfocussed, None, fx = 1.05, fy = 1.05)
                height_unfocussed, width_unfocussed, channels_unfocussed = croppedimage_unfocussed.shape
                print('height mica unfocussed from glass is '+str(height_unfocussed))
            
            # Resize
            while width_unfocussed > 0.80 * screen_height:
                croppedimage_unfocussed = cv2.resize(croppedimage_unfocussed, None, fx= 0.95, fy=0.95)
                height_unfocussed, width_unfocussed, channels_unfocussed = croppedimage_unfocussed.shape
            
            # Get size after resizing
            height_unfocussed, width_unfocussed, channels_unfocussed = croppedimage_unfocussed.shape 
            print('width unfocussed mica ' +str(width_unfocussed)) 
            print('height unfocussed mica' +str(height_unfocussed))
            
            # Draw a polygonal graticule if the polygon is not predefined
            if circular_graticule.get() == 0 and polygon_graticule.get() == 1 and len(close_window.t)==0:
                pd = PolygonDrawerAp("Please draw a polygon using left mouse button and finish with right mouse button")
                croppedimage = pd.run()
                polygon_points = pd.points
                print("Polygon = %s" % pd.points)
                
                # Calculate area of polygon
                area = PolygonArea(pd.points,width)
                
            # Draw a polygonal graticule if the polygon is predefined 
            elif circular_graticule.get() == 0 and polygon_graticule.get() == 1 and len(close_window.t)!=0:
                print('Now we initiate the predefined polygon part of the script')
                pd = PolygonDrawerApPredefined("Please draw a polygon using left mouse button and finish with right mouse button")
                croppedimage = pd.run()
                
                # Calculate area of polygon
                area = PolygonArea(close_window.t, width)
            
            # Draw a circular graticule
            elif circular_graticule.get() == 1 and polygon_graticule.get() == 0:
                pd = CircularROI("Please draw a polygon using left mouse button and finish with right mouse button",close_window.spot_diameter)
                croppedimage = pd.mask(close_window.spot_diameter)
                print(pd.coordinate_center_circle())
                
                # Calculate area of circle
                area = (float(0.5*close_window.spot_diameter)**2)*3.14
            
            # Draw no graticule if it was chosen not to have a graticule
            elif circular_graticule.get() == 0 and polygon_graticule.get() == 0 and no_graticule.get()== 1:
                polygon_points=[(0,0),(int(width),0),(int(width),int(width)),(0,int(width))]
                croppedimage = img # 
                
                # Calculate area of polygon
                area = PolygonArea(polygon_points, width)
            
            else:
                print('the type of graticule was not specified!')
            
            pix_int=int(close_intro_window.pix_entry)
            print('pix_int is '+str(pix_int))
           
            # Detecting objects using our deep neural network 
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # same as default
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            # Make some new lists to add stuff
            class_ids = []
            confidences = []
            boxes = []
            print('len boxes is '+str(len(boxes)))
            
            # Convert to grayscale object
            gray_version_apatite = cv2.cvtColor(croppedimage, cv2.COLOR_RGB2GRAY)
            
            # Add the detected objects to the lists that I've made before 
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.1: #default is 0.3, now it is 0.1 which means that every track with 10% confidence is picked
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                    
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
            
                        # Develop an if-else structure that erases the found tracks covered by the polygon
                        if np.any(gray_version_apatite[center_y,center_x]) == 0:
                            if np.any(gray_version_apatite[y,x]) == 0:
                               pass
                            else:
                                pass
  
                        else:
                            # both are not in black
                            if np.any(gray_version_apatite[y,x]) != 0:
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                            
                            # center is not in black , edge is in black
                            else:
                                #print('else')
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                                
            print('len boxes is '+str(len(boxes)))
                
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.6)  # 0.5 and 0.4 resp originally. 
            # But I tweeked it to have better results for the application for which we use it
            # If you tweak the first value (behind confidences), you change the identification treshokld (confidence if I remember well, see opencv2 website)
            # If you raise the second value, you better detect the overlapping tracks 
            
            # Determine font
            font2 = cv2.FONT_HERSHEY_PLAIN
            
            # Make list to store the rectangles for LabelImgFormatter function 
            rect_txt_list = list()
            
            # Draw rectangles for every detected track  
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    color = colors[class_ids[i]]
                    color_text=(1,1,0)
                    cv2.rectangle(croppedimage, (x, y), (x + w, y + h), (200,0,0), 1)
                    cv2.rectangle(croppedimage_epi, (x, y), (x + w, y + h), (200,0,0), 1)
                    cv2.rectangle(croppedimage_unfocussed, (x, y), (x + w, y + h), (200,0,0), 1)
                    
                    # Txt file needs fractions
                    rect_txt = list()
                    x_txt = str((0.5*w + x)/(width))
                    rect_txt.append(x_txt[:8])
                    y_txt = str((0.5*w + y)/width)
                    rect_txt.append(y_txt[:8])
                    w_txt = str(w/width)
                    rect_txt.append(w_txt[:8])
                    h_txt = str(h/width)
                    rect_txt.append((h_txt[:8]))
                    
                    # Append the self-identified rectangle to the list 
                    rect_txt_list.append(rect_txt)
                    
                    # produce the .txt file 
                    print(labelImgformatter(rect_txt_list,choose_samplenames_apatite.name_apatites[0]))     
                                                            
                    # If you want to see the confidence score for every detected track...
                    #if confidences[i]<0.30:    
                        #cv2.putText(croppedimage,f"{confidences[i]:.0%}",(x,y),font2,1,(200,0,0),1)
                
                #cv2.putText(croppedimage, str(len(boxes)) +"tracks", (50, 50), font, 3, color_text, 2)
                
                else: 
                    pass
            
            cv2.destroyAllWindows()
            cv2.imshow("Manual Review process", croppedimage)
            
            print("ML fission track recognition ended")
            
            #==================================================================
            # APATITE FISSION TRACK COUNTING: MANUALLY ADDING TRACKS STARTS HERE
            #==================================================================
            
            # Instructions window 
            if instructions_window.get()==1:
                instructions_ap_ft_m = np.zeros((300,600,1), np.uint8)
                instructions_ap_ft_m.fill(255)
                cv2.namedWindow('AI-Track-tive instructions apatite fission track recognition')
                cv2.putText(instructions_ap_ft_m,' - Automatic FT recognition has ended.', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                cv2.putText(instructions_ap_ft_m,' - You can now change z-level and RL/TL using the trackbar.', (10,55) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft_m,' - You can also switch light sources using your mouse wheel', (10,80) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft_m,' - Indicate the additional tracks by dragging a ', (10,105) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                cv2.putText(instructions_ap_ft_m,'  ... rectangle around the track using left mouse button', (10,130) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                cv2.putText(instructions_ap_ft_m,' - When a track is indicated, click on space on your keyboard', (10,155) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                cv2.putText(instructions_ap_ft_m,' - If the ML algorithm indicated something that is not a track', (10,180) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft_m,'  ... draw a rectangle using right mouse button and press space ', (10,205) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft_m,' - In order to appear these manually detected boxes, click space', (10,230) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.putText(instructions_ap_ft_m,' - If all tracks are added, click on the middle mouse button or CTRL+mouse move', (10,255) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                cv2.putText(instructions_ap_ft_m,' - Press space to continue', (10,280) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
    
                cv2.imshow('AI-Track-tive instructions apatite fission track recognition',instructions_ap_ft_m)
            
            # If you chose to review the fission tracks manually, proceed here 
            if var_manually_review.get()==1:
                print('review process starts')
                croppedimage_manually_added_tracks = []
                pdt = MissingTracksAp("Manual Review process")
                n_tracks_ap=len(indexes)
                n_tracks_ap_mistaken=0 
                count_ap_loop=0
                count_ap_loop_mistaken=0
                mft_ap=[]  # create lists otherwise if nothing's added it gives an error
                mft_ap_false=[] # create lists otherwise if nothing's added it gives an error
                
                    
                #==================================================================
                # CHANGE THE OPENCV WINDOW TO ONE WITH Z AND EPI_DIA 
                #==================================================================
                
                # Set some variables 
                z=10
                rl_or_tl=0
                alpha_slider_max = 10
                alpha_slider_max2 = 1
                title_window = 'Manual Review process'
                epi_window = 'Epi or Dia'
                
                
                # First window needs to start at a polygonated image (z=10)
                alpha = 10 / alpha_slider_max
                beta = ( 1.0 - alpha )
                
                # Make a blended image and display it
                dst = cv2.addWeighted(croppedimage, alpha, croppedimage_unfocussed, beta, 0.0)
                cv2.imshow(title_window, dst)  
                
                # Now define the trackbar functions that need to take action once we clicked the trackbars
                def on_trackbar(val):
                    alpha = val / alpha_slider_max
                    beta = ( 1.0 - alpha )
                    dst = cv2.addWeighted(croppedimage, alpha, croppedimage_unfocussed, beta, 0.0)
                    cv2.imshow(title_window, dst)  
                    
                def on_trackbar_dia_epi(val):
                    alpha = val / alpha_slider_max2
                    beta = ( 1.0 - alpha )
                    dst = cv2.addWeighted(croppedimage, alpha, croppedimage_epi, beta, 0.0)
                    cv2.imshow(title_window, dst)   
                                            
                cv2.namedWindow(title_window)  
                
                # Make Trackbar
                trackbar_name = 'Alpha x %d' % alpha_slider_max
                cv2.createTrackbar('z', title_window , z, alpha_slider_max, on_trackbar)
                cv2.createTrackbar('RL/TL', title_window, rl_or_tl, alpha_slider_max2, on_trackbar_dia_epi)
  
                # Create a loop (while the middle mouse button (=> conected with self.stop) is not clicked)
                while pdt.stop == False:
                    print(' ')
                    print('pdt.latest_track_found is '+str(pdt.latest_track_found()))
                    
                    # IF LAST ADDED TRACK WAS ONE NOT DETECTED BY THE ML ALGORHITM                    
                    if pdt.latest_track_found()==1:

                        print('UNIDENTIFIED TRACKS')
                        print('count ap loop ' +str(count_ap_loop))

                        direction = 'up'
                        # Now a new loop will start for the z- and RL/TL functionality 
                        croppedimage_manually_added_tracks = pdt.findtracksmanually()   
                        
                        # If it's the first track that's added
                        if count_ap_loop==0:
                            mft_ap = pdt.list_manually_found_tracks()   # this is a list !  Manually Found Tracks => MFT  
                        
                        # If it's not the first track that's manually added
                        else:
                            print('not the first track')
                            mft_ap = pdt.list_manually_found_tracks()

                        print('round '+str(count_ap_loop)+str(' with the following list of manually found tracks')+str(' '+str(mft_ap)))
                        print('Draw rectangles now')

                        if mft_ap==[]:
                            pass
                        else:               
                            print('mft_ap:  ' +str(mft_ap))
                            try:
                                l=mft_ap[count_ap_loop]  # new coordinate to be added
                            except IndexError:
                                error = 'You added the tracks too fast. Please follow the exact instructions.'
                                logger.error('ERROR: you added the tracks too fast. Slow down please.')
                                print('IndexError')
                                frameinfo = getframeinfo(currentframe())
                                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                                
                            print('l is '+str(l))

                            if len(l)==2:
                                print('length is alright')
                                corn_one=l[0]
                                corn_two=l[1]

                                if int(corn_one[0])<int(corn_two[0]):
                                    x = int(abs(corn_one[0]))
                                    
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                        
                                else: 
                                    x = int(abs(corn_two[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                
                                print('x is '+str(x)+' and y is '+str(y))
                                
                                # Find w
                                w = int(abs(corn_one[0]-corn_two[0]))

                                
                                # Find h
                                h = int(abs(corn_one[1]-corn_two[1]))              
                                
                                # Add text to all three images 
                                cv2.rectangle(croppedimage, (x, y), (x + w, y + h), (0, 200, 0), 1)
                                cv2.rectangle(croppedimage_epi, (x, y), (x + w, y + h), (0, 200, 0), 1)
                                cv2.rectangle(croppedimage_unfocussed, (x, y), (x + w, y + h), (0, 200, 0), 1)
                                
                                # Txt file needs fractions
                                rect_txt = list()
                                x_txt = str((0.5*w + x)/(width))
                                rect_txt.append(x_txt[:8])
                                y_txt = str((0.5*w + y)/width)
                                rect_txt.append(y_txt[:8])
                                w_txt = str(w/width)
                                rect_txt.append(w_txt[:8])
                                h_txt = str(h/width)
                                rect_txt.append((h_txt[:8]))
                                
                                # Append the self-identified rectangle to the list 
                                rect_txt_list.append(rect_txt)
                                
                                # produce the .txt file 
                                print(labelImgformatter(rect_txt_list,choose_samplenames_apatite.name_apatites[0]))     
                                
                                # Add one to the counter
                                count_ap_loop=count_ap_loop+1                            
                            
                            else: 
                                print('space was hit')
                            
                    # IF FALSE TRACKS WERE IDENTIFIED    
                    elif pdt.latest_track_found()==0: 

                        print('FALSELY IDENTIFIED TRACKS')
                        print('count ap mistaken tracks loop '+str(count_ap_loop_mistaken))

                        
                        croppedimage_manually_added_tracks_mistaken = pdt.findtracksmanually()   # this is the image!
                        
                        # If it's the first track that's mistakenly added
                        if count_ap_loop_mistaken == 0:
                            print('first mistaken track added')
                            mft_ap_false = pdt.list_manually_found_tracks_mistaken()       
                            n_tracks_ap_mistaken=1
                            
                        # If it's not the first track that's mistakenly added   
                        else:
                            print('not first track mistakenly added')
                            mft_ap_false = pdt.list_manually_found_tracks_mistaken()
                            print('mft_ap_false'+str(mft_ap_false))
                            
                        if mft_ap_false == []:
                            pass
                        else:
                            print('mft_ap_false'+str(mft_ap_false))
                            l=mft_ap_false[count_ap_loop_mistaken]
                            print('l is '+str(l))
                                
                            if len (l)==2:
                                corn_one=l[0]
                                corn_two=l[1]

                                if int(corn_one[0])<int(corn_two[0]):
                                    x = int(abs(corn_one[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                else: 
                                    x = int(abs(corn_two[0]))
                                    
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                    
                                print('x is '+str(x)+' and y is '+str(y))
                                
                                # Find w
                                w = int(abs(corn_one[0]-corn_two[0]))
                                
                                # Find h
                                h = int(abs(corn_one[1]-corn_two[1]))
                                
                                # Add text to all three images
                                cv2.rectangle(croppedimage, (x, y), (x + w, y + h), (0, 0, 200), 1)
                                cv2.rectangle(croppedimage_epi, (x, y), (x + w, y + h), (0, 0, 200), 1)
                                cv2.rectangle(croppedimage_unfocussed, (x, y), (x + w, y + h), (0, 0, 200), 1)
                                
                                #cv2.putText(croppedimage,"mistaken",(x, y), font,1,(0, 0, 200), 1)
                            
                            # Add to the count    
                            count_ap_loop_mistaken+=1

                    else:
                        print('if space was hit')
                        pass 
                        
                    cv2.imshow("Manual Review process", croppedimage)
                    cv2.waitKey(0)
                    name_apatite= os.path.basename(ap_path[0])
                    cv2.imwrite(str(name_apatite)+".png",croppedimage)                               
                    cv2.imshow("Manual Review process", croppedimage)
                    print(z_focussed + " has "+str(n_tracks_ap)+" tracks")
            
            # If manually reviewing is not enabled
            else: 
                cv2.imshow("Manual Review process", croppedimage)
                cv2.waitKey(0)
                name_apatite = os.path.basename(ap_path[0])
                cv2.imwrite(str(name_apatite)+".png",croppedimage)                     
                cv2.imshow("Apatite fission track counting result manually adjusted"+z_focussed, croppedimage)
                n_tracks_ap=float(str(len(boxes)))
            
            # If manually reviewing is enabled
            if var_manually_review.get()==1:
                # Calculate and print all three types fission tracks found in apatite
                print('review manually button was clicked')
                print('mft_ap is '+str(mft_ap))
                print('mft_ap_false is '+str(len(mft_ap_false)))
                print('n_tracks_ap is '+str(n_tracks_ap))
                n_tracks_ap=n_tracks_ap+len(mft_ap)-len(mft_ap_false)  
                print('n_tracks_ap after recalculating based on manual corrections: '+str(n_tracks_ap))
            else:
                print('review manually button was NOT clicked')
                pass
            
            if n_tracks_ap == 0:
                track_density_ros = '/'
            else:
                #give the average track density in tr/cm²
                track_density_ros=float(n_tracks_ap)/float(area)*(10**8) 
                print('track density is ' + "{:.2e}".format(track_density_ros))
            
            cv2.destroyAllWindows()
            
            #==================================================================
            # MICA FISSION TRACK COUNTING FOR APATITE COUPLE
            #==================================================================
                      
            # Proceed with finding FT in mica. Only if it's an apatite + external detector couple
            if var2.get()==1 and var1.get()==0 and var3.get()==0:                 

                print("Mica fission track recognition starts here")

                n_tracks=float(0)
                
                if instructions_window.get()==1:
                    instructions_mica_ft = np.zeros((200,950,1), np.uint8)
                    instructions_mica_ft.fill(255)
                    cv2.namedWindow('AI-Track-tive instructions mica fission track recognition')
                    cv2.putText(instructions_mica_ft,'Press space to continue', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.imshow('AI-Track-tive instructions mica fission track recognition',instructions_mica_ft)
                    
                mica_paths=list_of_lists_mica[int(count_mica)]
                list_laplacians_mica=list()
        
                # Name custom object
                classes = ["Track"]        
                print(len(mica_paths)) 
                
                # Loop to find the best focussed picture using Laplacian function
                print('mica paths'+str(mica_paths))
                for z in mica_paths:
                    print(z)
                    img_path_mica = glob.glob(z)  
                    img_mica = cv2.imread(z,cv2.IMREAD_GRAYSCALE)
                    laplacian_var_mica = cv2.Laplacian(img_mica, cv2.CV_64F).var()
                    list_laplacians_mica.append(laplacian_var_mica)
                      
                # Make a dictionary with the laplacian values
                dict_laplacians_mica=dict(zip(mica_paths, list_laplacians_mica))
                print('dict laplacians mica'+str(dict_laplacians_mica))
                
                #Now I get the best focussed image according to the Laplacian filter
                z_focussed_mica = max(dict_laplacians_mica, key=dict_laplacians_mica.get)
                print('z_focussed_mica before changing '+str(z_focussed_mica))
                            
                # Actually, I prefer to take one image above the most focussed image
                # So... if there is more than one image given
                mica_z_entry_int=2
                if mica_z_entry_int>int(step):
                    index_most_focussed_mica=list(dict_laplacians_mica.keys()).index(z_focussed_mica)
                    print('index most focussed')
                    print(index_most_focussed_mica) 
                    index_most_focussed_adapted_mica=index_most_focussed_mica-step
                    # Now change it
                    z_focussed_mica = list(dict_laplacians_mica)[index_most_focussed_adapted_mica]
                # If only one is given
                else: 
                    pass

                print('z_focussed after changing'+str(z_focussed_mica))
                
                print('DNN mica')
                print(close_intro_window.location_model_mica)
                # Load Yolo deep neural network and testing (configuration file)
                try: 
                    net = cv2.dnn.readNet(close_intro_window.location_model_mica, close_intro_window.location_model_testing)
                    
                except:
                    error = 'the location for the mica DNN or .cfg file is not right'
                    print('the location for the deep neural network for mica is not right')
                    logger.error('ERROR: the location for the deep neural network for apatite or the .cfg file is not right')
                    frameinfo = getframeinfo(currentframe())
                    print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                    fail='yes'
                    break
                
                layer_names_mica = net.getLayerNames()   
                glob_z_focussed_mica = glob.glob(z_focussed_mica)    

                output_layers_mica = [layer_names_mica[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                colors_mica = np.random.uniform(0, 255, size=(len(classes), 3))
        
                # Loading image
                img_mica = cv2.imread(z_focussed_mica)

                # Resize to a size which is ~80% of your screen 
                height_mica, width_mica, channels_mica = img_mica.shape 

                
                # Cut a window of 100 µm on 100 µm
                print('s1 is '+str(s1))
                print('s2 is '+str(s2)) 
                img_mica = img_mica[int(s1):int(s2), int(s1):int(s2)]
                
                # Check if the images have all the right size
                if px == width_mica and px == height_mica:
                    print('right size')
                else:
                    error = 'the TL image of mica is not the right size'
                    logger.error('ERROR: the size of the images is not right')
                    print('ERROR: the size of the image is not right')
                    fail = 'yes'
                    frameinfo = getframeinfo(currentframe())
                    print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                    break
                
                # Resize if too small
                while width_mica < 0.80*screen_height:
                    print(width_mica)
                    img_mica = cv2.resize(img_mica, None, fx=1.05, fy=1.05)
                    height_mica, width_mica, channels_mica = img_mica.shape
                
                # Resize if too large
                while width_mica > 0.80*screen_height:
                    print(width_mica)
                    img_mica = cv2.resize(img_mica, None, fx=0.95, fy=0.95)
                    height_mica, width_mica, channels_mica = img_mica.shape
                
                height_mica, width_mica, channels_mica = img_mica.shape
                print('width mica ' +str(width_mica)) 
                print('height_mica' +str(height_mica))
                cv2.destroyAllWindows()
                
                
                # DO NOW THE SAME FOR THE EPISCOPIC IMAGE OF THE MICA 
                print('episcopic light image resizing starts here') 
                print(choose_samplenames_mica_epi.name_micas[0])
                croppedimage_mica_epi = cv2.imread(choose_samplenames_mica_epi.name_micas[0])
                
                # Get sizes
                height_mica_epi, width_mica_epi, channels_mica = croppedimage_mica_epi.shape
                
                # Check if the images have all the right size
                if px == width_mica_epi and px == height_mica_epi:
                    print('right size')
                else:
                    error = 'the RL image of mica is not the right size'
                    logger.error('ERROR: the size of the images is not right')
                    print('ERROR: the size of the image is not right')
                    fail='yes'
                    frameinfo = getframeinfo(currentframe())
                    print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                    break
                
                # Window of 100 µm on 100 µm
                print('s1 epi is '+str(s1))
                print('s2 epi is '+str(s2)) 
                croppedimage_mica_epi = croppedimage_mica_epi[int(s1):int(s2), int(s1):int(s2)]
    
                # Half the size 
                width_mica_epi = s3
                print('width mica epi s3')
                print(width_mica_epi)
                
                # Resize mica depending on screen width and scale factor (see configuration window)
                # When too small, enlarge image
                while width_mica_epi < 0.80*screen_height:
                   croppedimage_mica_epi = cv2.resize(croppedimage_mica_epi, None, fx = 1.05, fy = 1.05)
                   height_mica_epi, width_mica_epi, channels_mica = croppedimage_mica_epi.shape
                   print('height mica epi from glass is '+str(height_mica_epi))
                   
                # When too large, make the image smaller
                while width_mica_epi > 0.80*screen_height:
                    croppedimage_mica_epi = cv2.resize(croppedimage_mica_epi, None, fx= 0.95, fy=0.95)
                    height_mica_epi, width_mica_epi, channels_mica = croppedimage_mica_epi.shape
                    print('width mica epi from glass is '+str(width_mica_epi))
                
                height_mica_epi, width_mica_epi, channels_mica_epi = croppedimage_mica_epi.shape
                print('width epi mica ' +str(width_mica_epi)) 
                print('height epi mica' +str(height_mica_epi))
                

                   
                # DO NOW THE SAME FOR THE DIASCOPIC IMAGE OF THE MICA WHICH IS UNFOCUSSED
                print('diascopic unfocussed light image resizing starts here') 
                index_focussed=index_most_focussed_mica 
                index_unfocussed=index_focussed-1 # setting for our microscope
                croppedimage_mica_unfocussed = cv2.imread(list_of_lists_mica[0][int(index_unfocussed)])
                
                # Get sizes
                height_mica_unfocussed, width_mica_unfocussed, channels_mica_unfocussed = croppedimage_mica_unfocussed.shape
                
                # Check if the images have all the right size
                if px == width_mica_unfocussed and px == height_mica_unfocussed:
                    print('right size')
                else:
                    error = 'an image of mica is not the right size'
                    logger.error('ERROR: the size of the images is not right')
                    print('ERROR: the size of the image is not right')
                    fail='yes'
                    frameinfo = getframeinfo(currentframe())
                    print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                    break
                
                
                # Window of 100 µm on 100 µm
                print('s1 unfocussed is '+str(s1))
                print('s2 unfocussed is '+str(s2)) 
                croppedimage_mica_unfocussed = croppedimage_mica_unfocussed[int(s1):int(s2), int(s1):int(s2)]
                
                # Half the size 
                width_mica_unfocussed = s3
                
                # Resize mica depending on screen width and scale factor (see configuration window)
                while width_mica_unfocussed < 0.80*screen_height:
                    croppedimage_mica_unfocussed = cv2.resize(croppedimage_mica_unfocussed, None, fx = 1.05, fy = 1.05)
                    height_mica_unfocussed, width_mica_unfocussed, channels_mica = croppedimage_mica_unfocussed.shape
                    print('height mica from ap/ed couple is increasing to '+str(height_mica_epi))
                    
                while width_mica_unfocussed > 0.80*screen_height:
                    croppedimage_mica_unfocussed = cv2.resize(croppedimage_mica_unfocussed, None, fx= 0.95, fy=0.95)
                    height_mica_unfocussed, width_mica_unfocussed, channels_mica_unfocussed = croppedimage_mica_unfocussed.shape
                    print('height mica from ap/ed couple is decreasing to '+str(height_mica_epi))
                
                height_mica_unfocussed, width_mica_unfocussed, channels_mica_unfocussed = croppedimage_mica_unfocussed.shape 
                print('width epi mica ' +str(width_mica_unfocussed)) 
                print('height epi mica' +str(height_mica_unfocussed))
                
                if polygon_graticule.get()==1 and len(close_window.t)==0:
                    print('mica needs a polygonal mask which is chosen by the user')
                    # Draw a polygon
                    if __name__ == "__main__":
                        pd_mica = PolygonDrawerMica("Manual Review process")
                        croppedimage_mica = pd_mica.runmica()
                        #cv2.imwrite("polygonmica.png", croppedimage_mica)  # not sure if it saves it well here (PROBLEM)
                        # Calculate area of polygon
                        area = PolygonArea(pd.points,width_mica)
                        pix_int=int(close_intro_window.pix_entry)
                        print("Polygon = %s" % pd.points)
                
                elif polygon_graticule.get()==1 and len(close_window.t)!=0:  # same line as before (PROBLEM)
                    print('mica needs a polygonal mask')
                    # Draw a polygon
                    if __name__ == "__main__":
                        pd_mica = PolygonDrawerMica("Click using left mouse button to define a polygon. End with right mouse button.")
                        croppedimage_mica = pd_mica.runmica()
                        #cv2.imwrite("polygonmica.png", croppedimage_mica)
                        # Calculate area of polygon
                        area = PolygonArea(close_window.t,width_mica)
                        pix_int=int(close_intro_window.pix_entry)
                        print("Polygon = %s" % close_window.t)
                
                elif circular_graticule.get()==1:
                    print('mica needs to have a circular mask')
                    # Draw a circle
                    if __name__ == "__main__":    
                        pd_mica = CircularROIMica("Click using left mouse button to define a polygon. End with right mouse button.")
                        coord=pd.coordinate_center_circle()
                        croppedimage_mica = pd_mica.mask(close_window.spot_diameter,coord)      
                        #cv2.imwrite("polygonmica.png",croppedimage_mica)
                               
                # Draw no graticule if it was chosen that no graticule is needed
                elif no_graticule.get()== 1:
                    if __name__ == "__main__":    
                        polygon_points=[(0,0),(int(width_mica),0),(int(width_mica),int(width_mica)),(0,int(width_mica))]
                        croppedimage_mica = img_mica 
                        #cv2.imwrite("polygonmica.png",croppedimage_mica)
                                   
                else:
                    print('no circle or polygonal mask')
                    pass
                    
                cv2.imshow("Manual Review process",croppedimage_mica)

                print('size mica'+str(croppedimage_mica.shape))
              
                # Detecting objects using deep neural network 
                blob_mica = cv2.dnn.blobFromImage(img_mica, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob_mica)
                outs_mica = net.forward(output_layers_mica)
                n_tracks=0
                
                # set some variables again
                class_ids_mica = []
                confidences_mica = []
                boxes_mica = []
                rect_txt_list = []
                w=float(0)
                h=float(0)
                x=float(0)
                y=float(0)
                
                gray_version = cv2.cvtColor(croppedimage_mica, cv2.COLOR_RGB2GRAY)
                
                #==============================================================
                # MICA FISSION TRACK: RECTANGLE DRAWING
                #==============================================================
                
                for out_mica in outs_mica:
                    for detection_mica in out_mica:
                        
                        scores_mica = detection_mica[5:]
                        class_id_mica = np.argmax(scores_mica)
                        confidence_mica = scores_mica[class_id_mica]
                        
                        if confidence_mica > 0.1: #default is 0.3, it was 0.5 before 18 december 
                            center_x_mica = int(detection_mica[0] * width_mica)
                            center_y_mica = int(detection_mica[1] * height_mica)
                            w = int(detection_mica[2] * width_mica)
                            h = int(detection_mica[3] * height_mica)
                                                        
                            # Rectangle coordinates
                            x = int(center_x_mica - w / 2)
                            y = int(center_y_mica - h / 2)
        
                            # If center of rectangle is black
                            if np.any(gray_version[center_y_mica,center_x_mica]) == 0:  
                                # If left upper corner is black
                                if np.any(gray_version[y,x]) == 0:   #ATTENTION: Numpy coordinate system is reversed
                                    pass
                                else:
                                    pass
                            
                            # If center of rectangle is colored
                            else:
                                # If left upper corner is also colored 
                                if np.any(gray_version[y,x])!=0: #ATTENTION: Numpy coordinate system is reversed
                                    boxes_mica.append([x, y, w, h])
                                    confidences_mica.append(float(confidence_mica))
                                    class_ids_mica.append(class_id_mica)
                                
                                # If left upper corner is black but the center is not
                                else:
                                    boxes_mica.append([x, y, w, h])
                                    confidences_mica.append(float(confidence_mica))
                                    class_ids_mica.append(class_id_mica)
                
                indexes_mica = cv2.dnn.NMSBoxes(boxes_mica, confidences_mica, 0.01,0.6)  #0.01 was originally 0.5 and 0.4
                font2 = cv2.FONT_HERSHEY_PLAIN
                
                # DRAW RECTANGLES 
                
                for j in range(len(boxes_mica)):
                    if j in indexes_mica:
                        
                        # prepare to add rectangles
                        x, y, w, h = boxes_mica[j]
                        color_mica = colors_mica[class_ids_mica[j]]
                        color_text=(200,0,0)
 
                        # Txt file needs fractions
                        rect_txt = list()
                        x_txt = (w + 0.5*x)/width_mica
                        rect_txt.append(x_txt)
                        y_txt = (w + 0.5*y)/width_mica
                        rect_txt.append(y_txt)
                        w_txt = w/width_mica
                        rect_txt.append(w_txt)
                        h_txt = h/width_mica
                        rect_txt.append(h_txt)
                        rect_txt_list.append(rect_txt)
                        print(rect_txt)                        
                        
                        # Add rectangles
                        cv2.rectangle(croppedimage_mica, (x, y), (x + w, y + h), (200,0,0), 1)
                        cv2.rectangle(croppedimage_mica_epi, (x, y), (x + w, y + h), (200,0,0), 1)
                        cv2.rectangle(croppedimage_mica_unfocussed, (x, y), (x + w, y + h), (200,0,0), 1)
                        
                        #if confidences_mica[j] <0.949:
                            # Add text with every track to indicate the confidence score 
                            #cv2.putText(croppedimage_mica,f"{confidences_mica[j]:.0%}",(x,y),font,1,(200,0,0),2)
                            #pass
                            
                cv2.imshow("Manual Review process", croppedimage_mica)
    
                #==============================================================
                # MICA FISSION TRACK: MANUALLY ADDING FISSION TRACKS
                #==============================================================        
                
                print('len indexes mica'+str(len(indexes_mica)))
                n_tracks_mica_from_apatite=float(len(indexes_mica))
                print(n_tracks_mica_from_apatite)
                
                # If the manually reviewed button is clicked
                if var_manually_review.get()==1:                    
                    if instructions_window.get()==1:
                        instructions_mica_ft_m = np.zeros((300,600,1), np.uint8)
                        instructions_mica_ft_m.fill(255)
                        cv2.namedWindow('AI-Track-tive instructions mica fission track recognition')
                        cv2.putText(instructions_mica_ft_m,' - Automatic FT recognition has ended.', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                        cv2.putText(instructions_mica_ft_m,' - You can now change z-level and RL/TL using the trackbar.', (10,55) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                        cv2.putText(instructions_mica_ft_m,' - You can also handle switch light sources using your mouse wheel', (10,80) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                        cv2.putText(instructions_mica_ft_m,' - Indicate the additional tracks by dragging a ', (10,105) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                        cv2.putText(instructions_mica_ft_m,'  ... rectangle around the track using left mouse button', (10,130) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                        cv2.putText(instructions_mica_ft_m,' - When a track is indicated, click on space on your keyboard', (10,155) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                        cv2.putText(instructions_mica_ft_m,' - If the ML algorithm indicated something that is not a track', (10,180) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                        cv2.putText(instructions_mica_ft_m,'  ... draw a rectangle using right mouse button and press space ', (10,205) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                        cv2.putText(instructions_mica_ft_m,' - In order to appear these manually detected boxes, click space', (10,230) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                        cv2.putText(instructions_mica_ft_m,' - If all tracks are added, click on the middle mouse button or or CTRL+mouse move', (10,255) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                        cv2.putText(instructions_mica_ft_m,' - Press space to continue', (10,280) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                        cv2.imshow('AI-Track-tive instructions mica fission track recognition',instructions_mica_ft_m)
    
                    # Add the unrecognized fission tracks in mica                
                    pdt_mica=MissingTracksMica("Manual Review process")

                    #==================================================================
                    # CHANGE THE OPENCV WINDOW TO ONE WITH Z AND EPI_DIA 
                    #==================================================================
                    z=10
                    rl_or_tl=0
                    alpha_slider_max = 10
                    alpha_slider_max2 = 1
                    title_window = 'Manual Review process'
                    epi_window = 'Epi or Dia'
                    
                    alpha = 10 / alpha_slider_max
                    beta = (1.0 - alpha )
                    dst = cv2.addWeighted(croppedimage, alpha, croppedimage_unfocussed, beta, 0.0)
                    cv2.imshow(title_window,dst)
                    
                    # Now define the trackbar
                    def on_trackbar(val):
                        alpha = val / alpha_slider_max
                        beta = ( 1.0 - alpha )
                        dst = cv2.addWeighted(croppedimage_mica, alpha, croppedimage_mica_unfocussed, beta, 0.0)
                        cv2.imshow(title_window, dst)  
                        
                    def on_trackbar_dia_epi(val):
                        alpha = val / alpha_slider_max2
                        beta = ( 1.0 - alpha )
                        dst = cv2.addWeighted(croppedimage_mica, alpha, croppedimage_mica_epi, beta, 0.0)
                        cv2.imshow(title_window, dst)   
                                                
                    cv2.namedWindow(title_window)  
                    trackbar_name = 'Alpha x %d' % alpha_slider_max
                    cv2.createTrackbar('z', title_window , z, alpha_slider_max, on_trackbar)
                    cv2.createTrackbar('RL/TL', title_window, rl_or_tl, alpha_slider_max2, on_trackbar_dia_epi)
                    
                    # Set some variables
                    count_mica_loop=0
                    count_mica_loop_mistaken=0
                    mft_mica=[]
                    mft_mica_false=[]
                    
                    
                    
                    while pdt_mica.stop == False:                       
                        # If last added track was one not detected by the ML algorithm                   
                        if pdt_mica.latest_track_found()==1:
                            print('UNIDENTIFIED TRACKS')
                            print('count mica loop ' +str(count_mica_loop))
                            croppedimage_manually_added_tracks_mica = pdt_mica.findtracksmanually()   # this is the image!
                        
                            # If it's the first track that's added
                            if count_mica_loop==0:
                                mft_mica = pdt_mica.list_manually_found_tracks()   # this is a list !  #Manually Found Tracks => MFT  
                                
                            # If it's not the first track that's manually added
                            else:
                                mft_mica = pdt_mica.list_manually_found_tracks()    
                                print(len(mft_mica))
    
                            print('round '+str(count_mica_loop)+str(' with the following list of manually found tracks')+str(' '+str(mft_mica)))
                            print('Draw rectangles now')
                            
                            if mft_mica==[]:
                                pass
                            else:               
                                # New coordinate to be added
                                l=mft_mica[int(count_mica_loop)]  
                                print('mft mica is '+str(mft_mica))
                                print('l is '+str(l))
                                print('len (l) is '+str(len(l)))
    
                                if len(l)==2:
                                    corn_one=l[0]
                                    corn_two=l[1]
                                    
                                    if int(corn_one[0])<int(corn_two[0]):
                                        x = int(abs(corn_one[0]))
                                        if int(corn_one[1])<int(corn_two[1]):
                                            y = int(abs(corn_one[1]))
                                        else:
                                            y = int(abs(corn_two[1]))
                                            
                                    else: 
                                        x = int(abs(corn_two[0]))
                                        if int(corn_one[1])<int(corn_two[1]):
                                            y = int(abs(corn_one[1]))
                                        else:
                                            y = int(abs(corn_two[1]))
                                    
                                    print('x is '+str(x)+' and y is '+str(y))
                                    
                                    # Find w
                                    w = int(abs(corn_one[0]-corn_two[0]))
                                    print(w)
                                    
                                    # Find h
                                    h = int(abs(corn_one[1]-corn_two[1]))
                                    
                                    # Add rectangles to the image
                                    cv2.rectangle(croppedimage_mica, (x, y), (x + w, y + h), (0, 200, 0), 1)
                                    cv2.rectangle(croppedimage_mica_epi, (x, y), (x + w, y + h), (0, 200, 0), 1 ) 
                                    cv2.rectangle(croppedimage_mica_unfocussed, (x, y), (x + w, y + h), (0, 200, 0), 1) 
                                    
                                    # Txt file needs fractions
                                    rect_txt = list()
                                    x_txt = (w + 0.5*x)/width
                                    rect_txt.append(x_txt)
                                    y_txt = (w + 0.5*y)/width
                                    rect_txt.append(y_txt)
                                    w_txt = w/width
                                    rect_txt.append(w_txt)
                                    h_txt = h/width
                                    rect_txt.append(h_txt)
                                    rect_txt_list.append(rect_txt)
                                    print(rect_txt)    
                                    
                                    # Add text
                                    #cv2.putText(croppedimage_mica,"manually",(x,y),font,1,(0,100,0), 1)
                                    
                                count_mica_loop=count_mica_loop+1
                                print('end of the loop')
                                
                        # IF FALSE TRACKS WERE IDENTIFIED    
                        elif pdt_mica.latest_track_found()==0: 

                            print('FALSELY IDENTIFIED TRACKS')
                            print('count mica mistaken tracks loop '+str(count_mica_loop_mistaken))

                            
                            croppedimage_manually_added_tracks_mica_mistaken = pdt_mica.findtracksmanually()   # this is the image!
                            
                            # If it's the first track that's mistakenly added
                            if count_mica_loop_mistaken == 0:
                                mft_mica_false = pdt_mica.list_manually_false_tracks()   # this is a list !    
                                n_tracks_mica_mistaken=1
                                
                            # If it's not the first track that's mistakenly added   
                            else:
                                mft_mica_false = pdt_mica.list_manually_false_tracks()
                                print('mft_mica_false'+str(mft_mica_false))
                                
                            if mft_mica_false == []:
                                pass
                            else:
                                print('mft_mica_false'+str(mft_mica_false))
                                l=mft_mica_false[count_mica_loop_mistaken]
                                print('l is '+str(l))
                                    
                                if len (l)==2:
                                    corn_one=l[0]
                                    corn_two=l[1]
    
                                    if int(corn_one[0])<int(corn_two[0]):
                                        x = int(abs(corn_one[0]))
                                        if int(corn_one[1])<int(corn_two[1]):
                                            y = int(abs(corn_one[1]))
                                        else:
                                            y = int(abs(corn_two[1]))
                                            
                                    else: 
                                        x = int(abs(corn_two[0]))
                                        
                                        if int(corn_one[1])<int(corn_two[1]):
                                            y = int(abs(corn_one[1]))
                                        else:
                                            y = int(abs(corn_two[1]))
                                        
                                    # Find w
                                    w = int(abs(corn_one[0]-corn_two[0]))
                                    print(w)
                                    
                                    # Find h
                                    h = int(abs(corn_one[1]-corn_two[1]))
                                    
                                    # Add rectangles                                    
                                    cv2.rectangle(croppedimage_mica, (x, y), (x + w, y + h), (0, 0, 200), 1)
                                    cv2.rectangle(croppedimage_mica_epi, (x, y), (x + w, y + h), (0, 0, 200), 1)
                                    cv2.rectangle(croppedimage_mica_unfocussed, (x, y), (x + w, y + h), (0, 0, 200), 1)
                                    
                                    # Add text 
                                    #cv2.putText(croppedimage_mica,"mistaken",(x,y),font,1,(0,0,100),1)
                            
                            count_mica_loop_mistaken+=1    
    
                        else:
                            pass
                        
                        cv2.imshow("Manual Review process", croppedimage_mica)
                        cv2.waitKey(0)
                        name_mica = os.path.basename(mica_paths[0])
                        cv2.imwrite(str(name_mica)+".png",croppedimage_mica)

                    print(len(mft_mica))
                    print(len(mft_mica_false))
                    print(n_tracks_mica_from_apatite)

                else:
                    
                    cv2.imshow("Manual Review process", croppedimage_mica)
                    cv2.waitKey(0)                               
                    cv2.imshow("Manual Review process", croppedimage_mica)         
                    cv2.destroyAllWindows()  
            
                name_mica = os.path.basename(z_focussed_mica)    # this command takes the last part of the path
                #cv2.imwrite(str(name_mica)+"_.png", croppedimage_mica)
            
                # Calculate track densities in tr/cm²
                if n_tracks_ap == 0:
                    print('zero spontaneous tracks in apatite')
                    track_density_ros = '/'
                    rosroi='/'
                elif n_tracks_mica_from_apatite == 0:
                    print('zero tracks in mica found')
                    trac_density_roi = '/'
                    rosroi='/'
                else:
                    track_density_roi=float(n_tracks_mica_from_apatite)/float(area)*(10**8)  
                    print('induced track density is ' + "{:.2e}".format(track_density_roi))
                    if n_tracks_ap!=0:
                        rosroi=float(track_density_ros/track_density_roi)
                        print('rhos/rhoi is '+"{:.2e}".format(rosroi))
                    else:
                        rosroi='/'
                    
                # Calculate the accuracies of the track detector models
                if var_manually_review.get()==1:
                    accuracy_ap=round(float(100*n_tracks_ap)/(n_tracks_ap+len(mft_ap)))
                    accuracy_mica=round(float(100*n_tracks_mica_from_apatite)/(n_tracks_mica_from_apatite+len(mft_mica)))
                    inaccuracy_ap=float(100*len(mft_ap_false))/(len(mft_ap_false)+n_tracks_ap)
                    print('inaccuracy_ap is '+str(inaccuracy_ap))
                    inaccuracy_mica=float(100*len(mft_mica_false)/(n_tracks_mica_from_apatite))
                    print('inaccuracy_mica is '+str(inaccuracy_mica))     
                
                #==============================================================
                # EXPORT DATA
                #==============================================================
                # Manually reviewing enabled
                if var_manually_review.get()==1:
                    #Polygon                    
                    if polygon_graticule.get()==1 and circular_graticule.get()==0 and len(close_window.t)==0:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Name mica':[str(name_mica)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'Ni':[n_tracks_mica_from_apatite],
                                           'rhos/rhoi':[rosroi], 
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'Manually added induced tracks':[len(mft_mica)], 
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'model ed':[close_intro_window.location_model_mica], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           'laplacian mica':dict_laplacians_mica[z_focussed_mica],
                                           '% accuracy apatite':[accuracy_ap],
                                           '% accuracy mica':[accuracy_mica]}
                        
                    #If there was a custom chosen polygon     
                    elif polygon_graticule.get()==1 and circular_graticule.get()==0 and len(close_window.t)!=0:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Name mica':[str(name_mica)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'Ni':[n_tracks_mica_from_apatite],
                                           'rhos/rhoi':[rosroi], 
                                           'polygon':[close_window.t], 
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'Manually added induced tracks':[len(mft_mica)], 
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'model ed':[close_intro_window.location_model_mica], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           'laplacian mica':dict_laplacians_mica[z_focussed_mica],
                                           '% accuracy apatite':[accuracy_ap],
                                           '% accuracy mica':[accuracy_mica]}
                        
                    # Spot
                    elif polygon_graticule.get()==0 and circular_graticule.get()==1:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Name mica':[str(name_mica)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'Ni':[n_tracks_mica_from_apatite],
                                           'rhos/rhoi':[rosroi], 
                                           'center circ grat':[pd.coordinate_center_circle()],
                                           'diameter circ grat':[close_window.spot_diameter],
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'Manually added induced tracks':[str(len(mft_mica))], 
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'model ed':[close_intro_window.location_model_mica], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           'laplacian mica':dict_laplacians_mica[z_focussed_mica],
                                           '% accuracy apatite':[accuracy_ap],
                                           '% accuracy mica':[accuracy_mica]}  
                        
                    elif no_graticule.get()==1:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Name mica':[str(name_mica)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'Ni':[n_tracks_mica_from_apatite],
                                           'rhos/rhoi':[rosroi], 
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'Manually added induced tracks':[len(mft_mica)], 
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'model ed':[close_intro_window.location_model_mica], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           'laplacian mica':dict_laplacians_mica[z_focussed_mica],
                                           '% accuracy apatite':[accuracy_ap],
                                           '% accuracy mica':[accuracy_mica]}
                
                # Manually reviewing not enabled
                else:
                    # Polgyon
                    if polygon_graticule.get()==1 and circular_graticule.get()==0:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Name mica':[str(name_mica)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'Ni':[n_tracks_mica_from_apatite],
                                           'rhos/rhoi':[rosroi], 
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[str('not enabled')], 
                                           'Manually added induced tracks':[str('not enabled')],
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'model ed':[close_intro_window.location_model_mica], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           'laplacian mica':dict_laplacians_mica[z_focussed_mica]}   
                    # Spot
                    elif polygon_graticule.get()==1 and circular_graticule.get()==0:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Name mica':[str(name_mica)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'Ni':[n_tracks_mica_from_apatite],
                                           'rhos/rhoi':[rosroi], 
                                           'center cruc grat':[pd.coordinate_center_circle()], 
                                           'diameter circ grat':[close_window.spot_diameter],
                                           'Manually added spontaneous tracks':[str('not enabled')], 
                                           'Manually added induced tracks':[str('not enabled')],
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'model ed':[close_intro_window.location_model_mica], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           'laplacian mica':dict_laplacians_mica[z_focussed_mica]}  
                    
                    # No graticule
                    elif polygon_graticule.get()==1 and circular_graticule.get()==0:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Name mica':[str(name_mica)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'Ni':[n_tracks_mica_from_apatite],
                                           'rhos/rhoi':[rosroi], 
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[str('not enabled')], 
                                           'Manually added induced tracks':[str('not enabled')],
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'model ed':[close_intro_window.location_model_mica], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           'laplacian mica':dict_laplacians_mica[z_focussed_mica]}  

                # Append data to the 
                d_pd=d_pd.append((pdd.DataFrame(d)),ignore_index=True)    
                   
                print('panda below')
                print(d_pd)
                #print(d_pd.head())        
                count_mica+=1
            
            # LAFT 
            elif var3.get()==1 and var2.get()==0 and var1.get()==0:                               
                # export manually reviewed data
                if var_manually_review.get()==1:
                    
                    if n_tracks_ap!=0:    
                        accuracy_ap=float(100*n_tracks_ap)/(float(len(mft_ap))+float(n_tracks_ap))
                    else:
                        accuracy_ap='no spontaneous tracks found'
                    
                    # If it was a polygonal graticule
                    if circular_graticule.get()==0 and len(close_window.t)==0:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           '% accuracy apatite':[accuracy_ap]}
                    
                    # If there was a custom chosen polygon inserted 
                    elif circular_graticule.get()==0 and len(close_window.t)!=0:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'custom chosen polygon':[close_window.t], 
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           '% accuracy apatite':[accuracy_ap]}
                    
                    # If it was a circular graticule
                    elif circular_graticule.get()==1:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'center circ grat':[pd.coordinate_center_circle()],
                                           'diameter circ grat':[close_window.spot_diameter], 
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           '% accuracy apatite':[accuracy_ap]}
                    
                    # If it was chosen not to have a graticule
                    elif no_graticule.get()==1:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[len(mft_ap)],
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'laplacian ap':dict_laplacians[z_focussed],
                                           '% accuracy apatite':[accuracy_ap]}
                
                
                # Not manually reviewed: export the data here             
                elif var_manually_review.get()!=1:
                    # Polygonal region of interest 
                    if circular_graticule.get()==0:                    
                        d={'Name apatite':[str(name_apatite)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[str('not enabled')], 
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'laplacian ap':dict_laplacians[z_focussed]}
                    
                    # No region of interest, so 100x100 µm
                    elif no_graticule.get()==1:                    
                        d={'Name apatite':[str(name_apatite)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'polygon':[polygon_points], 
                                           'Manually added spontaneous tracks':[str('not enabled')], 
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'laplacian ap':dict_laplacians[z_focussed]}
                        
                    # Circular region of interest
                    else:
                        d={'Name apatite':[str(name_apatite)], 
                                           'Area':[area],
                                           'Ns':[n_tracks_ap],
                                           'center circ grat':[pd.coordinate_center_circle()],
                                           'diameter circ grat':[close_window.spot_diameter],
                                           'Manually added spontaneous tracks':[str('not enabled')], 
                                           'model apatite':[close_intro_window.location_model_apatite], 
                                           'laplacian ap':dict_laplacians[z_focussed]}
                        
                # Append data to the panda
                d_pd=d_pd.append((pdd.DataFrame(d)),ignore_index=True)    

        # Export to csv file  
        d_pd.to_csv(str(close_window.name)+".csv", sep=';', encoding='utf-8',index=False)

        root.mainloop()
    
    #==========================================================================
    # FISSION TRACK RECOGNITION IN GLASS STARTS HERE
    #==========================================================================
          
    elif var2.get()==0 and var1.get()==1 and var3.get()==0:        
        print("Arrived at glass section")
        
        # Show intruction window if it's desired
        if instructions_window.get()==1:
            instructions_glass_ft = np.zeros((200,950,1), np.uint8)
            instructions_glass_ft.fill(255)
            cv2.namedWindow('AI-Track-tive instructions glass fission track recognition')
            cv2.putText(instructions_glass_ft,'Press space to continue', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)    
            cv2.imshow('AI-Track-tive instructions glass fission track recognition',instructions_glass_ft)
        
        # Set some variables 
        CANVAS_SIZE = (800,800)
        FINAL_LINE_COLOR = (255, 255, 255)
        WORKING_LINE_COLOR = (1, 1, 1)

        # For every mica, fission track recognition initiates
        for mica_zstack in list_of_lists_mica:
            fail='no'
            sum_tracks=float(0)
            n_tracks=float(0)
            sum_tracks=0
            mica_paths=list_of_lists_mica[int(count_glass)]
            
            # Name custom object
            classes = ["Track"]
                    
            # Look for the most focussed picture using a Laplacian filter
            list_laplacians_mica=list()
            for z in mica_zstack:
                img_path_mica = glob.glob(z)
                img_mica = cv2.imread(z,cv2.IMREAD_GRAYSCALE) 
                laplacian_var_mica = cv2.Laplacian(img_mica, cv2.CV_64F).var()
                list_laplacians_mica.append(laplacian_var_mica)
                
            # Dictionary with paths and laplacian values
            dict_laplacians_mica=dict(zip(mica_zstack, list_laplacians_mica))
            print(dict_laplacians_mica)
            
            #Now I get the best focussed image according to the Laplacian filter 
            z_foc_mica_focussed = max(dict_laplacians_mica, key=dict_laplacians_mica.get)
            print('z_focussed_mica before changing'+str(z_foc_mica_focussed))
                        
            # Actually, I prefer to take one image above the most focussed image
            # So... if there is more than one image given
            mica_z_entry=2
            if mica_z_entry>int(step):
                index_most_focussed_mica=list(dict_laplacians_mica.keys()).index(z_foc_mica_focussed)
                print('index most focussed '+str(index_most_focussed_mica))
                index_most_focussed_adapted_mica=index_most_focussed_mica-step
                # Now change it
                z_focussed_mica = list(dict_laplacians_mica)[index_most_focussed_adapted_mica]
            
            # If only one is given
            else: 
                pass

            print('z_focussed after changing'+str(z_focussed_mica))

            # Load deep neural network 
            try:
                net = cv2.dnn.readNet(close_intro_window.location_model_mica, close_intro_window.location_model_testing)
            except:
                error = 'the location for the mica DNN or .cfg file is not right'
                print('the location for the deep neural network for mica or the .cfg file is not right')
                logger.error('ERROR: the location for the deep neural network for apatite or the .cfg file is not right')
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                fail='yes'
                break
                
                
            # Layer names
            layer_names_mica = net.getLayerNames()                     
            glob_z_foc_mica_focussed = glob.glob(z_foc_mica_focussed)    
            output_layers_mica = [layer_names_mica[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors_mica = np.random.uniform(0, 255, size=(len(classes), 3))
               
            # Insert here the path of your images
            random.shuffle(glob_z_foc_mica_focussed)
                      
            # Loading image
            img_mica = cv2.imread(z_foc_mica_focussed)
            
            # Get sizes
            height_mica, width_mica, channels_mica = img_mica.shape
            
            # Check if the images have all the right size
            if px == width_mica and px == height_mica:
                print('right size')
            else:
                error = 'the TL image of mica is not the right size'
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                fail='yes'
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                break
            
            # Slice a window of 100 µm on 100 µm
            print('s1 is '+str(s1)+' s2 is ' + str(s2))
            img_mica = img_mica[int(s1):int(s2), int(s1):int(s2)]
            print('width mica from glass is '+str(s3))
            # Half the size 
            width_mica = s3
            s_g=1
            
            print('screen height is '+str(screen_height))
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            while width_mica < 0.80*screen_height:
                img_mica = cv2.resize(img_mica, None, fx = 1.05, fy = 1.05)
                height_mica, width_mica, channels_mica = img_mica.shape
                print('height mica from glass is increasing to '+str(width_mica))
            
            # Resize mica 
            while width_mica > 0.80*screen_height:
                img_mica = cv2.resize(img_mica, None, fx= 0.95, fy=0.95)
                height_mica, width_mica, channels_mica = img_mica.shape
                print('height mica from glass is decreasing to '+str(width_mica))

            print('glass mica image resized')
            height_mica, width_mica, channels_mica = img_mica.shape
            print(str(width_mica)+str(' is size of width mica after the shape step'))
            
            
            # DO NOW THE SAME FOR THE EPISCOPIC IMAGE OF THE MICA 
            print('below')
            print(choose_samplenames_mica_epi.name_micas[0])
            img_mica_epi = cv2.imread(choose_samplenames_mica_epi.name_micas[0])
            
            # Get sizes
            height_mica_epi, width_mica_epi, channels_mica_epi = img_mica_epi.shape
            
            # Check if the images have all the right size
            if px == width_mica_epi and px == height_mica_epi:
                print('right size')
            else:
                error = 'the RL image of mica is not the right size'
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                fail='yes'
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                break
            
            # Window of 100 µm on 100 µm
            img_mica_epi = img_mica_epi[int(s1):int(s2), int(s1):int(s2)]

            # Half the size 
            width_mica_epi = s3
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            while width_mica_epi < 0.80*screen_height:
                img_mica_epi = cv2.resize(img_mica_epi, None, fx = 1.05, fy = 1.05)
                height_mica_epi, width_mica_epi, channels_mica = img_mica_epi.shape
                
            while width_mica_epi > 0.80*screen_height:
                img_mica_epi = cv2.resize(img_mica_epi, None, fx= 0.95, fy=0.95)
                height_mica_epi, width_mica_epi, channels_mica_epi = img_mica_epi.shape
            
            height_mica_epi, width_mica_epi, channels_mica_epi = img_mica_epi.shape
           
            
            # DO NOW THE SAME FOR THE DIASCOPIC IMAGE OF THE MICA WHICH IS UNFOCUSSED
            index_focussed=index_most_focussed_mica 
            index_unfocussed=index_focussed-1 # setting for our microscope
            img_mica_unfocussed = cv2.imread(list_of_lists_mica[0][int(index_unfocussed)])
            
            # Get sizes
            height_mica_unfocussed, width_mica_unfocussed, channels_mica = img_mica_unfocussed.shape
            
            # Check if the images have all the right size
            if px == width_mica_unfocussed and px == height_mica_unfocussed:
                print('right size')
            else:
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                fail='yes'
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                break
                
                
            # Window of 100 µm on 100 µm
            img_mica_unfocussed = img_mica_unfocussed[int(s1):int(s2), int(s1):int(s2)]
            
            # Half the size 
            width_mica_unfocussed = s3
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            while width_mica_unfocussed < 0.80*screen_height:
                img_mica_unfocussed = cv2.resize(img_mica_unfocussed, None, fx = 1.05, fy = 1.05)
                height_mica_unfocussed, width_mica_unfocussed, channels_mica = img_mica_unfocussed.shape
                #print('height mica from glass is '+str(height_mica_epi))
            
            # Resize 
            while width_mica_unfocussed > 0.80*screen_height:
                img_mica_unfocussed = cv2.resize(img_mica_unfocussed, None, fx= 0.95, fy=0.95)
                height_mica_unfocussed, width_mica_unfocussed, channels_mica_unfocussed = img_mica_unfocussed.shape
                #print('height mica from glass is '+str(height_mica_epi))
            
            height_mica_unfocussed, width_mica_unfocussed, channels_mica_unfocussed = img_mica_unfocussed.shape           
            
            # Calculate area of polygon
            print('width mica before another step')
            print(width_mica)
            polygon_points=[(0,0),(int(width_mica),0),(int(width_mica),int(width_mica)),(0,int(width_mica))]
            area = PolygonArea(polygon_points,width_mica)         
           
            # Detecting objects
            print('detecting objects in glass now')
            blob_mica = cv2.dnn.blobFromImage(img_mica, 0.00392, (416,416), (0, 0, 0), True, crop=False)
            net.setInput(blob_mica)
            outs_mica = net.forward(output_layers_mica)
            n_tracks=0                        
            
            # Showing informations on the screen
            class_ids_mica = []
            confidences_mica = []
            boxes_mica = []

            w=float(0)
            h=float(0)
            x=float(0)
            y=float(0)

            for out_mica in outs_mica:
                for detection_mica in out_mica:
                    scores_mica = detection_mica[5:]
                    class_id_mica = np.argmax(scores_mica)
                    confidence_mica = scores_mica[class_id_mica]
                    if confidence_mica > 0.1: #default is 0.3
                        
                        # Object detected
                        center_x_mica = int(detection_mica[0] * width_mica)
                        center_y_mica = int(detection_mica[1] * height_mica)
                        w = int(detection_mica[2] * width_mica)
                        h = int(detection_mica[3] * height_mica)
               
                        # Rectangle coordinates
                        x = int(center_x_mica - w / 2)
                        y = int(center_y_mica - h / 2)
           
                        # If the area in the box is not black 
                        if cv2.countNonZero(img_mica[center_x_mica,center_y_mica]) != 0:
                            boxes_mica.append([x, y, w, h])
                            confidences_mica.append(float(confidence_mica))
                            class_ids_mica.append(class_id_mica)
                        else:
                           pass
                       
            indexes_mica = cv2.dnn.NMSBoxes(boxes_mica, confidences_mica, 0.01,0.6)  #0.01 was originally 0.5 and 0.4
            font2 = cv2.FONT_HERSHEY_PLAIN
            n_tracks_mica = len(indexes_mica)
            
            rect_txt_list_glass_mica = list()
            
            for j in range(len(boxes_mica)):
                if j in indexes_mica:
                    x, y, w, h = boxes_mica[j]
                    color_mica = colors_mica[class_ids_mica[j]]
                    
                    # Txt file needs fractions
                    rect_txt = list()
                    x_txt = str((0.5*w + x)/(width_mica))
                    rect_txt.append(x_txt[:8])
                    y_txt = str((0.5*w + y)/width_mica)
                    rect_txt.append(y_txt[:8])
                    w_txt = str(w/width_mica)
                    rect_txt.append(w_txt[:8])
                    h_txt = str(h/width_mica)
                    rect_txt.append((h_txt[:8]))
                    rect_txt_list_glass_mica.append(rect_txt)
                    #print(rect_txt)
                    
                    # Draw rectangles
                    cv2.rectangle(img_mica, (x, y), (x + w, y + h), (200,0,0), 1)
                    cv2.rectangle(img_mica_epi, (x, y), (x + w, y + h), (200,0,0), 1)
                    cv2.rectangle(img_mica_unfocussed, (x, y), (x + w, y + h), (200,0,0), 1)
                    
                    # Put confidence intervals also there, only if they are below 95% 
                    #if confidences_mica[j]<1:    
                        #cv2.putText(img_mica,f"{confidences_mica[j]:.0%}",(x,y),font2,1.5,(0,0,0),1)
            
            print(rect_txt_list_glass_mica)   
            
            
            cv2.imshow("Manual Review process", img_mica)
    
            #key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            #==================================================================
            # FISSION TRACK RECOGNITION IN MICA FROM GLASS: MANUALLY ADDING FT 
            #==================================================================
            
            if var_manually_review.get()!=0:
                # Make an instructions window if it's checked 
                if instructions_window.get()==1:
                    instructions_glass_ft_m = np.zeros((360,650,1), np.uint8)
                    instructions_glass_ft_m.fill(255)
                    cv2.namedWindow('AI-Track-tive instructions glass fission track recognition')
                    cv2.putText(instructions_glass_ft_m,' - Automatic FT recognition has ended.', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - You can now change z-level and RL/TL using the trackbar.', (10,55) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - You can also handle switch light sources using your mouse wheel', (10,80) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - Indicate the additional tracks by dragging a ', (10,105) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,'  ... rectangle around the track using left mouse button', (10,130) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - When a track is indicated, click on space on your keyboard', (10,155) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - If the ML algorithm indicated something that is not a track', (10,180) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,'  ... draw a rectangle using right mouse button and press space ', (10,205) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - In order to appear these manually detected boxes, click space', (10,230) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - If all tracks are added, click on the middle mouse button or or CTRL+mouse move', (10,255) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - Press space to continue', (10,280) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    
                    cv2.imshow('AI-Track-tive instructions glass fission track recognition',instructions_glass_ft_m)
                            
                pdt_glass=MissingTracksMicaGlass("Manual Review process")
                print("pdt glass latest track found is: " +str(pdt_glass.latest_track_found()))
                
                #==================================================================
                # CHANGE THE OPENCV WINDOW TO ONE WITH Z AND EPI_DIA 
                #==================================================================
                   
                z=10
                rl_or_tl=1
                alpha_slider_max = 10
                alpha_slider_max2 = 1
                title_window = 'Manual Review process'
                epi_window = 'Epi or Dia'
                
                def on_trackbar(val):
                    alpha = val / alpha_slider_max
                    beta = ( 1.0 - alpha )
                    dst = cv2.addWeighted(img_mica, alpha, img_mica_unfocussed, beta, 0.0)
                    cv2.imshow(title_window, dst)  
                    
                def on_trackbar_dia_epi(val):
                    alpha = val / alpha_slider_max2
                    beta = ( 1.0 - alpha )
                    dst = cv2.addWeighted(img_mica, alpha, img_mica_epi, beta, 0.0)
                    cv2.imshow(title_window, dst)   
                                            
                cv2.namedWindow(title_window)  # naam aanpassen
                
                # Create trackbars
                trackbar_name = 'Alpha x %d' % alpha_slider_max
                cv2.createTrackbar('z', title_window , z, alpha_slider_max, on_trackbar)
                cv2.createTrackbar('RL/TL', title_window, rl_or_tl, alpha_slider_max2, on_trackbar_dia_epi)

                # Set some variables
                count_glass_loop=0
                count_glass_loop_mistaken=0
                mft_glass=[]
                mft_glass_false=[]
                
                while pdt_glass.stop==False:      
                    if pdt_glass.latest_track_found () == int(1):
                        print('UNIDENTIFIED TRACKS')
                        print('glass count mica loop '+str(count_glass_loop))
                        
                        image_glass_manually_added_tracks_mica = pdt_glass.findtracksmanually()
                        
                        # If it's the first track that's added
                        if count_glass_loop == 0:
                            mft_glass = pdt_glass.list_manually_found_tracks()
                            
                        # If it's not the first track that's manually added
                        else:
                            print('if it"s not the first track that"s to be added')
                            mft_glass = pdt_glass.list_manually_found_tracks()                        
                            print('round '+str(count_glass_loop)+str(' with the following list of manually added tracks')+str(mft_glass))
                        
                        print('Draw rectangles now')
                        
                        if mft_glass ==[]:
                            pass
                        else:
                            
                            l=mft_glass[int(count_glass_loop)]
                            print('mft glass) is '+str(mft_glass))
                            print('l is '+str(l))
                            print('len(l) is '+str(len(l)))
                            
                            # Add the track on the window
                            if len(l)==2:
                                corn_one=l[0]
                                print(corn_one)
                                corn_two=l[1]
                                if int(corn_one[0])<int(corn_two[0]):
                                    x = int(abs(corn_one[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                    
                                else: 
                                    x = int(abs(corn_two[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                    
                                print('x is '+str(x)+' and y is '+str(y))

                                w = int(abs(corn_one[0]-corn_two[0]))
                                h = int(abs(corn_one[1]-corn_two[1]))
                                
                                # Add rectangles
                                cv2.rectangle(img_mica, (x, y), (x + w, y + h), (0,200, 0), 1)
                                cv2.rectangle(img_mica_epi, (x, y), (x + w, y + h), (0,200, 0), 1)
                                cv2.rectangle(img_mica_unfocussed, (x, y), (x + w, y + h), (0,200, 0), 1)
                                
                                # Add text 
                                #cv2.putText(img_mica,"manually",(x,y),font,1,(0,100,0), 1)
                                
                                # Txt file needs fractions
                                rect_txt = list()
                                x_txt = str((0.5*w + x)/(width_mica))
                                rect_txt.append(x_txt[:8])
                                y_txt = str((0.5*w + y)/width_mica)
                                rect_txt.append(y_txt[:8])
                                w_txt = str(w/width_mica)
                                rect_txt.append(w_txt[:8])
                                h_txt = str(h/width_mica)
                                rect_txt.append((h_txt[:8]))
                                
                                # Append the self-identified rectangle to the list 
                                rect_txt_list_glass_mica.append(rect_txt)  
                                #print(rect_txt)
                                
                                # produce the .txt file 
                                print(labelImgformatter(rect_txt_list_glass_mica,choose_samplenames_mica.name_micas))
                                
                            count_glass_loop=count_glass_loop+1
                    
                    elif pdt_glass.latest_track_found()==0:

                        print('FALSELY IDENTIFIED TRACKS')
                        print('count mica mistaken tracks loop '+str(count_glass_loop_mistaken))

                        
                        image_glass_manually_added_tracks_mica=pdt_glass.findtracksmanually()
                        
                        if count_glass_loop_mistaken == 0:
                            mft_glass_false = pdt_glass.list_manually_false_tracks()
                            print('mft_glass_false'+str(mft_glass_false))
                        else:
                            mft_glass_false = pdt_glass.list_manually_false_tracks()
                            
                        if mft_glass_false == []:
                            pass
                        else:
                            print('mft_glass_false '+str(mft_glass_false))
                            l=mft_glass_false[count_glass_loop_mistaken]
                            print('l is '+str(l))
                            
                        if len(l)==2:
                                corn_one=l[0]
                                corn_two=l[1]

                                if int(corn_one[0])<int(corn_two[0]):
                                    x = int(abs(corn_one[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                else: 
                                    x = int(abs(corn_two[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                    
                                print('x is '+str(x)+' and y is '+str(y))
                                
                                # Find w
                                w = int(abs(corn_one[0]-corn_two[0]))
                                
                                # Find h
                                h = int(abs(corn_one[1]-corn_two[1]))
                                print('i is '+ str(i))
                
                                color_text=(0,0,200)
                                
                                # Add rectangles
                                cv2.rectangle(img_mica, (x, y), (x + w, y + h), color_text, 1)
                                cv2.rectangle(img_mica_epi, (x, y), (x + w, y + h), color_text, 1) 
                                cv2.rectangle(img_mica_unfocussed, (x, y), (x + w, y + h), color_text, 1) 
                                
                                # Add text
                                #cv2.putText(img_mica, "manually",(x,y),font,1,color_text, 2)
                                
                        count_glass_loop_mistaken+=1
                        
                    else:
                        pass
                    
                    cv2.imshow("Manual Review process", img_mica)
                    cv2.waitKey(0)
                    name_mica_glass = os.path.basename(mica_paths[0])
                    cv2.imwrite(str(name_mica_glass)+".png",img_mica)
                
                print(len(mft_glass))
                print(len(mft_glass_false))
                print(n_tracks_mica)
                                
            else:
                cv2.imshow("Manual Review process", img_mica)
                cv2.waitKey(0)
                cv2.imshow("Manual Review process", img_mica)                      
                cv2.destroyAllWindows()
                
            # Export 
            name_mica = os.path.basename(z_foc_mica_focussed)
            #cv2.imwrite(str(name_mica)+"_.png", img_mica)
            
            # Give the average track density in tr/cm²
            if n_tracks_mica == 0:
                track_density_roi = '/'
            else:
                track_density_roi=float(n_tracks_mica)/float(area)*(10**8)  
            
            print('induced track density is ' + "{:.2e}".format(track_density_roi))
                       
            # Append data 
            if var_manually_review.get()==1:
                d={'Name mica glass':[str(name_mica)],
                           'Ni':[n_tracks_mica],
                           'Area':[area],
                           'ind. tr. dens.':[track_density_roi],
                           'polygon':[polygon_points],
                           'Manually added induced tracks':[len(mft_glass)], 
                           'model ed' :[close_intro_window.location_model_mica],
                           'laplacian mica':dict_laplacians_mica[z_foc_mica_focussed],
                           'Accuracy':[(100*n_tracks_mica)/(len(mft_glass)+n_tracks_mica)]}
                
            elif var_manually_review.get()==0:
                d={'Name mica glass':[str(name_mica)],
                           'Ni':[n_tracks_mica],
                           'Area':[area],
                           'ind. tr. dens.':[track_density_roi],
                           'polygon':[polygon_points],
                           'model ed' :[close_intro_window.location_model_mica],
                           'laplacian mica':dict_laplacians_mica[z_foc_mica_focussed]}           
            else:
                pass
            
            #Append the data from this grain to the panda
            d_pd=d_pd.append((pdd.DataFrame(d)),ignore_index=True)   
            
        # Export data 
        print(d_pd)
        print(d_pd.head())  
        
        # Change working directory 
        d_pd.to_csv(str(close_window.name)+".csv", sep=';', encoding='utf-8',index=False)  
        count_glass+=1
       
        
       
    #==================================================================
    # USE AITRACKTIVE FOR ANNOTATING IMAGES AND PRODUCING .TXT FILES
    #==================================================================        
    
        

    
    elif annotate.get()==1:     
    # Show intruction window if it's desired
        if instructions_window.get()==1:
            instructions_glass_ft = np.zeros((200,950,1), np.uint8)
            instructions_glass_ft.fill(255)
            cv2.namedWindow('AI-Track-tive instructions glass fission track recognition')
            cv2.putText(instructions_glass_ft,'Press space to continue', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)    
            cv2.imshow('AI-Track-tive instructions glass fission track recognition',instructions_glass_ft)
        
        # # Set some variables 
        # CANVAS_SIZE = (800,800)
        # FINAL_LINE_COLOR = (255, 255, 255)
        # WORKING_LINE_COLOR = (1, 1, 1)
    
        # For every mica, fission track recognition initiates
        for mica_zstack in list_of_lists_mica:
            fail='no'
            sum_tracks=float(0)
            n_tracks=float(0)
            sum_tracks=0
            mica_paths=list_of_lists_mica[int(count_glass)]
            
            # Name custom object
            classes = ["Track"]
                    
            # Look for the most focussed picture using a Laplacian filter
            list_laplacians_mica=list()
            for z in mica_zstack:
                img_path_mica = glob.glob(z)
                img_mica = cv2.imread(z,cv2.IMREAD_GRAYSCALE) 
                laplacian_var_mica = cv2.Laplacian(img_mica, cv2.CV_64F).var()
                list_laplacians_mica.append(laplacian_var_mica)
                
            # Dictionary with paths and laplacian values
            dict_laplacians_mica=dict(zip(mica_zstack, list_laplacians_mica))
            print(dict_laplacians_mica)
            
            #Now I get the best focussed image according to the Laplacian filter 
            z_foc_mica_focussed = max(dict_laplacians_mica, key=dict_laplacians_mica.get)
            print('z_focussed_mica before changing'+str(z_foc_mica_focussed))
                        
            # Actually, I prefer to take one image above the most focussed image
            # So... if there is more than one image given
            mica_z_entry=2
            if mica_z_entry>int(step):
                index_most_focussed_mica=list(dict_laplacians_mica.keys()).index(z_foc_mica_focussed)
                print('index most focussed '+str(index_most_focussed_mica))
                index_most_focussed_adapted_mica=index_most_focussed_mica-step
                # Now change it
                z_focussed_mica = list(dict_laplacians_mica)[index_most_focussed_adapted_mica]
            
            # If only one is given
            else: 
                pass
    
            print('z_focussed after changing'+str(z_focussed_mica))
    
            # Load deep neural network 
            try:
                net = cv2.dnn.readNet(close_intro_window.location_model_mica, close_intro_window.location_model_testing)
            except:
                error = 'the location for the mica DNN or .cfg file is not right'
                print('the location for the deep neural network for mica or the .cfg file is not right')
                logger.error('ERROR: the location for the deep neural network for apatite or the .cfg file is not right')
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                fail='yes'
                break
                
                
            # Layer names
            layer_names_mica = net.getLayerNames()                     
            glob_z_foc_mica_focussed = glob.glob(z_foc_mica_focussed)    
            output_layers_mica = [layer_names_mica[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors_mica = np.random.uniform(0, 255, size=(len(classes), 3))
               
            # Insert here the path of your images
            random.shuffle(glob_z_foc_mica_focussed)
                      
            # Loading image
            img_mica = cv2.imread(z_foc_mica_focussed)
            
            # Get sizes
            height_mica, width_mica, channels_mica = img_mica.shape
            
            # Check if the images have all the right size
            if px == width_mica and px == height_mica:
                print('right size')
            else:
                error = 'the TL image of mica is not the right size'
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                fail='yes'
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                break
            
            # Slice a window of 100 µm on 100 µm
            print('s1 is '+str(s1)+' s2 is ' + str(s2))
            img_mica = img_mica[int(s1):int(s2), int(s1):int(s2)]
            print('width mica from glass is '+str(s3))
            # Half the size 
            width_mica = s3
            s_g=1
            
            print('screen height is '+str(screen_height))
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            while width_mica < 0.80*screen_height:
                img_mica = cv2.resize(img_mica, None, fx = 1.05, fy = 1.05)
                height_mica, width_mica, channels_mica = img_mica.shape
                print('height mica from glass is increasing to '+str(width_mica))
            
            # Resize mica 
            while width_mica > 0.80*screen_height:
                img_mica = cv2.resize(img_mica, None, fx= 0.95, fy=0.95)
                height_mica, width_mica, channels_mica = img_mica.shape
                print('height mica from glass is decreasing to '+str(width_mica))
    
            print('glass mica image resized')
            height_mica, width_mica, channels_mica = img_mica.shape
            print(str(width_mica)+str(' is size of width mica after the shape step'))
            
            
            # DO NOW THE SAME FOR THE EPISCOPIC IMAGE OF THE MICA 
            print('below')
            print(choose_samplenames_mica_epi.name_micas[0])
            img_mica_epi = cv2.imread(choose_samplenames_mica_epi.name_micas[0])
            
            # Get sizes
            height_mica_epi, width_mica_epi, channels_mica_epi = img_mica_epi.shape
            
            # Check if the images have all the right size
            if px == width_mica_epi and px == height_mica_epi:
                print('right size')
            else:
                error = 'the RL image of mica is not the right size'
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                fail='yes'
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                break
            
            # Window of 100 µm on 100 µm
            img_mica_epi = img_mica_epi[int(s1):int(s2), int(s1):int(s2)]
    
            # Half the size 
            width_mica_epi = s3
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            while width_mica_epi < 0.80*screen_height:
                img_mica_epi = cv2.resize(img_mica_epi, None, fx = 1.05, fy = 1.05)
                height_mica_epi, width_mica_epi, channels_mica = img_mica_epi.shape
                
            while width_mica_epi > 0.80*screen_height:
                img_mica_epi = cv2.resize(img_mica_epi, None, fx= 0.95, fy=0.95)
                height_mica_epi, width_mica_epi, channels_mica_epi = img_mica_epi.shape
            
            height_mica_epi, width_mica_epi, channels_mica_epi = img_mica_epi.shape
           
            
            # DO NOW THE SAME FOR THE DIASCOPIC IMAGE OF THE MICA WHICH IS UNFOCUSSED
            index_focussed=index_most_focussed_mica 
            index_unfocussed=index_focussed-1 # setting for our microscope
            img_mica_unfocussed = cv2.imread(list_of_lists_mica[0][int(index_unfocussed)])
            
            # Get sizes
            height_mica_unfocussed, width_mica_unfocussed, channels_mica = img_mica_unfocussed.shape
            
            # Check if the images have all the right size
            if px == width_mica_unfocussed and px == height_mica_unfocussed:
                print('right size')
            else:
                logger.error('ERROR: the size of the images is not right')
                print('ERROR: the size of the image is not right')
                fail='yes'
                frameinfo = getframeinfo(currentframe())
                print('ERROR located in line '+str(frameinfo.lineno)+' of the source code')
                break
                
                
            # Window of 100 µm on 100 µm
            img_mica_unfocussed = img_mica_unfocussed[int(s1):int(s2), int(s1):int(s2)]
            
            # Half the size 
            width_mica_unfocussed = s3
            
            # Resize mica depending on screen width and scale factor (see configuration window)
            while width_mica_unfocussed < 0.80*screen_height:
                img_mica_unfocussed = cv2.resize(img_mica_unfocussed, None, fx = 1.05, fy = 1.05)
                height_mica_unfocussed, width_mica_unfocussed, channels_mica = img_mica_unfocussed.shape
                #print('height mica from glass is '+str(height_mica_epi))
            
            # Resize 
            while width_mica_unfocussed > 0.80*screen_height:
                img_mica_unfocussed = cv2.resize(img_mica_unfocussed, None, fx= 0.95, fy=0.95)
                height_mica_unfocussed, width_mica_unfocussed, channels_mica_unfocussed = img_mica_unfocussed.shape
                #print('height mica from glass is '+str(height_mica_epi))
            
            height_mica_unfocussed, width_mica_unfocussed, channels_mica_unfocussed = img_mica_unfocussed.shape           
            
            # Calculate area of polygon
            print('width mica before another step')
            print(width_mica)
            polygon_points=[(0,0),(int(width_mica),0),(int(width_mica),int(width_mica)),(0,int(width_mica))]
            area = PolygonArea(polygon_points,width_mica)         
           
            # Detecting objects
            print('detecting objects in glass now')
            blob_mica = cv2.dnn.blobFromImage(img_mica, 0.00392, (416,416), (0, 0, 0), True, crop=False)
            net.setInput(blob_mica)
            outs_mica = net.forward(output_layers_mica)
            n_tracks=0                        
            
            # Showing informations on the screen
            class_ids_mica = []
            confidences_mica = []
            boxes_mica = []
    
            w=float(0)
            h=float(0)
            x=float(0)
            y=float(0)
    
            for out_mica in outs_mica:
                for detection_mica in out_mica:
                    scores_mica = detection_mica[5:]
                    class_id_mica = np.argmax(scores_mica)
                    confidence_mica = scores_mica[class_id_mica]
                    if confidence_mica > 0.1: #default is 0.3
                        
                        # Object detected
                        center_x_mica = int(detection_mica[0] * width_mica)
                        center_y_mica = int(detection_mica[1] * height_mica)
                        w = int(detection_mica[2] * width_mica)
                        h = int(detection_mica[3] * height_mica)
               
                        # Rectangle coordinates
                        x = int(center_x_mica - w / 2)
                        y = int(center_y_mica - h / 2)
           
                        # If the area in the box is not black 
                        if cv2.countNonZero(img_mica[center_x_mica,center_y_mica]) != 0:
                            boxes_mica.append([x, y, w, h])
                            confidences_mica.append(float(confidence_mica))
                            class_ids_mica.append(class_id_mica)
                        else:
                           pass
                       
            indexes_mica = cv2.dnn.NMSBoxes(boxes_mica, confidences_mica, 0.01,0.6)  #0.01 was originally 0.5 and 0.4
            font2 = cv2.FONT_HERSHEY_PLAIN
            n_tracks_mica = len(indexes_mica)
            
            rect_txt_list_glass_mica = list()
            
            for j in range(len(boxes_mica)):
                if j in indexes_mica:
                    x, y, w, h = boxes_mica[j]
                    color_mica = colors_mica[class_ids_mica[j]]
                    
                    # Txt file needs fractions
                    rect_txt = list()
                    x_txt = str((0.5*w + x)/(width_mica))
                    rect_txt.append(x_txt[:8])
                    y_txt = str((0.5*w + y)/width_mica)
                    rect_txt.append(y_txt[:8])
                    w_txt = str(w/width_mica)
                    rect_txt.append(w_txt[:8])
                    h_txt = str(h/width_mica)
                    rect_txt.append((h_txt[:8]))
                    rect_txt_list_glass_mica.append(rect_txt)
                    #print(rect_txt)
                    
                    # Draw rectangles
                    cv2.rectangle(img_mica, (x, y), (x + w, y + h), (200,0,0), 1)
                    cv2.rectangle(img_mica_epi, (x, y), (x + w, y + h), (200,0,0), 1)
                    cv2.rectangle(img_mica_unfocussed, (x, y), (x + w, y + h), (200,0,0), 1)
                    
                    # Put confidence intervals also there, only if they are below 95% 
                    #if confidences_mica[j]<0.50:    
                        #cv2.putText(img_mica,f"{confidences_mica[j]:.0%}",(x,y),font2,1,(200,200,0),1)
            
            print(rect_txt_list_glass_mica)   
            
            
            cv2.imshow("Manual Review process", img_mica)
    
            #key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            #==================================================================
            # FISSION TRACK RECOGNITION IN MICA FROM GLASS: MANUALLY ADDING FT 
            #==================================================================
            
            if var_manually_review.get()!=0:
                # Make an instructions window if it's checked 
                if instructions_window.get()==1:
                    instructions_glass_ft_m = np.zeros((360,650,1), np.uint8)
                    instructions_glass_ft_m.fill(255)
                    cv2.namedWindow('AI-Track-tive instructions glass fission track recognition')
                    cv2.putText(instructions_glass_ft_m,' - Automatic FT recognition has ended.', (10,30) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - You can now change z-level and RL/TL using the trackbar.', (10,55) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - You can also handle switch light sources using your mouse wheel', (10,80) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - Indicate the additional tracks by dragging a ', (10,105) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,'  ... rectangle around the track using left mouse button', (10,130) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - When a track is indicated, click on space on your keyboard', (10,155) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - If the ML algorithm indicated something that is not a track', (10,180) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,'  ... draw a rectangle using right mouse button and press space ', (10,205) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - In order to appear these manually detected boxes, click space', (10,230) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                    cv2.putText(instructions_glass_ft_m,' - If all tracks are added, click on the middle mouse button or CTRL+mouse move', (10,255) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    cv2.putText(instructions_glass_ft_m,' - Press space to continue', (10,280) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                    
                    cv2.imshow('AI-Track-tive instructions glass fission track recognition',instructions_glass_ft_m)
                            
                pdt_glass=MissingTracksMicaGlass("Manual Review process")
                print("pdt glass latest track found is: " +str(pdt_glass.latest_track_found()))
                
                #==================================================================
                # CHANGE THE OPENCV WINDOW TO ONE WITH Z AND EPI_DIA 
                #==================================================================
                   
                z=10
                rl_or_tl=1
                alpha_slider_max = 10
                alpha_slider_max2 = 1
                title_window = 'Manual Review process'
                epi_window = 'Epi or Dia'
                
                def on_trackbar(val):
                    alpha = val / alpha_slider_max
                    beta = ( 1.0 - alpha )
                    dst = cv2.addWeighted(img_mica, alpha, img_mica_unfocussed, beta, 0.0)
                    cv2.imshow(title_window, dst)  
                    
                def on_trackbar_dia_epi(val):
                    alpha = val / alpha_slider_max2
                    beta = ( 1.0 - alpha )
                    dst = cv2.addWeighted(img_mica, alpha, img_mica_epi, beta, 0.0)
                    cv2.imshow(title_window, dst)   
                                            
                cv2.namedWindow(title_window)  # naam aanpassen
                
                # Create trackbars
                trackbar_name = 'Alpha x %d' % alpha_slider_max
                cv2.createTrackbar('z', title_window , z, alpha_slider_max, on_trackbar)
                cv2.createTrackbar('RL/TL', title_window, rl_or_tl, alpha_slider_max2, on_trackbar_dia_epi)
    
                # Set some variables
                count_glass_loop=0
                count_glass_loop_mistaken=0
                mft_glass=[]
                mft_glass_false=[]
                
                while pdt_glass.stop==False:      
                    if pdt_glass.latest_track_found () == int(1):
                        print('UNIDENTIFIED TRACKS')
                        print('glass count mica loop '+str(count_glass_loop))
                        
                        image_glass_manually_added_tracks_mica = pdt_glass.findtracksmanually()
                        
                        # If it's the first track that's added
                        if count_glass_loop == 0:
                            mft_glass = pdt_glass.list_manually_found_tracks()
                            
                        # If it's not the first track that's manually added
                        else:
                            print('if it"s not the first track that"s to be added')
                            mft_glass = pdt_glass.list_manually_found_tracks()                        
                            print('round '+str(count_glass_loop)+str(' with the following list of manually added tracks')+str(mft_glass))
                        
                        print('Draw rectangles now')
                        
                        if mft_glass ==[]:
                            pass
                        else:
                            
                            l=mft_glass[int(count_glass_loop)]
                            print('mft glass) is '+str(mft_glass))
                            print('l is '+str(l))
                            print('len(l) is '+str(len(l)))
                            
                            # Add the track on the window
                            if len(l)==2:
                                corn_one=l[0]
                                print(corn_one)
                                corn_two=l[1]
                                if int(corn_one[0])<int(corn_two[0]):
                                    x = int(abs(corn_one[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                    
                                else: 
                                    x = int(abs(corn_two[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                    
                                print('x is '+str(x)+' and y is '+str(y))
    
                                w = int(abs(corn_one[0]-corn_two[0]))
                                h = int(abs(corn_one[1]-corn_two[1]))
                                
                                # Add rectangles
                                cv2.rectangle(img_mica, (x, y), (x + w, y + h), (0,200, 0), 1)
                                cv2.rectangle(img_mica_epi, (x, y), (x + w, y + h), (0,200, 0), 1)
                                cv2.rectangle(img_mica_unfocussed, (x, y), (x + w, y + h), (0,200, 0), 1)
                                
                                # Add text 
                                #cv2.putText(img_mica,"manually",(x,y),font,1,(0,100,0), 1)
                                
                                # Txt file needs fractions
                                rect_txt = list()
                                x_txt = str((0.5*w + x)/(width_mica))
                                rect_txt.append(x_txt[:8])
                                y_txt = str((0.5*w + y)/width_mica)
                                rect_txt.append(y_txt[:8])
                                w_txt = str(w/width_mica)
                                rect_txt.append(w_txt[:8])
                                h_txt = str(h/width_mica)
                                rect_txt.append((h_txt[:8]))
                                
                                # Append the self-identified rectangle to the list 
                                rect_txt_list_glass_mica.append(rect_txt)  
                                #print(rect_txt)
                                
                                # produce the .txt file 
                                print(labelImgformatter(rect_txt_list_glass_mica, choose_samplenames_mica.name_micas))
                                
                            count_glass_loop=count_glass_loop+1
                    
                    elif pdt_glass.latest_track_found()==0:
    
                        print('FALSELY IDENTIFIED TRACKS')
                        print('count mica mistaken tracks loop '+str(count_glass_loop_mistaken))
    
                        
                        image_glass_manually_added_tracks_mica=pdt_glass.findtracksmanually()
                        
                        if count_glass_loop_mistaken == 0:
                            mft_glass_false = pdt_glass.list_manually_false_tracks()
                            print('mft_glass_false'+str(mft_glass_false))
                        else:
                            mft_glass_false = pdt_glass.list_manually_false_tracks()
                            
                        if mft_glass_false == []:
                            pass
                        else:
                            print('mft_glass_false '+str(mft_glass_false))
                            l=mft_glass_false[count_glass_loop_mistaken]
                            print('l is '+str(l))
                            
                        if len(l)==2:
                                corn_one=l[0]
                                corn_two=l[1]
    
                                if int(corn_one[0])<int(corn_two[0]):
                                    x = int(abs(corn_one[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                else: 
                                    x = int(abs(corn_two[0]))
                                    if int(corn_one[1])<int(corn_two[1]):
                                        y = int(abs(corn_one[1]))
                                    else:
                                        y = int(abs(corn_two[1]))
                                    
                                print('x is '+str(x)+' and y is '+str(y))
                                
                                # Find w
                                w = int(abs(corn_one[0]-corn_two[0]))
                                
                                # Find h
                                h = int(abs(corn_one[1]-corn_two[1]))
                                print('i is '+ str(i))
                
                                color_text=(0,0,200)
                                
                                # Add rectangles
                                cv2.rectangle(img_mica, (x, y), (x + w, y + h), color_text, 1)
                                cv2.rectangle(img_mica_epi, (x, y), (x + w, y + h), color_text, 1) 
                                cv2.rectangle(img_mica_unfocussed, (x, y), (x + w, y + h), color_text, 1) 
                                
                                # Add text
                                #cv2.putText(img_mica, "manually",(x,y),font,1,color_text, 2)
                                
                        count_glass_loop_mistaken+=1
                        
                    else:
                        pass
                    
                    cv2.imshow("Manual Review process", img_mica)
                    cv2.waitKey(0)
                    name_mica_glass = os.path.basename(mica_paths[0])
                    cv2.imwrite(str(name_mica_glass)+".png",img_mica)
                
                print(len(mft_glass))
                print(len(mft_glass_false))
                print(n_tracks_mica)
                                
            else:
                cv2.imshow("Manual Review process", img_mica)
                cv2.waitKey(0)
                cv2.imshow("Manual Review process", img_mica)                      
                cv2.destroyAllWindows()
                
            # Export 
            name_mica = os.path.basename(z_foc_mica_focussed)
            #cv2.imwrite(str(name_mica)+"_.png", img_mica)
            
            # Give the average track density in tr/cm²
            if n_tracks_mica == 0:
                track_density_roi = '/'
            else:
                track_density_roi=float(n_tracks_mica)/float(area)*(10**8)  
            
            print('induced track density is ' + "{:.2e}".format(track_density_roi))
                       
            # Append data 
            if var_manually_review.get()==1:
                d={'Name mica glass':[str(name_mica)],
                           'Ni':[n_tracks_mica],
                           'Area':[area],
                           'ind. tr. dens.':[track_density_roi],
                           'polygon':[polygon_points],
                           'Manually added induced tracks':[len(mft_glass)], 
                           'model ed' :[close_intro_window.location_model_mica],
                           'laplacian mica':dict_laplacians_mica[z_foc_mica_focussed],
                           'Accuracy':[(100*n_tracks_mica)/(len(mft_glass)+n_tracks_mica)]}
                
            elif var_manually_review.get()==0:
                d={'Name mica glass':[str(name_mica)],
                           'Ni':[n_tracks_mica],
                           'Area':[area],
                           'ind. tr. dens.':[track_density_roi],
                           'polygon':[polygon_points],
                           'model ed' :[close_intro_window.location_model_mica],
                           'laplacian mica':dict_laplacians_mica[z_foc_mica_focussed]}           
            else:
                pass
            
            #Append the data from this grain to the panda
            d_pd=d_pd.append((pdd.DataFrame(d)),ignore_index=True)   
            
        # Export data 
        print(d_pd)
        print(d_pd.head())  
        
        # Change working directory 
        d_pd.to_csv(str(close_window.name)+".csv", sep=';', encoding='utf-8',index=False)  
        count_glass+=1
        
    else: 
        pass   

        
    #==========================================================================
    # TRACK COUNTING STOPPED: CONTINUE WITH MORE TKINTER WINDOWS
    #==========================================================================
    
    #change working directory in order to find back the microscope.ico icon 
    os.chdir(path_script_dirname)

    # If there was a problem with the input values
    if fail=="yes":
        cv2.destroyAllWindows()
        root_fail = Tk()
        #root_fail.iconbitmap('microscope.ico')
        root_fail.title('An error occurred')   
        root_fail.geometry('400x200+'+str(place_width)+'+'+str(place_height))
        space = Label(root_fail, text=" ").pack()
        text_fail = Label(root_fail, text=" The following error occured: ").pack()
        text_fail2 = Label(root_fail, text=str(error)).pack()
        text_fail3 = Label(root_fail, text="also check the info.log file").pack()
        space = Label(root_fail, text=" ").pack()         
        
        def close_third_window():
            # Close the window
            root_fail.destroy()      
        
        def quit_third_window():
            # Close the window and quit            
            global shutdown_script
            shutdown_script = 1
            root_fail.destroy()
            root_fail.quit()
            
            
        quit_button2 = Button(root_fail,text = "Ok, I'll restart", command = close_third_window, cursor="hand2")
        quit_button2.pack()
        
        quit_button3 = Button(root_fail,text = "Quit", command = quit_third_window, cursor="hand2")
        quit_button3.pack()
           
        root_fail.mainloop()    
        
    # If it didn't fail 
    else: 
        # New Dialogue window
        root_ap_ed = Tk()
        root_ap_ed.title('Trackflow Automatic Fission Track Counting software has ended')
        root_ap_ed.geometry('+'+str(place_width)+'+'+str(place_height))
        #root_ap_ed.iconbitmap('microscope.ico')
        list_samples=list()
        space = Label(root_ap_ed, text=" ")
        space.pack()   
        
        # End message   
        Header = Label(root_ap_ed, text="The results have been exported")
        Header.pack()
        
        # Quit button
        def quit_function():
            print('quit it')
            root_ap_ed.destroy()
            cv2.destroyAllWindows()
            global shutdown_script
            shutdown_script = 1
            root_ap_ed.quit()
            
        # Restart button
        def restart_window():
            print('restart it')
            root_ap_ed.destroy()
            cv2.destroyAllWindows()
            global shutdown_script
            shutdown_script = 0
            
        space = Label(root_ap_ed, text=" ")
        space.pack()
        quit_button = Button(root_ap_ed, text = "Quit", command = quit_function, cursor="hand2")
        quit_button.pack()        
        space = Label(root_ap_ed, text=" ")
        space.pack()        
        restart_button = Button(root_ap_ed, text = "Continue with another grain", command = restart_window, cursor="hand2")
        restart_button.pack()        
        space = Label(root_ap_ed, text=" ")
        space.pack()
           
        # Quit button
        def close_second_window():
            # Close the window
            root_ap_ed.destroy()         
        root_ap_ed.mainloop()
        



#==================================================================
# USE DNN FOR LIVE WINDOW VIEW 
#==================================================================               
                
#author: https://github.com/mukundsharma1995/yolov3-object-detection/blob/master/video.py

if live.get()==1 or live_mica.get()==1: 
    import argparse
    close_intro_window.location_model_apatite, 
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', help = 'path to yolo config file', default=close_intro_window.location_model_testing)
    
    if live.get()==1:
        try:
            ap.add_argument('-w', '--weights', help = 'path to yolo pre-trained weights', default=close_intro_window.location_model_apatite)
            args = ap.parse_args()
        except:
            ap.add_argument('-w', '--weights', help = 'path to yolo pre-trained weights', default=close_intro_window.location_model_mica)
            args = ap.parse_args()
    
    elif live_mica.get()==1:
        try:
            ap.add_argument('-w', '--weights', help = 'path to yolo pre-trained weights', default=close_intro_window.location_model_mica)
            args = ap.parse_args()
        except:
            ap.add_argument('-w', '--weights', help = 'path to yolo pre-trained weights', default=close_intro_window.location_model_mica)
            args = ap.parse_args()
        
    
    # Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
    def getOutputsNames(net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    
    # Darw a rectangle surrounding the object and its class name 
    def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    
        #label = str(classes[class_id])
    
        #color = COLORS[class_id]
    
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (255,0,0), 2)
    
        #cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # Define a window to show the cam stream on it
    window_title= "Custom Detector"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    
    
    # Load names classes
    # classes = None
    # with open(args.classes, 'r') as f:
    #     classes = [line.strip() for line in f.readlines()]
    # print(classes)
    
    #Generate color for each class randomly
    #COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Define network from configuration file and load the weights from the given weights file
    net = cv2.dnn.readNet(args.weights,args.config)
    
    # Define video capture for default cam
    #cap = cv2.VideoCapture(0)
    
    from PIL import ImageGrab 
        
    while cv2.waitKey(1) < 0:
        #img = ImageGrab.grab(bbox=(0,0,int(screen_height),int(screen_height))) # capturing NOT the full HD screen, but a square
        img = ImageGrab.grab(bbox=(0.25*int(screen_height),0.25*int(screen_height),0.90*int(screen_height),0.90*int(screen_height))) # capturing the full HD screen!
        image = np.array(img)
        
        blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
        width = image.shape[1]
        height = image.shape[0]
        net.setInput(blob)
        
        outs = net.forward(getOutputsNames(net))
        
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.01
        nms_threshold = 0.6
        
        
        for out in outs: 
            for detection in out:
                scores = detection[5:]#classes scores starts from index 5
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        # apply  non-maximum suppression algorithm on the bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
       
        # Put efficiency information.
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(image, label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        
        cv2.imshow(window_title, image)
        
    cv2.destroyAllWindows()
print('terminated')