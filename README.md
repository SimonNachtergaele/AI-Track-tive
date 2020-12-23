# AI-Track-tive: free software for automated recognition and counting of surface semi-tracks using computer vision (Artificial Intelligence)

https://gchron.copernicus.org/preprints/gchron-2020-32/

Welcome!

***A. Obtaining the freeware AI-Track-tive software:

WINDOWS users:
You can find a folder with the executable (application) file, which is easy to open for Windows users.
In order to start the program, just click on the AITracktivev1.7.exe file.

MAC users:
You have to use the source code (.py). The source code can be opened using Spyder IDE. (https://www.spyder-ide.org/)  
  
LINUX users:
Under construction

Tutorials can be found here: 
Old tutorial: https://www.youtube.com/watch?v=fSfit87vkrA&feature=youtu.be
New tutorial: coming soon!

***B. How to train your own Deep Neural Network?

1. Convert the training images 
I used for example 804x804px images. In order to resize the images I used ImageJ or Fiji (https://imagej.net/Downloads). I used process => Batch => Convert in order to convert the original 1608x1608px tiff files to a 804x804px jpeg format. 

2. Install LabelImg and draw boxes around the tracks in the images 
https://github.com/tzutalin/labelImg
Then, compress the pictures and .txt files in one .zip file and place this file images.zip in the yolov3 folder on your google Drive. 

3. Execute the Google Colab notebook for some hours 
Aim for 2000-3000 iterations. The iteration speed depends on the size of your images. 
From my experience it seems that from then on, the DNN is overtrained. 
