[AI-Track-tive: free software for automated recognition and counting of surface semi-tracks using computer vision (Artificial Intelligence)](https://gchron.copernicus.org/preprints/gchron-2020-32/)

![GitHub Logo](/logo-01.png)
Format: ![Alt Text](url)

**Welcome!**

*A. Obtaining the freeware AI-Track-tive software:* 

WINDOWS users:
The easiest option to run the code is by downloading the folder from Github and double-click the .exe file. 
You can find a folder with the executable (application) and also with the source code (.py). 

If you would like to view the Python code and perhaps do some changes, you can download the source code (.py) file. I did the debugging using the highly recommendable [Spyder IDE](https://docs.spyder-ide.org/current/index.html). 

MAC users:
You have to use the source code (.py). The source code can be opened using [Spyder IDE](https://www.spyder-ide.org/)  
Before you can use it, you need to install some Python modules:
[opencv](https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html)
[pyqt](https://pythonbasics.org/install-pyqt/)
[opencv-python-headless](https://pypi.org/project/opencv-python-headless/)
  
LINUX users:
You have to open the source code (.py) in a Python-based IDE, for example Thonny. 
I had no problems using the software on a Raspberry Pi 4B with 4GB RAM (€50) using Linux, after installing the following Python packages:
First of all: install pip for linux 
Then, "pip install pandas" (to install the pandas)
And don't forget "pip install opencv-python" (to install the open cv module). 
Using my ultra-cheap Raspberry Pi I noticed that the program didn't want to continue because opencv2 was not entirely well installed. I found some help on my favorite website (stackoverflow) and found that it's possible to solve this error with: 
*pip3 install opencv-python*
*sudo apt-get install libcblas-dev*
*sudo apt-get install libhdf5-dev*
*sudo apt-get install libhdf5-serial-dev*
*sudo apt-get install libatlas-base-dev*
*sudo apt-get install libjasper-dev*
*sudo apt-get install libqtgui4*
*sudo apt-get install libqt4-test*

During debugging I noticed that Windows uses typically "\" and Linux uses "/". This means that this needs to be changed because the software will give a bug.  

Tutorials can be found here: 
Old version tutorial: https://www.youtube.com/watch?v=fSfit87vkrA&feature=youtu.be
New tutorial: coming soon! Hopefully somewhere in January 2021!

*B. How to train your own Deep Neural Network?*
1. Convert the training images: I used for example 804x804px images. In order to resize the images I used ImageJ or Fiji (https://imagej.net/Downloads). I used process => Batch => Convert in order to convert the original 1608x1608px tiff files to a 804x804px jpeg format. 
2. Install LabelImg and draw boxes around the tracks in the images 
https://github.com/tzutalin/labelImg
Then, compress the pictures and .txt files in one .zip file and place this file images.zip in the yolov3 folder on your google Drive. 
3. Execute the Google Colab notebook for some hours 
Aim for 2000-3000 iterations. The iteration speed depends on the size of your images. 
From my experience it seems that from then on, the DNN is overtrained. 
