## Read this!

[AI-Track-tive: free software for automated recognition and counting of surface semi-tracks using computer vision (Artificial Intelligence)](https://gchron.copernicus.org/preprints/gchron-2020-32/)

![logo](/logo-01.png)

## A. Obtaining the freeware AI-Track-tive software
## Download Deep Neural Networks 
[Clic here to download the Deep Neural Network for apatite](https://drive.google.com/file/d/1BxlqKlNSEpXzhBCoR_-P_cEwyhxSw833/view?usp=sharing)

[Click here to download the Deep Neural Network for mica](https://drive.google.com/file/d/1FHB0UTmEpTOI_QKx_0Boa3VgudgOfJS9/view?usp=sharing)

### Windows users
The easiest option to run the code is by downloading the folder from Github and double-click the .exe file. 

You can find a folder with the executable (application) and also with the source code (.py). 

If you would like to view the Python code and perhaps do some changes, you can download the source code (.py) file. I did the programming-testing-debugging using the highly recommendable [Spyder IDE](https://docs.spyder-ide.org/current/index.html). 


### Mac-OS users
You have to open and execute the source code (.py) in a free Python-based IDE, for example [Spyder](https://www.spyder-ide.org/) or [Pycharm](https://www.jetbrains.com/pycharm/download/#section=windows). 

Before you can use it, you need to install some Python modules using the command terminal:
- [opencv](https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html)
- [pyqt](https://pythonbasics.org/install-pyqt/)
- [opencv-python-headless](https://pypi.org/project/opencv-python-headless/)
  
### Linux users
You have to open the source code (.py) in a free Python-based IDE, for example Thonny. 
I had no problems using the software on a Raspberry Pi 4B with 4GB RAM (€50) using Linux, after installing the following Python packages using the terminal. 

First of all: install pip for linux 
```apt install python3-pip```

To install the pandas:
```pip install pandas```

To install the opencv module
``` pip install opencv-python ```

Using my ultra-cheap Raspberry Pi I noticed that the program didn't want to initiate because opencv2 was not entirely well installed. I found some help on my favorite website (stackoverflow) and found that it's possible to solve this error while executing the following code in the Linux command shell: 
```pip3 install opencv-python
sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
sudo apt-get install libqtgui4
sudo apt-get install libqt4-test
```
During debugging I noticed that Windows uses typically " \ " and Linux uses "/". This means that this needs to be changed because the software will give a bug.  

## B. How to use this program once it's downloadeded on my pc? 
[Old version tutorial](https://www.youtube.com/watch?v=fSfit87vkrA&feature=youtu.be)

New version tutorial: coming soon! Hopefully somewhere in January 2021!

## C. How to train your own Deep Neural Network (DNN)?

- Convert the training images: I used for example 804x804px images. In order to resize the images I used [ImageJ or Fiji](https://imagej.net/Downloads). I used process => Batch => Convert in order to convert the original 1608x1608px tiff files to a 804x804px jpeg format.

- Install [LabelImg](https://github.com/tzutalin/labelImg) and draw boxes around the tracks in the images 

- Compress the labelled files and their .txt files in one .zip file and place this file images.zip in the yolov3 folder on your google Drive. 

- Execute the Google Colab notebook for some hours. Aim for 2000-3000 iterations. The iteration speed depends on the size of your images. From my experience it seems that from then on, the DNN is overtrained. 

### If you have further questions or suggestions for this tutorial: e-mail me please Simon.Nachtergaele@UGent.be
