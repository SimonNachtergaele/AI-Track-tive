# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:01:57 2020

@author: smanacht
"""
import os
import math

os.chdir('E:/Machine learning FT/Training/Yolov3/Trainingsset for AI-Track-tive Colab/804 px datasets ed/13082020 25 labelled ones')

# Ed old 

mica_track_density = []
apatite_track_density = []

ed_list_old = ['M12_FCT-G4_EDr_crop_XY002_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY003_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY005_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY006_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY007_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY012_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY017_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY033_Z4_RGB.txt',
            'M12_FCT-G4_EDr_crop_XY054_Z4_RGB.txt',
            'M20_DUR-G2_EDr-Uc_crop_XY01_Z4_RGB.txt',
            'M20_DUR-G2_EDr-Uc_crop_XY02_Z4_RGB.txt',
            'M20_DUR-G2_EDr-Uc_crop_XY03_Z4_RGB.txt',
            'M20_DUR-G2_EDr-Uc_crop_XY04_Z4_RGB.txt',
            'M20_DUR-G2_EDr-Uc_crop_XY05_Z4_RGB.txt',
            'M20_DUR-G2_EDr-Uc_crop_XY20_Z4_RGB.txt',
            'Seq0000_XY001_Z4_RGB.txt',
            'Seq0000_XY002_Z4_RGB.txt',
            'Seq0000_XY003_Z4_RGB.txt',
            'Seq0000_XY004_Z3_RGB.txt',
            'Seq0000_XY005_Z4_RGB.txt',
            'Seq0000_XY006_Z4_RGB.txt',
            'Seq0000_XY007_Z4_RGB.txt',
            'Seq0000_XY008_Z4_RGB.txt',
            'Seq0000_XY009_Z4_RGB.txt',
            'Seq0000_XY010_Z4_RGB.txt']
sum_old = 0

for ed in ed_list_old:
    count=0
    for regel in open(ed, 'r'):
        #print(regel)
        
        count+=1
        
    sum_old+=count
    print(str(ed)+str(' ')+str(count))
    mica_track_density.append(math.log(count/(117.5*117.5*10**(-8)),10))
    print(str(ed)+str(' ')+str(math.log(count/(117.5*117.5*10**(-8)),10)))
    
print('sum of old list is '+str(sum_old))

print('\n')



# 2. ED new

ed_list_new = ['F115 ed flipped_XY02_Z5.txt',
                'F115 ed flipped_XY03_Z5.txt',
                'F115 ed flipped_XY04_Z5.txt',
                'F115 ed flipped_XY05_Z5.txt',
                'F115 ed flipped_XY07_Z5.txt',
                'F115 ed flipped_XY08_Z5.txt',
                'F115 ed flipped_XY09_Z5.txt',
                'F115 ed flipped_XY10_Z5.txt',
                'F115 ed flipped_XY11_Z5.txt',
                'F115 ed flipped_XY12_Z5.txt',
                'F115 ed flipped_XY13_Z5.txt',
                'F115 ed flipped_XY14_Z5.txt',
                'F115 ed flipped_XY15_Z5.txt',
                'F115 ed flipped_XY16_Z5.txt',
                'F115 ed flipped_XY17_Z5.txt',
                'F115 ed flipped_XY18_Z5.txt',
                'F115 ed flipped_XY19_Z5.txt',
                'F115 ed flipped_XY20_Z5.txt',
                'F115 ed flipped_XY21_Z5.txt',
                'F115 ed flipped_XY22_Z5.txt',
                'F115 ed flipped_XY23_Z5.txt',
                'F115 ed flipped_XY24_Z5.txt',
                'F115 ed flipped_XY25_Z5.txt',
                'F115 ed flipped_XY26_Z5.txt',
                'F115 ed flipped_XY27_Z5.txt']
               
               
sum_new = 0
grains=0   
tr_density=list()            
for ed in ed_list_new:
    f = open(ed, 'r')
    grains+=1
    count=0
    for regel in open(ed, 'r'):
        #print(regel)
        count+=1
    
    t = 100000000*float(count) / (117.5*117.5)
    print(t)
    tr_density.append(t)
    
    mica_track_density.append(math.log(count/(117.5*117.5*10**(-8)),10))
    print(str(ed)+str(' ')+str(math.log(count/(117.5*117.5*10**(-8)),10)))
    
    sum_new+=count
    print(str(ed)+str(' ')+str(count))
          

print('sum of new ed list is '+str(sum_new)+str(' for number of grains ')+str(grains))
               
print('\n')


# Import matplotlib 
import matplotlib.pyplot as plt

# Create a figure
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 3))
  
print('mica track density')
print(mica_track_density)
print(len(mica_track_density))

# Histogram
n, bins, patches = axes[0].hist(mica_track_density,10)
axes[0].set_xlim(right=7)
axes[0].set_xlim(left=5)
axes[0].set_xlabel('log(track density)')
axes[0].set_ylabel('count')
axes[0].set_ylim(bottom=0,top=10)
axes[0].set_title('apatite')

               
# 3. Apatite

os.chdir('E:/Machine learning FT/Training/Yolov3/Trainingsset for AI-Track-tive Colab/804 px trainingsset apatite 20 DUR 5 BC 25 F115')
              
ap_list_old = ['BC-04_Ap_crop_XY05_Z04_RGB.txt',
            'BC-04_Ap_crop_XY08_Z04_RGB.txt',
            'BC-04_Ap_crop_XY09_Z04_RGB.txt',
            'BC-04_Ap_crop_XY11_Z04_RGB.txt',
            'BC-04_Ap_crop_XY13_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY01_Z03_RGB.txt',
            'DUR-G2_Ap-Lc_XY02_Z03_RGB.txt',
            'DUR-G2_Ap-Lc_XY03_Z03_RGB.txt',
            'DUR-G2_Ap-Lc_XY04_Z03_RGB.txt',
            'DUR-G2_Ap-Lc_XY05_Z03_RGB.txt',
            'DUR-G2_Ap-Lc_XY06_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY07_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY08_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY09_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY10_Z04_RGB.txt']
sum_old = 0

for ap in ap_list_old:
    f = open(ap, 'r')
    
    count=0
    for regel in open(ap, 'r'):
        #print(regel)
        count+=1
        
    sum_old+=count
    print(str(ap)+str(count))
    apatite_track_density.append(math.log(count/(117.5*117.5*10**(-8)),10))
    print(str(ap)+str(' ')+str(math.log(count/(117.5*117.5*10**(-8)),10)))
    
print('apatite sum of old list is '+str(sum_old))

print('\n')

# 4. Apatite new 

ap_list_new = ['DUR-G2_Ap-Lc_XY11_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY12_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY13_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY14_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY15_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY16_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY17_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY18_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY19_Z04_RGB.txt',
            'DUR-G2_Ap-Lc_XY20_Z04_RGB.txt',
            'img_XY02_Z06.txt',
            'img_XY06_Z05.txt',
            'img_XY12_Z05.txt',
            'img_XY13_Z05.txt',
            'img_XY14_Z05.txt',
            'img_XY15_Z05.txt',
            'img_XY17_Z05.txt',
            'img_XY19_Z05.txt',
            'img_XY20_Z05.txt',
            'img_XY21_Z05.txt',
            'img_XY22_Z05.txt',
            'img_XY23_Z05.txt',
            'img_XY24_Z04.txt',
            'img_XY25_Z04.txt',
            'img_XY26_Z04.txt',
            'img_XY27_Z04.txt',
            'img_XY28_Z06.txt',
            'img_XY29_Z06.txt',
            'img_XY30_Z06.txt',
            'img_XY31_Z06.txt',
            'img_XY32_Z06.txt',
            'img_XY33_Z06.txt',
            'img_XY34_Z06.txt',
            'img_XY35_Z06.txt',
            'img_XY36_Z06.txt']
               
               
sum_new = 0
               
for ap in ap_list_new:
    f = open(ap, 'r')
    
    count=0
    for regel in open(ap, 'r'):
        #print(regel)
        count+=1
        
    sum_new+=count
    apatite_track_density.append(math.log(count/(117.5*117.5*10**(-8)),10))
    print(str(ap)+str(' ')+str(math.log(count/(117.5*117.5*10**(-8)),10)))
    
print('apatite sum of new list is '+str(sum_new))
               
print('\n')
              
print('apatite track density')
print(apatite_track_density)
print(len(apatite_track_density))

# Histogram
n, bins, patches = axes[1].hist(apatite_track_density,10)
axes[1].set_xlim(right=7)
axes[1].set_xlim(left=5)
axes[1].set_xlabel('log(track density)')
axes[1].set_ylabel('count')
axes[1].set_ylim(bottom=0,top=10)
axes[1].set_title('muscovite')
               
os.chdir(r'E:/Machine learning FT/Manuscript/Figures manuscript')
fig.tight_layout()
plt.savefig('track_density_training_images.pdf',dpi=900)
plt.savefig('track_density_training_images.png',dpi=900)
plt.show()   
               
               