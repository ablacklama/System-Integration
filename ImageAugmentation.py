import os
import csv
import sklearn
from sklearn.utils import shuffle
from IPython.display import Image
from IPython.display import display
import cv2
import math
import random
import numpy as np


### Data exploration visualization code goes here.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


with open('TrafficLightLabels.csv', mode='r') as f:
    fread = csv.reader(f)
    next(fread, f)
    TL_labels = [rows[1] for rows in fread]

images = []
labels = []

imgsr = []
lblsr = []
imgsy = []
lblsy = []
imgsg = []
lblsg = []
imgsn = []
lblsn = []


def readImagesLabels(pathcsv, fileDir):

    lines_file = []

    with open(pathcsv) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines_file.append(line)

    imgs = []
    lbls = []
    # Load each row into line
    for line in lines_file:
    # Load the images and labels
        filename = line[0]
        #print ("filename:", filename)
        local_path = fileDir + filename
        #print ("filename with full path:", local_path)
        image = cv2.imread(local_path)
        #print ("shape of image:", image.shape)
        imgs.append(image)
        label = line[1]
        #print ("label:", label)
        lbls.append(label)
		
    return imgs, lbls
	
redCsvPath = './data/sim-slack/Red/redimgs.csv'
redFileDir = './data/sim-slack/Red/'

yellowCsvPath = './data/sim-slack/Yellow/yellowimgs.csv'
yellowFileDir = './data/sim-slack/Yellow/'

greenCsvPath = './data/sim-slack/Green/greenimgs.csv'
greenFileDir = './data/sim-slack/Green/'

noneCsvPath = './data/sim-slack/None/noneimgs.csv'
noneFileDir = './data/sim-slack/None/'

imgsr, lblsr = readImagesLabels(redCsvPath, redFileDir)
#print ("Shape of image:", imgsr[0].shape)
#print ("filename with full path:", local_path)
#print ("Total # of red images:", len(imgsr))
#print ("Total # of red labels:", len(lblsr))

imgsg, lblsg = readImagesLabels(greenCsvPath, greenFileDir)
#print ("Total # of green images:", len(imgsg))
#print ("Total # of green labels:", len(lblsg))

imgsy, lblsy = readImagesLabels(yellowCsvPath, yellowFileDir)
#print ("Total # of yellow images:", len(imgsy))
#print ("Total # of yellow labels:", len(lblsy))

imgsn, lblsn = readImagesLabels(noneCsvPath, noneFileDir)
#print ("Total # of none images:", len(imgsn))
#print ("Total # of none labels:", len(lblsn))

images = imgsr+imgsy+imgsg+imgsn
labels = lblsr+lblsy+lblsg+lblsn

#print ("Total # of images:", len(images))
#print ("Total # of labels:", len(labels))

    
# Converting images & labels to numpy arrays
images = np.array(images)
labels = np.array(labels)
labels = labels.astype(int)
#labels = np.dtype(int)

print ("Total # of images:", len(images))
print ("Total # of labels:", len(labels))

print ("Original shape of image array:", images.shape)
# Print an example image
#plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#plt.show()

print ("Original Image Shape:", images[0].shape)

#Image augmentation functions
def transform_image(image,ang_range,shear_range,trans_range):

    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = image.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    image = cv2.warpAffine(image,Rot_M,(cols,rows))
    image = cv2.warpAffine(image,Trans_M,(cols,rows))
    image = cv2.warpAffine(image,shear_M,(cols,rows))
    
    return image
	
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

# n_add is total number of samples in each category after augmenting. Example: If class 0 had 20 samples, n_add=250  
# will give 250 samples when the gen_new_images function is called (adds 250-20 = 230 samples in this example)
def gen_new_images(X_train,y_train,n_add,ang_range,shear_range,trans_range):
   
    ## checking that the inputs are the correct lengths
    assert X_train.shape[0] == len(y_train)
    # Number of classes: 4 (Red, Yellow, Green, None)
    n_class = len(np.unique(y_train))
    print("Number of classes:", n_class)
    X_arr = []
    Y_arr = []
    #print(y_train.dtype)	
    n_samples = np.bincount(y_train)
    print("Samples in each class:", n_samples)
	
    for i in range(n_class):
        # Number of samples in each class

        if n_samples[i] < n_add:
            print ("Adding %d samples for class %d" %(n_add-n_samples[i], i))
            for i_n in range(n_add - n_samples[i]):
                img_trf = transform_image(X_train[i_n],ang_range,shear_range,trans_range)
                #Add random shadows to a 3rd of the images 
                ind_flip = np.random.randint(2)				
                if ind_flip==0:
                   img_trf = add_random_shadow(img_trf)				
                X_arr.append(img_trf)
                Y_arr.append(i)
#                print ("Number of images in class %d:%f" %(i, X_arr[0])) 
           
    X_arr = np.array(X_arr,dtype = np.float32())
    Y_arr = np.array(Y_arr,dtype = np.int32())
   
    return X_arr,Y_arr
	
#Save a copy of the original images
X_train = images
X_train_gold = X_train
#X_test_gold = X_test
#X_valid_gold = X_valid

#Minimum number of samples for each class. If a class does not contain this number, it will add extra
#samples so it meets this minimum number 
augment_num = 50

y_train = labels
	
# Use the next three lines in the code if normalization is needed. Error with mean function.
# Normalize all original training images
#X_train_norm = (X_train - np.mean(X_train)) / (np.max(X_train) - np.min(X_train))
#X_test_norm = (X_test - X_test.mean()) / (np.max(X_test) - np.min(X_test))
#X_valid_norm = (X_valid - X_valid.mean()) / (np.max(X_valid) - np.min(X_valid))

# Assign normalized data back to original variables for ease of use of naming 
#X_train = X_train_norm
#X_test = X_test_norm
#X_valid = X_valid_norm

print('Color Image shape before augmentation:', X_train.shape)
#Generate new images using the original (not normalized) images
X_train_aug,y_train_aug = gen_new_images(X_train_gold,y_train,augment_num,30,5,5)
## combine the generated images and the original training set
X_train = np.append(X_train, X_train_aug, axis=0)
y_train = np.append(y_train, y_train_aug, axis=0)
print('Color Image shape after augmentation:', X_train.shape)
