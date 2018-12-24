import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import *
from PIL import Image
%matplotlib notebook

# CONST
PATH = "img/"
MIN_MATCH_COUNT = 10

surf = cv2.xfeatures2d.SURF_create(350)
sift = cv2.xfeatures2d.SIFT_create(350)
orb = cv2.ORB()

bf = cv2.BFMatcher()

index = dict(algorithm = 0, size = 5)
search = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index, search)

def scale_image(input_image_path,
                output_image_path,
                width=None,
                height=None
                ):
    original_image = Image.open(input_image_path)
    w, h = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=w, height=h))
 
    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')
 
    original_image.thumbnail(max_size, Image.ANTIALIAS)
    original_image.save(output_image_path)
 
    scaled_image = Image.open(output_image_path)
    width, height = scaled_image.size
    print('The scaled image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))

class LabelRecognition():
    def __init__(self, detector, descriptor,
                 descriptorComparisonKNN = None, 
                 descriptorComparisonMATCHES = None):
        self.detector = detector
        self.descriptor = descriptor
        self.descriptorComparisonKNN = descriptorComparisonKNN
        self.descriptorComparisonMATCHES = descriptorComparisonMATCHES
    def build(self, trainImg, queryImg_array):
        i = 0
        for queryImg in queryImg_array:
            nameResultImage = 'res/res_' + str(i) + '.jpg'
            i+=1
            # detector and descriptor
            kp1, des1 = self.fDetecctAndDescript(queryImg)
            kp2, des2 = self.fDetecctAndDescript(trainImg)
            
            print("--------------- %d -----------" % i)
            # detector
            '''
            kp1 = self.fDetector(trainImg)
            kp2 = self.fDetector(queryImg)

            # descriptor
            des1 = self.fDescriptor(trainImg, kp1)
            des2 = self.fDescriptor(queryImg, kp2)
            '''

            # descriptorComparison
            matches = self.fDescriptorComparison(des1, des2)

            # search good matches
            good_matches = self.fGoodMatches(matches)

            if len(good_matches)>=MIN_MATCH_COUNT:
                matchesMask = self.fRansac(kp1, kp2, good_matches, [cv2.RANSAC, 10.0])
                print("Matches found enough - %d/%d" % (len(matchesMask),MIN_MATCH_COUNT))
                draw_params = dict(matchColor = (0,255,0),
                                   singlePointColor = None,
                                   matchesMask = matchesMask,
                                   flags = 2)

                img3 = cv2.drawMatches(queryImg,kp1,trainImg,kp2,good_matches,None,**draw_params)
                cv2.imwrite(nameResultImage,img3)
            else:
                print("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT))
                matchesMask = None
        
    def fDetecctAndDescript(self, img):
        kp1, des1 = self.detector.detectAndCompute(img,None)
        return kp1, des1
    
    def fDetector(self, img):
        return self.detector.detect(img, None)
    
    def fDescriptor(self, img, kp):
        f, des = self.descriptor.compute(img, kp)
        return des
    
    def fDescriptorComparison(self, des1, des2):
        if self.descriptorComparisonKNN:
            matches = self.descriptorComparisonKNN.knnMatch(des1,des2,2) 
        elif self.descriptorComparisonMATCHES:
            matches = self.descriptorComparisonMATCHES.match(des1,des2)
        return matches
    
    def fGoodMatches(self, matches):
        good_matches_array = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good_matches_array.append(m)
        return good_matches_array
    
    def fRansac(self, kp1, kp2, goodMatches, param):
        queryIdx = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        trainIdx = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(queryIdx, trainIdx, param[0], param[1])
        matchesMask = mask.ravel().tolist()

        return matchesMask

#BODY
	# Opened image
path = os.listdir("img/")

imageArray = []
for i in path:
    pathImage = str(i)
    if pathImage.find('jpg') == -1:
        print('ERROR!! NameImage: ', pathImage)
    else:
        imageArray.append(cv2.imread(PATH + pathImage))
print(len(imageArray))

	#Search
search_image = cv2.imread('search_small_new.jpg')
LR = LabelRecognition(surf, surf, descriptorComparisonKNN=flann)
LR.build(search_image, imageArray)
