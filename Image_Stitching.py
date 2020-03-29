import cv2
import numpy as np
import sys
from heapq import nsmallest
import matplotlib.pyplot as plt
import argparse
import os

sift = cv2.xfeatures2d.SIFT_create()
class Image:
    def __init__(self,name, img, keypoints, descriptors):
        self.name = name
        self.img = img
        self.keypoints = keypoints
        self.descriptors = descriptors

def check_matches(descriptors1, descriptors2):
    matches_index = []
    for i in range(descriptors2.shape[0]):
        distances = np.linalg.norm(descriptors1 - descriptors2[i], axis=1)
        two_smallest = nsmallest(2, distances)
        min1, min2 = two_smallest[0], two_smallest[1]
        ratio = min1 / min2
        if ratio > 0.4:
            continue
        else:
            min_index = np.argmin(distances)
            matches_index.append((i, min_index))
    print(len(matches_index))
    ratio = len(matches_index) / descriptors1.shape[0]
    print(ratio)
    return ratio, matches_index

# Source: Hartley and Zisserman, Mutiple View Geometry
def calculate_Homography(pts1, pts2):
    A = []
    for i in range(len(pts1)):
        x1 = pts1[i][0]
        y1 = pts1[i][1]
        x2 = pts2[i][0]
        y2 = pts2[i][1]
        A.append([x1, y1, 1, 0, 0, 0, -1 * x2 * x1, -1 * x2 * y1, -1 * x2])
        A.append([0, 0, 0, x1, y1, 1, -1 * y2 * x1, -1 * y2 * y1, -1 * y2])

    u, s, vh = np.linalg.svd(A)
    # print(vh[-1,:].shape)
    H = vh[-1, :] / vh[-1, -1]
    H = H.reshape(3, 3)
    return H

#Idea and Algorithm Source: Richard Szeliski, Computer Vision: Algorithms and Applications
# Hartley and Zisserman, Multiple View Geometry
def get_RANSAC(src_points, dest_points):
    trials = 500
    column_ones = np.ones(src_points.shape[0])
    column_ones = column_ones.reshape(dest_points.shape[0],1)
    src_points2 = np.append(src_points, column_ones, axis = 1)
    src_points2 = src_points2.T
    dest_points2 = np.append(dest_points, column_ones, axis = 1)
    dest_points2 = dest_points2.T
    final_H = np.ones((3,3))
    inlier_f = int(0.9 * len(src_points))
    for i in range(trials):
        index_rand = np.random.randint(0, len(src_points), size = 4)
        pts1 = src_points[index_rand] # Points which matched in right image
        pts2 = dest_points[index_rand]
        H = calculate_Homography(pts1,pts2)
        c_ones = np.ones(pts1.shape[0])
        c_ones = c_ones.reshape(c_ones.shape[0],1)
        pts1_1 = np.append(pts1, c_ones, axis =1)
        pts1_1 = pts1_1.T
        pts2_2 = np.append(pts2, c_ones, axis =1)
        pts2_2 = pts2_2.T
        dists = []
        for j in range(pts1_1.shape[1]):
            dist = np.linalg.norm(pts2_2[:,j] - np.dot(H, pts1_1[:,j]))
            dists.append(dist)
        dists = np.asarray(dists)
        threshold = np.max(dists)
        distances = []
        for j in range(src_points2.shape[1]):
            src = np.dot(H,src_points2[:,j])
            distance = np.linalg.norm(dest_points2[:,j] - src)
            distances.append(distance)
        distances = np.asarray(distances)
        inliers_src = []
        inliers_dest = []
        inliers = 0
        for j in range(distances.shape[0]):
            if distances[j] < threshold:
                inliers_src.append(src_points[j])
                inliers_dest.append(dest_points[j])
                inliers += 1
        inliers_src = np.asarray(inliers_src)
        inliers_dest = np.asarray(inliers_dest)
        if inliers > inlier_f:
            final_H = calculate_Homography(inliers_src, inliers_dest)
            inlier_f = inliers
            break
    return final_H

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument(
        "--img_path", type=str, default="",
        help="path to the image used for edge detection")
    args = parser.parse_args()
    return args

def read_images(img_path):
    entries = os.listdir(img_path)
    imgs = []

    for entry in entries:
        print(entry)
        img = cv2.imread(img_path+"/"+entry)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img_gray,None)
        imgobj = Image(entry, img, keypoints, descriptors)
        imgs.append(imgobj)
    return imgs

def check_overlap_create_pairs(imgs):
    img_pairs = []
    for i in range(len(imgs)):
        ratio_overlap = []
        img_pair = []
        img_pair.append(imgs[i])
        for j in range(i+1, len(imgs)):
            ratio,_ = check_matches(imgs[i].descriptors, imgs[j].descriptors)
            if ratio == 1.0:
                continue

            elif ratio == 0:
                continue
            else:
                ratio_overlap.append((j,ratio))
        if len(ratio_overlap) != 0:
            max_ratio_element = max(ratio_overlap, key = lambda x: x[1])
            max_ratio_index = max_ratio_element[0]
            print(imgs[max_ratio_index].name)
            img_pair.append(imgs[max_ratio_index])
            img_pairs.append(img_pair)
    print(img_pairs)
    return img_pairs

def stitch_images(imgl,imgr, H):
    dst2 = cv2.warpPerspective(imgr.img, H, (imgr.img.shape[1]+imgl.img.shape[1], imgr.img.shape[0]))
    #dst2[0:imgl.img.shape[0],0:imgl.img.shape[1]] = imgl.img

    for i in range(imgl.img.shape[0]):
        for j in range(imgl.img.shape[1]):
            dst2[i,j] = np.maximum(dst2[i,j], imgl.img[i,j])
   

    return dst2

def create_panorama_in_pairs(imgs):
    if len(imgs) == 1:
        print(imgs[0])
        cv2.imwrite("E:/Computer Vision/data/stitched_image.jpg",imgs[0].img)
        exit(0)
    img_pairs = check_overlap_create_pairs(imgs)
    panoramas = []
    for i,img_pair in enumerate(img_pairs):
        imgr = img_pair[1]
        imgl = img_pair[0]
        _, matches_index = check_matches(imgl.descriptors, imgr.descriptors)
        src_indices = [l[0] for l in matches_index]
        dest_indices = [l[1] for l in matches_index]
        src_points = np.float32([imgr.keypoints[index].pt for index in src_indices])
        dest_points = np.float32([imgl.keypoints[index].pt for index in dest_indices])
        H = get_RANSAC(src_points, dest_points)
        sys.setrecursionlimit(10 ** 6)
        #stitched_image = cv2.warpPerspective(imgr.img, H, (imgr.img.shape[1] + imgl.img.shape[1], imgr.img.shape[0]))
        #stitched_image[0:imgl.img.shape[0], 0:imgl.img.shape[1]] = imgl.img
        stitched_image = stitch_images(imgl, imgr, H)
        cv2.imwrite("E:/stitched_test_image4.jpg", stitched_image)
        print(stitched_image.shape)
        plt.imshow(stitched_image)
        plt.show()
        keypointsp,descriptorsp = sift.detectAndCompute(cv2.cvtColor(stitched_image,cv2.COLOR_BGR2GRAY), None)
        panorama_obj = Image("panorama{}".format(i), stitched_image, keypointsp, descriptorsp)
        panoramas.append(panorama_obj)
    create_panorama_in_pairs(panoramas)

if __name__ == '__main__':

    args = parse_args()
    imgs = read_images(args.img_path)
    if imgs == None:
        print("Read not successful")
    else:
        print("Read successful")
    create_panorama_in_pairs(imgs)
    #print(imgobj.img)
    #plt.imshow(img)
    #plt.show()
    '''
    img1 = cv2.imread("E:/Computer Vision/CSE473573Project_2_Sample_Test/data/nevada3.jpg")
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("E:/Computer Vision/CSE473573Project_2_Sample_Test/data/nevada5.jpg")
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)
    ratio, matches_index = check_matches(descriptors1, descriptors2)
    src_indices = [l[0] for l in matches_index]
    dest_indices = [l[1] for l in matches_index]
    src_points = np.float32([keypoints2[index].pt for index in src_indices])
    dest_points = np.float32([keypoints1[index].pt for index in dest_indices])
    H = get_RANSAC(src_points, dest_points)
    sys.setrecursionlimit(10**6)
    dst2 = cv2.warpPerspective(img2,H,(img1.shape[1] + img2.shape[1],img2.shape[0]))
    dst2[0:img1.shape[0],0:img1.shape[1]] = img1
    dst2 = trim(dst2)
    #cv2.imwrite("E:/stitched_test_image4.jpg", dst2)
    print(dst2.shape)
    plt.imshow(dst2)
    plt.show()
'''


