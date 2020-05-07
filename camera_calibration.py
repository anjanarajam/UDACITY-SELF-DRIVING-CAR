import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#Read the set of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

#3D points in real space world of the undistorted image
objpoints = []
#2D points in image plane of the distorted image
imgpoints = []

def calibrate_camera():
    """ This functions calibrates the camera to get calibration matrix and distortion coefficients
    by running through a series of images"""
    for frame in images:
        # Read the image from the list of distorted images
        img = mpimg.imread(frame)
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Prepare object points for (x, y, z) like (0, 0, 0), (1, 0, 0)...
        prepared_objpoints = np.zeros((9*6, 3), np.float32)
        #Generate co=ordinate for the given grid size, reshapes to 2x2 matrix for x and y
        prepared_objpoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        #Detect chessboard corners of the distorted image in grayscale image
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        #Append object and image points
        if ret == True:
            #Append the corners to the image points 
            imgpoints.append(corners)
            objpoints.append(prepared_objpoints)
            
            #Draw the detected corners on an image
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        
    #Calibrate the camera to obtain calibration matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
    return mtx, dist

def undistort_image(dist_image, mtx, dist):
        """ This function undistorts the image"""
        undist_image = cv2.undistort(dist_image, mtx, dist, None, mtx)
        
        return undist_image 
    
def plot_undistorted_image(dist_image, undist_image):
        # Plot the original distorted and undistorted image
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(dist_image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist_image)
        ax2.set_title('Undistorted Image', fontsize=50)
        
