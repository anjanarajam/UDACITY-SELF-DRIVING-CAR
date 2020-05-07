import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def transform_perspective(threshold_image):
    # Define the four source points
    src = np.array([[200,720], [1150, 720], [750, 480], [550, 480]], np.float32)
    
    # Define the four destination points
    dst = np.array([[300,720], [900, 720], [900, 0], [300, 0]], np.float32)
    
    # Get the transformation matrix by performing perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Get the inverse transformation matrix 
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Get the image size 
    image_size = (threshold_image.shape[1], threshold_image.shape[0])
    
    # Warp the image 
    warped_image = cv2.warpPerspective(threshold_image, M, image_size)
    
    return warped_image, M, Minv