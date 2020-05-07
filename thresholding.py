import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def hls_threshold(image, thresh=(0, 255)):
    """This function applies threshold to saturation colour space to detect the lane line"""
        
    # Convert the image to HLS colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # Separate the 'S' channel
    S = hls[:,:,2]
    
    #  Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    return binary_output

def magnitude_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    """This function applies threshold to magnitude gradient to detect the lane line"""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)

    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # Return this mask as your binary_output image
    return binary_output

def combine_binary(image):
    #Get the binary image of hls threshold
    hls_binary = hls_threshold(image, thresh=(90, 255))
    
    #Get the binary image of magnitude threshold
    mag_binary = magnitude_threshold(image, sobel_kernel=3, thresh=(30, 100))
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(mag_binary), mag_binary, hls_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(mag_binary)
    combined_binary[(hls_binary == 1) | (mag_binary == 1)] = 1
    
    return combined_binary 
    