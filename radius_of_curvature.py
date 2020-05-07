import numpy as np

def get_radius_of_curvature(x_pixels):
    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ymeters_per_pixel = 30/720 
    # meters per pixel in x dimension
    xmeters_per_pixel = 3.7/700 
    
    # Get x, y values from image
    y_image_values = np.linspace(0, 719, num=720)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image since 
    # we need curvature closest to the vehicle
    y_max = np.max(y_image_values)
    
    # Get the left and right pixels
    left_x_pixel = x_pixels[0]
    right_x_pixel = x_pixels[1]
    
    # Get the left and right coefficients
    left_x_coeff = np.polyfit(y_image_values * ymeters_per_pixel, left_x_pixel * xmeters_per_pixel, 2)
    right_x_coeff = np.polyfit(y_image_values * ymeters_per_pixel, right_x_pixel * xmeters_per_pixel, 2)
       
    # Calculate radius of curvature 
    left_curvature = ((1 + (2* left_x_coeff[0] * y_max * ymeters_per_pixel + left_x_coeff[1]) ** 2) ** 1.5) / np.absolute(2 *                                    left_x_coeff[0])
    right_curvature = ((1 + (2 * right_x_coeff[0] * y_max * ymeters_per_pixel + right_x_coeff[1]) ** 2) ** 1.5) / np.absolute(2 *                                 right_x_coeff[0])
    
    return (left_curvature, right_curvature)
    