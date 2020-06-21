#-*- coding: utf-8 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 19/06/2020
#LABORATORY 2: CONVOLUCIÓN 2D (Test)

# IMPORTS
import numpy as np
import time

# CONSTANTS
# GLOBAL VARIABLES
# CLASSES
class Kernel:
    """ Kernel Object
    
    Parameters:
    ----------
    matrix : numpy matrix
        Input kernel matrix. 

   """
    def __init__(self, matrix):
        self.__matrix = matrix
        self.__height = 0
        self.__width = 0

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def set_matrix(self, matrix):
        self.__matrix = matrix

    @property
    def height(self):
        return self.__matrix.shape[0]

    @height.setter
    def set_heigth(self, height):
        self.__height = height

    @property
    def width(self):
        return self.__matrix.shape[1]

    @width.setter
    def set_width(self, width):
        self.__width = width

# FUNCTIONS
def apply_filter(image, kernel):
    """ Check if the image file exists.
    
    Returns a matrix with the filter applied on the image.
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix

    kernel : kernel object
        kernel

    Returns:
    -------
    matrix: numpy matrix
        matrix with the filter applied on the image.
    """
    border_size = get_border_size(kernel)
    bordered_image = add_border_to_image(image, border_size)
    convolution_image = create_image(get_image_height(image), get_image_width(image))
    horizontal_range = range(border_size, get_image_height(bordered_image) - border_size)
    vertical_range = range(border_size, get_image_width(bordered_image) - border_size)

    for i in horizontal_range:
        for j in vertical_range:
            convolution_image[i - border_size, j - border_size] = convolution_2(
                i, j, bordered_image, kernel, border_size)
    return convolution_image

def get_border_size(kernel):
    """ Get the size of the border to apply to the image.
    
    Returns image border based on kernel size.
    
    Parameters:
    ----------
    kernel : kernel object
        kernel

    Returns:
    -------
    border_size : int
        image border based on kernel size.
    """
    return int((kernel.height - 1) / 2)

def add_border_to_image(image, border_size):
    """ Add a border to the image depending on the width of the kernel.
    
    Returns original image matrix with borders
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix

    border_size : int
        image border based on kernel size

    Returns:
    -------
    matrix : numpy matrix
        original image matrix with borders
    """
    return np.pad(image, pad_width=border_size, mode='constant', constant_values=0)

def create_image(height, width):
    """ Create an image depending on the height and width of the image.
    
    Returns the matrix of the image started at 0.
    
    Parameters:
    ----------
    heigth : int
       matrix height  

    width : int
        matrix width

    Returns:
    -------
    matrix: numpy matrix
        the matrix of the image started at 0.
    """
    if height > 0 and width > 0:
        return np.zeros((height, width))

def get_image_height(image):
    """Get the height in pixels of an image.
    
    Returns the number of pixels corresponding to the image height.
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix 

    Returns:
    -------
    height: int
        The number of pixels corresponding to the image height.
    """
    return image.shape[0]

def get_image_width(image):
    """Get the width in pixels of an image.
    
    Returns the number of pixels corresponding to the image width.
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix 

    Returns:
    -------
    width: int
        The number of pixels corresponding to the image width.
    """
    return image.shape[1]

def convolution(i, j, image, kernel, border_size):
    """ Perform the convolution between a kernel and an image.
    
    Returns the pixel corresponding to the result of the convolution
    
    Parameters:
    ----------
    i : int
        x position of the image matrix

    j : int
        y position of the image matrix

    image : numpy matrix
        image matrix

    kernel : kernel object
        kernel

    border_size : int
        image border based on kernel size

    Returns:
    -------
    pixel : int
        pixel corresponding to the result of the convolution
    """
    pixel = 0
    move_down = - border_size
    for k in range(0, kernel.height):
        move_right = - border_size
        for l in range(0, kernel.width):
            pixel = pixel + image[i + move_down,
                                  j + move_right] * kernel.matrix[k, l]
            move_right = move_right + 1
        move_down = move_down + 1
    return pixel


def convolution_2(i, j, image, kernel, border_size):
    """ (Optimized version of convolution) Perform the convolution between a kernel and an image.
    
    Returns the pixel corresponding to the result of the convolution
    
    Parameters:
    ----------
    i : int
        x position of the image matrix

    j : int
        y position of the image matrix
        
    image : numpy matrix
        image matrix

    kernel : kernel object
        kernel

    border_size : int
        image border based on kernel size

    Returns:
    -------
    pixel : int
        pixel corresponding to the result of the convolution
    """
    return np.multiply(image[i - border_size:i + border_size + 1, j - border_size:j + border_size + 1],
                       kernel.matrix[:,:]).sum()


def is_valid_kernel(kernel):
    """ Check if the kernel is valid.

    Returns True if it's valid, False if it's not.

    Parameters:
    ----------
    kernel : kernel object
        kernel

    Returns:
    -------
    Status: boolean
        True if it's valid, False if it's not.
    """ 
    if is_number_odd(kernel.height) and (kernel.height == kernel.width):
        return True
    else:
        print("El kernel debe ser cuadrado y de dimensiones impares: 1x1, 3x3, 5x5, 7x7...")
        return False


def is_number_odd(number):
    """ Check if a number is odd.
    
    Returns True if it's odd, False if it's not.
    
    Parameters:
    ----------
    number : int
        number to verify 

    Returns:
    -------
    Status: boolean
        True if it's odd, False if it's not.
    """
    if (number % 2) != 0:
        return True
    else:
        return False

def is_valid_image_matrix(image):
    """ Check if the image size is valid.
    
    Returns True if it's valid, False if it's not.
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix

    Returns:
    -------
    Status: boolean
        True if it's valid, False if it's not.
    """
    if image.size > 0:
        return True
    else:
        return False

def main():
    start_time = time.time()
    laplacian_filter = Kernel(np.matrix([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]]))

    gaussian_filter = Kernel(np.matrix([[1, 4, 6, 4, 1],
                                        [4, 16, 24, 16, 4],
                                        [6, 24, 36, 24, 6],
                                        [4, 16, 24, 16, 4],
                                        [1, 4, 6, 4, 1]])*1/256)

    edge_detection_filter = Kernel(np.matrix([[1, 2, 0, -2, -1],
                                              [1, 2, 0, -2, -1],
                                              [1, 2, 0, -2, -1],
                                              [1, 2, 0, -2, -1],
                                              [1, 2, 0, -2, -1]]))

    #Just modify this line to change the image to test.
    test_image = np.matrix([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]])

    #Just modify this line to change the kernel to test.
    test_kernel = laplacian_filter

    if is_valid_image_matrix(test_image) and is_valid_kernel(test_kernel):
        
        border_size = get_border_size(test_kernel)
        test_image_bordered = add_border_to_image(test_image, border_size)
        first_pixel_pos_x = range(border_size, get_image_height(test_image_bordered) - border_size)[0]
        first_pixel_pos_y = range(border_size, get_image_width(test_image_bordered) - border_size)[0]

        print("\n---- Imagen de prueba. ----")
        print(test_image)
        print("\n---- Kernel de prueba. ----")
        print(test_kernel.matrix)
        print("\n----- Iniciando el proceso de convolución: -----")
        print("\n---- Se agregan bordes a la imagen según el kernel. ----")
        print(test_image_bordered)
        print("\n---- Para cada pixel en la imagen, se selecciona una matriz adyacente")
        print("---- de iguales dimensiones que el kernel.\n----(Por la extensión se usa sólo el primer pixel como ejemplo)")
        print("Sub-matriz del pixel:")
        print(test_image_bordered[first_pixel_pos_x - border_size:first_pixel_pos_x + border_size + 1, first_pixel_pos_y - border_size: first_pixel_pos_y + border_size + 1])
        print("\nkernel: ")
        print(test_kernel.matrix[:,:])
        print("\n---- Se multiplica cada elemento del kernel por la matriz")
        print(np.multiply(test_image_bordered[first_pixel_pos_x - border_size:first_pixel_pos_x + border_size + 1, first_pixel_pos_y - border_size: first_pixel_pos_y + border_size + 1]
                        ,test_kernel.matrix[:,:]))
        print("\n---- Se suma cada valor de la matriz resultante, el resultado es el primer pixel de la nueva matriz:")
        print(np.multiply(test_image_bordered[first_pixel_pos_x - border_size:first_pixel_pos_x + border_size +
                                            1, first_pixel_pos_y - border_size: first_pixel_pos_y + border_size + 1], test_kernel.matrix[:, :]).sum())
        print("\n---- Resultado de la convolución. ----")
        print(apply_filter(test_image, test_kernel))
    else:
        print("\nError: La imagen ingresada es invalida. La matriz de la imagen no debe ser vacia.")
    print("\n--- Tiempo de ejecución: %s seconds ---" % (time.time() - start_time))
    

main()
