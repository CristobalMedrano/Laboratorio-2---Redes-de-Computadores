#-*- coding: utf-8 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 19/06/2020
#LABORATORY 2: CONVOLUCIÓN 2D

# IMPORTS
import os.path
import numpy as np
import matplotlib.pyplot as plt
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
def exists_image_file(filename):
    """ Check if the image file exists.
    
    Returns True if it's valid, False if it's not.
    
    Parameters:
    ----------
    filename : string
        Input image filename. 

    Returns:
    -------
    Status: boolean
        True if it's valid, False if it's not.
   """
    if os.path.exists(filename):
        return True
    else:
        print(f"El archivo de imagen {filename} no existe.")
        return False


def read_image(filename):
    """ Read an image using matplotlib. (if the image is in rgb, convert it to grayscale)
    
    Returns an image numpy matrix.
    
    Parameters:
    ----------
    filename : string
        Input image filename. 

    Returns:
    -------
    matrix: numpy matrix
        Image matrix
    """
    image = plt.imread(filename)
    if image.ndim > 2:
        return convert_image_to_gray_scale(image)
    else:
        return image

def save_image(filename, matrix):
    """ Save the image to a file.
    
    Parameters:
    ----------
    filename : string
        image filename.
    
    matrix : numpy matrix
        matrix image.

    """
    new_filename = "salida_"+filename
    plt.imsave(f"{new_filename}", matrix, cmap="gray")
    print(f"-- Archivo: '{new_filename}' ¡generado exitosamente!.")

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

def convert_image_to_gray_scale(image):
    """ Convert a grayscale image.
    
    Returns the grayscale image.
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix 

    Returns:
    -------
    matrix: numpy matrix
        The grayscale image matrix.
    """
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

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
            convolution_image[i - border_size, j - border_size] = convolution_2(i, j, bordered_image, kernel, border_size)
    return convolution_image

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
            pixel = pixel + image[i + move_down, j + move_right] * kernel.matrix[k, l]
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
    return np.multiply( image[i - border_size:i + border_size + 1, j - border_size:j + border_size + 1],
                        kernel.matrix[:,:]).sum()

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

def normalize_image(image):
    """ Normalizes the values of a matrix between 0 and 1.
    
    Returns the normalized image    
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix

    Returns:
    -------
    matrix: numpy matrix
        Normalized image
    """
    min_value = np.min(image)
    max_value = np.max(image)
    return (image - min_value) / (max_value - min_value)

def plot_image(image, label):
    """ Show the resulting images.
    
    Parameters:
    ----------
    image : numpy matrix
        image matrix

    label : string
        title of plot
    """
    plt.figure(label)
    plt.imshow(image, cmap="gray")
    plt.xlabel("Ancho (px)")
    plt.ylabel("Alto (px)")
    plt.title(label)

def plot_fft2d_image(fft, label):
    """ Show the resulting fourier transform in 2d.
    
    Parameters:
    ----------
    fft : discrete fourier transform in 2d

    label : string
        title of plot
    """
    plt.figure("Transformada de Fourier en 2D - "+label)
    plt.imshow(np.log(np.abs(fft)))
    plt.suptitle("Transformada de Fourier en 2D.", fontsize=12)
    plt.title(label, fontsize=10)
    plt.colorbar()


def show_start_time_message(label):
    """ Show start message.
    
    Parameters:
    ----------
    label : string
        label 
    """
    print(f"- Iniciando convolución usando '{label}'.")

def show_end_time_message(current_time, label):
    """ Show end time message.

    Parameters:
    ----------
    current_time : float
        current time
    label : string
        label
    """
    print(f"-- Convolución usando '{label}' terminada en {format((time.time() - current_time), '.3f')} segundos. --\n")

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

# MAIN
def main():
    """ Main function of program """
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
    image_in_jgp = "lena.jpg"
    image_in_bmp = "lena512.bmp"

    #filename = image_in_bmp
    filename = input("Write the name of the image to read: ")
    print("Laboratorio 2 - Convolución 2D")

    #The input files are verified to be correct
    if exists_image_file(filename) and is_valid_kernel(laplacian_filter) and is_valid_kernel(gaussian_filter) and is_valid_kernel(edge_detection_filter):
        
        #The image is read.
        original_image = read_image(filename)
        if is_valid_image_matrix(original_image):

            current_time = time.time()
            show_start_time_message('filtro laplaciano')

            #Filter is applied to the image
            laplacian_image = apply_filter(original_image, laplacian_filter)
            show_end_time_message(current_time, 'filtro laplaciano')

            current_time = time.time()
            show_start_time_message('filtro gaussiano')

            #Filter is applied to the image
            smoothed_image = apply_filter(original_image, gaussian_filter)
            show_end_time_message(current_time, 'filtro gaussiano')
            
            current_time = time.time()
            show_start_time_message('filtro detector de bordes')

            #Filter is applied to the image
            borders_image = apply_filter(original_image, edge_detection_filter)
            show_end_time_message(current_time, 'filtro detector de bordes')

            #Image is normalized
            normalized_laplacian_image = normalize_image(laplacian_image)
            #Image is normalized
            normalized_smoothed_image = normalize_image(smoothed_image)
            #Image is normalized
            normalized_borders_image = normalize_image(borders_image)

            print("- Guardando los resultados obtenidos.")
            #The obtained image is saved
            save_image("filtro_laplaciano_"+filename, normalized_laplacian_image)
            #The obtained image is saved
            save_image("filtro_gaussiano_"+filename, normalized_smoothed_image)
            #The obtained image is saved
            save_image("filtro_detector_bordes_"+filename, normalized_borders_image)
            
            #Fourier transform is calculated in 2 dimensions
            fft_original_image = np.fft.fft2(original_image)
            #Fourier transform is calculated in 2 dimensions
            fft_laplacian_image = np.fft.fft2(normalized_laplacian_image)
            #Fourier transform is calculated in 2 dimensions
            fft_smoothed_image = np.fft.fft2(normalized_smoothed_image)
            #Fourier transform is calculated in 2 dimensions
            fft_borders_image = np.fft.fft2(normalized_borders_image)

            print("\n- Generando Imagenes resultantes...")
            #The resulting image is displayed
            plot_image(original_image, f"Imagen Original {filename}")
            #The resulting image is displayed
            plot_image(normalized_laplacian_image, f"Filtro 'Laplaciano' aplicado en {filename}")
            #The resulting image is displayed
            plot_image(normalized_smoothed_image, f"Filtro 'Suavizado Gaussiano' aplicado en {filename}")
            #The resulting image is displayed
            plot_image(normalized_borders_image, f"Filtro 'Detector de Bordes' aplicado en {filename}")
            
            print("\n- Generando gráficos de la transformada de fourier...")
            plot_fft2d_image(fft_original_image, f"Imagen Original ({filename})")
            plot_fft2d_image(fft_laplacian_image, f"Filtro 'Laplaciano' ({filename})")
            plot_fft2d_image(fft_smoothed_image, f"Filtro 'Suavizado Gaussiano' ({filename})")
            plot_fft2d_image(fft_borders_image, f"Filtro 'Detector de Bordes' ({filename})")
            print("-- Mostrando gráficos obtenidos. --")
            #Fourier transforms shown
            plt.show()
            print("--- Gráficos cerrados. ---\n")

    print("- Finalizando programa. -")
    print("--- Programa ejecutado durante %.3f segundos ---" % (time.time() - start_time))

# MAIN
main()

# REFERENCES
#https://pypi.org/project/imageio/
#https://qastack.mx/stats/70801/how-to-normalize-data-to-0-1-range
#https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-9.php
#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
#https://numpy.org/doc/stable/reference/generated/numpy.nditer.html#numpy.nditer
#https://stackoverflow.com/questions/38332642/plot-the-2d-fft-of-an-image
#https://stackoverflow.com/questions/21362843/interpret-numpy-fft-fft2-output
#https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
