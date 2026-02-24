import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import os
import glob

def average_dataset(directory_path):
    """
    Computes the mean image from a directory containing multiple 2D FITS files.
    
    Args:
        directory_path (str): Path to the folder containing .fits files.
        
    Returns:
        numpy.ndarray: The 2D averaged image.
    """
    # 1. Gather all fits files
    # Change extension to *.fit or *.fits.gz if necessary
    file_list = glob.glob(os.path.join(directory_path, '*.fits'))
    
    if not file_list:
        raise FileNotFoundError(f"No .fits files found in directory: {directory_path}")

    image_sum = None
    count = 0

    print(f"Found {len(file_list)} files. Averaging...")

    # 2. Iterate through files
    for file_path in file_list:
        try:
            with fits.open(file_path) as hdul:
                # Load data and ensure it is float (to prevent integer overflow during sum)
                data = hdul[0].data.astype(float)
                
                # Handle edge cases where 2D data is stored as (1, Y, X)
                if data.ndim == 3:
                    data = np.squeeze(data)
                
                # Initialize the accumulator using the shape of the first valid file
                if image_sum is None:
                    image_sum = np.zeros_like(data)
                    expected_shape = data.shape
                
                # Ensure all files share the same dimensions
                if data.shape != expected_shape:
                    print(f"Skipping {file_path}: shape mismatch {data.shape} vs {expected_shape}")
                    continue

                # Add to running total
                image_sum += data
                count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 3. Compute final average
    if count == 0:
        raise ValueError("No valid files were processed.")
        
    avg_image = image_sum / count
    
    return avg_image

def construct_average_psd_from_dataset(directory_path):
    """
    Constructs the averaged PSD from a directory containing 2D FITS files.
    
    Args:
        directory_path (str): Path to the folder containing .fits files.
        
    Returns:
        numpy.ndarray: The 2D averaged Power Spectral Density.
    """
    # 1. Gather all fits files in the directory
    # You can adjust '*.fits' to '*.fit' or '*.fits.gz' if needed
    file_list = glob.glob(os.path.join(directory_path, '*.fits'))
    
    if not file_list:
        raise FileNotFoundError(f"No .fits files found in directory: {directory_path}")

    psd_sum = None
    count = 0

    print(f"Found {len(file_list)} files. Processing...")

    # 2. Iterate through files individually to save memory
    for file_path in file_list:
        try:
            with fits.open(file_path) as hdul:
                # Assuming data is in the primary extension (index 0)
                data = hdul[0].data.astype(float)
                
                # Handle cases where data might be saved as (1, Y, X)
                if data.ndim == 3:
                    data = np.squeeze(data)
                
                # Check dimensions on the first file to initialize the accumulator
                if psd_sum is None:
                    psd_sum = np.zeros_like(data, dtype=float)
                    expected_shape = data.shape
                
                # Ensure all files have the same dimensions
                if data.shape != expected_shape:
                    print(f"Skipping {file_path}: shape mismatch {data.shape} vs {expected_shape}")
                    continue

                # 3. Compute PSD for this specific frame
                # Magnitude squared of fft
                current_psd = (np.abs(np.fft.fft2(data)))**2
                
                # Add to running total
                psd_sum += current_psd
                count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 4. Compute the average
    if count == 0:
        raise ValueError("No valid files were processed.")
        
    avg_psd = psd_sum / count
    
    return avg_psd


def dark_sub(image, dark, saturation_limit=15000, transition_width=250):
    """
    Subtracts the dark image from the science image. Since noise patterns are only
    visible on nonsaturated portions of the image, dark subtraction is not applied
    to the saturated pixels, as doing so would leave negative imprints of the dark
    image on the science image. 
    
    Args:
        image (numpy.ndarray): The input science image.
        dark (numpy.ndarray): The corresponding dark image to be subtracted off.
        saturation_limit (int): The numerical pixel value corresponding to saturation of the detector.
        transition_width (int): Determines how sharp the subtraction cutoff gradient is at the saturation limit,
        The dark sub starts tapering off for pixel values greater than saturation_limit - transition_width,
        and for pixel values greater than saturation_limit + transition width, no subtraction is applied.
        
    Returns:
        numpy.ndarray: The dark subtracted science image.
    """
    start_limit = saturation_limit - transition_width
    end_limit = saturation_limit + transition_width
    weights = np.zeros(shape=image.shape)
    # compute weight gradient so that dark sub is less aggressive for saturated parts of the image
    weights[image < start_limit] = 1
    weights[image > end_limit] = 0
    in_btwn_indices = (image >= start_limit) & (image <= end_limit)
    weights[in_btwn_indices] = (end_limit - image[in_btwn_indices]) / (2 * transition_width)

    # compute dark sub with the added weight gradient
    return image - (weights * dark)

def create_bandstop_filter(dark_psd, threshold, zeroth_order_transmission_width, gaussian_kernel_radius):
    """
    Creates a transmission profile as a function of spatial frequency. This filter estimates 
    the spatial frequencies of the structural noise using the power spectral density of the dark image
    and places a stopband at those frequencies. 
    
    Args:
        dark_psd (numpy.ndarray): The power spectral density of the dark image.
        threshold (float): Controls the aggressiveness of the filter, ranges from 0 to 1, higher value corresponds to a
        more aggressive filter.
        zeroth_order_transition_width (float): The width of the zeroth order passband as a percentage of the total image width,
        ranges from 0 to 1. 
        gaussian_kernel_radius (int): The radius of the gaussian kernel used for convolution with the filter mask 
        
    Returns:
        numpy.ndarray: The bandstop filter mask.
    """
    # check for valid input conditions
    if threshold < 0 or threshold > 1:
        raise ValueError('threshold must be in between 0 and 1')
    
    if zeroth_order_transmission_width < 0 or zeroth_order_transmission_width > 1:
        raise ValueError('zeroth_order_transmission_width must be in between 0 and 1')
    
    # create bandstop filter from dark psd
    # set the cutoff percentile that determines which frequencies are transmitted or attenuated
    # for example threshold of 0.1 creates bandstop filter where the top 10% highest value pixels in the dark PSD is blocked,
    # everything else is transmitted with a coefficient of 1
    percentile_cutoff = (1 - threshold) * 100
    cutoff = np.percentile(dark_psd, percentile_cutoff)
    print(cutoff)
    bandstop_filter = np.ones_like(dark_psd)
    # zero out everything above cutoff
    bandstop_filter[dark_psd>cutoff] = 0

    # force DC component transmission to be one so DC component is conserved
    # shift fft so that DC component is in the middle
    bandstop_filter = np.fft.fftshift(bandstop_filter)
    img_shape = bandstop_filter.shape
    center_y = img_shape[0] // 2
    center_x = img_shape[1] // 2
    n_pix = round(img_shape[0] * zeroth_order_transmission_width)
    bandstop_filter[center_y-n_pix:center_y+n_pix, center_x-n_pix:center_x+n_pix] = 1
    # shift back so things work as normal
    bandstop_filter = np.fft.ifftshift(bandstop_filter)

    # applying convolve with a gaussian to minimize ringing artifacts in the spatial domain
    bandstop_filter = gaussian_filter(bandstop_filter, gaussian_kernel_radius)

    return bandstop_filter

def create_wiener_filter(image_psd, dark_psd, weight, power):
    """
    Creates a transmission profile as a function of spatial frequency. This filter compares the spatial frequencies
    occupied by the science image and the spatial frequencies occupied by the noise and selectively attenuates the
    frequencies more dominated by the noise. The transmission profile is given generally by the formula
    [|(S(u,v)|^2)/(|S(u,v)|^2+weight*|N(u,v)|^2)]^power where |S(u,v)|^2 is the power spectral density of the science
    image and |N(u,v)|^2 is the power spectral density of the dark image.
    
    Args:
        image_psd (numpy.ndarray): The power spectral density of the science image
        dark_psd (numpy.ndarray): The power spectral density of the dark image.
        Weight (float): Determines how heavily the noise is weighted against the signal. A higher weight value corresponds
        to a more aggressive filter.
        power (float): Determines how aggressive the filter is. A higher power value creates a steeper attenuation gradient
        as a function of the signal to noise ratio.  
        
    Returns:
        numpy.ndarray: The Wiener filter mask.
    """
    # check for valid input conditions
    if weight < 0:
        raise ValueError('Weight must be a positive value')
    
    if power < 0:
        raise ValueError('Power must be a positive value')
    
    # construct Wiener filter
    return (image_psd / (image_psd + (weight * dark_psd)))**power