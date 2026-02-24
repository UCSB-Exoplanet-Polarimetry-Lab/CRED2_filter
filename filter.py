import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
import os
import glob

def average_dataset(file_path):
    # averages the images in a fits dataset
    hdul = fits.open(file_path)
    image_datacube = np.array(hdul[0].data)
    if (image_datacube.ndim == 2):
        return image_datacube
    return np.mean(image_datacube, axis=0)

def average_dataset_from_directory(directory_path):
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


def construct_psd(file_path):
    # construct averaged psd when given a fits dataset
    hdul = fits.open(file_path)
    size_x = hdul[0].header['NAXIS1']
    size_y = hdul[0].header['NAXIS2']
    try: 
        nframes = hdul[0].header['NAXIS3']
    except KeyError as e:
        nframes = 1

    if nframes == 1:
        image_datacube = np.array([hdul[0].data])
    else:
        image_datacube = np.array(hdul[0].data)
    psd_datacube = np.zeros(shape=(nframes, size_y, size_x)) 
    hdul.close()

    # compute the power spectral density of each frane
    for i in range(nframes):
        # magnitude squared of fft
        psd_datacube[i] = (np.abs(np.fft.fft2(image_datacube[i])))**2
        
    # average and return final averaged PSD
    psd = np.mean(psd_datacube, axis=0)
    return psd

def construct_psd_from_directory(directory_path):
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

def scipy_interpolate(data, mask, method):
    """
    Uses cubic interpolation to recover signal structure in masked regions.
    mask: 0 where data was removed (hole), 1 where data is valid.
    """
    # 1. Setup coordinates
    h, w = data.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # 2. Separate into "Known" points and "Unknown" points
    # We want to interpolate values at 'unknown_points' using 'known_points'
    known_y = y[mask == 1]
    known_x = x[mask == 1]
    known_points = np.column_stack((known_x, known_y))
    
    unknown_y = y[mask == 0]
    unknown_x = x[mask == 0]
    target_points = np.column_stack((unknown_x, unknown_y))
    
    # Optimization: If the image is huge (e.g. 4k x 4k), griddata is slow.
    # It is faster to only pass valid points that are NEAR the holes.
    # But for standard 512x512 or 1k x 1k, this is usually acceptable.
    
    # 3. Get the known values
    known_data = data[mask == 1]
    
    print(f"Interpolating {len(target_points)} pixels... this might take a moment.")

    # 4. Interpolate Real and Imaginary parts separately
    # 'cubic' method is CRITICAL here. 'nearest' creates steps, 'linear' creates cones.
    # 'cubic' creates smooth curves which mimics real signal diffraction patterns.
    
    # We use fill_value=0 just in case valid data doesn't perfectly convex hull the hole
    # (though in a notch filter, the hole is usually internal, so it's fine)
    
    real_interp = griddata(
        known_points, known_data.real, target_points, method=method, fill_value=0
    )
    
    imag_interp = griddata(
        known_points, known_data.imag, target_points, method=method, fill_value=0
    )
    
    # 5. Recombine
    reconstructed_hole = real_interp + 1j * imag_interp
    
    # 6. Patch the original array
    interpolated_data = data.copy()
    interpolated_data[mask == 0] = reconstructed_hole
    
    return interpolated_data

def astropy_interpolate(data, mask, radius):
    bad_pixels = (mask == 0)

    # set 0 values to NaN in order to perform interpolation
    data[bad_pixels] = np.nan

    # split ft into real and imaginary parts to interpolate separtely
    data_real = data.real
    data_imag = data.imag

    # radius of interpolation kernel
    radius = 1
    kernel = Gaussian2DKernel(x_stddev=radius)

    # perform interpolation
    interpolated_data_real = interpolate_replace_nans(data_real, kernel)
    interpolated_data_imag = interpolate_replace_nans(data_imag, kernel)
    interpolated_data = interpolated_data_real + (1j * interpolated_data_imag)
    return interpolated_data

def improve_mask(mask, neighbor_threshold):
    # make filter more robust so that single pixels of 0 with no neighbors are replaced with 1,
    # pixels with at least the neighbor threshold have all their neighbors filled with with 0s
    mask_new = mask.copy()
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] == 0:
                n_neighbors = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        if mask[k, l] == 0:
                            n_neighbors += 1
                if n_neighbors >= neighbor_threshold:
                    for k in range(i - 1, i + 2):
                        for l in range(j - 1, j + 2):
                            mask_new[k, l] = 0
                else:
                    mask_new[i, j] = 1
    return mask_new

def add_gradient(mask, n_iterations):
    if (n_iterations <= 0):
        return mask
    mask_new = mask.copy()
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i,j] > 0:
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        if mask[k, l] == 0:
                            mask_new[k, l] = 0.5 * mask[i,j]
    if (n_iterations > 1):
         return add_gradient(mask_new, n_iterations - 1)
    return mask_new

def smart_dark_sub(image, dark, saturation_limit=14000, transition_width=500):
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

def apply_notch_filter(image, dark_psd, threshold, lowpass_transmission_width_percentage, gaussian_kernel_radius):
    # create notch filter from dark psd
    # set threshold from 0 to 1, determines how agressive notch filter is
    # for example threshold of 0.1 creates notch filter where the top 10% highest value pixels in the dark PSD is blocked,
    # everything else is transmitted with a coefficient of 1
    percentile_cutoff = (1 - threshold) * 100
    cutoff = np.percentile(dark_psd, percentile_cutoff)
    print(cutoff)
    notch_filter = np.ones_like(dark_psd)
    # zero out everything above cutoff
    notch_filter[dark_psd>cutoff] = 0

    # force DC component transmission to be one so DC component is conserved
    # shift fft so that DC component is in the middle
    notch_filter = np.fft.fftshift(notch_filter)
    img_shape = notch_filter.shape
    center_y = img_shape[0] // 2
    center_x = img_shape[1] // 2
    # what percentage of the total image width do we want the size of the DC/low passband to be
    lowpass_transmission_width_percentage = 30
    n_pix = img_shape[0] // lowpass_transmission_width_percentage
    notch_filter[center_y-n_pix:center_y+n_pix, center_x-n_pix:center_x+n_pix] = 1
    # shift back so things work as normal
    notch_filter = np.fft.ifftshift(notch_filter)
    # applying blurring to filter to minimize artifacts
    notch_filter = gaussian_filter(notch_filter, gaussian_kernel_radius)

    # apply filter to image
    image_ft = np.fft.fft2(image)
    filtered_ft = image_ft * notch_filter
    filtered_image = np.abs(np.fft.ifft2(filtered_ft))

    return filtered_image

def apply_wiener_filter(image, dark_psd, weight, gain):
    image_ft = np.fft.fft2(image)
    image_psd = (np.abs(image_ft))**2
    wiener_filter = (image_psd / (image_psd + (weight * dark_psd)))**gain
    filtered_ft = image_ft * wiener_filter
    filtered_image = np.abs(np.fft.ifft2(filtered_ft))

    return filtered_image