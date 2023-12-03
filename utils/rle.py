import numpy as np
import cv2


def rle_to_mask(rle: str, shape=(768, 768)):
    """
    Convert run-length encoded pixels to a binary mask.

    Parameters:
        rle (str): Run-length encoded pixels as a string.
        shape (tuple): Dimensions (height, width) of the mask.

    Returns:
        numpy.ndarray: Binary mask, where 1 represents the mask and 0 represents the background.
    """
    # Convert run-length encoded pixels to arrays of starting and ending positions
    encoded_pixels = np.array(rle.split(), dtype=int)
    starts = encoded_pixels[::2] - 1
    ends = starts + encoded_pixels[1::2]

    # Create an array and set pixels within the specified ranges to 1
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction


def mask_to_rle(img, shape=(768, 768)) -> str:
    """
    Convert a binary mask to run-length encoded pixels.

    Parameters:
        img (numpy.ndarray): Binary mask, where 1 represents the mask and 0 represents the background.
        shape (tuple): Dimensions (height, width) of the image.

    Returns:
        str: Run-length encoded pixels as a string.
    """
    # Resize the image and convert to binary mask
    img = img.astype('float32')
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)

    # Flatten the mask and concatenate start and end markers
    pixels = img.T.flatten()
    pixels = np.concatenate([np.array([0]), pixels, [0]])

    # Calculate run-length encoding
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
