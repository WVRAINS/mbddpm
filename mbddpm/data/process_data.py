import numpy as np
import math


def process_data(data, add_method):

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy ndarray")

    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")

    if add_method == 'code':
        # Process data using the ImageCoding method
        try:
            processed_data, shape_info = image_coding(data)
        except NameError:
            raise NameError("ImageCoding method is not defined. Please make sure it is properly imported or implemented.")
    else:
        n_samples, n_features = data.shape
        processed_data = data.reshape(n_samples, 1, n_features)
        shape_info = [n_features]

    return processed_data, shape_info


def image_coding(matrix):
    num_features = matrix.shape[1]
    min_pixels = math.ceil(num_features / 64) * 64
    m = int(math.ceil(math.sqrt(min_pixels) / 8)) * 8
    n = int(math.ceil(min_pixels / m / 8)) * 8
    if m < n:
        m, n = n, m
    target_dim = m * n
    padded = np.zeros((matrix.shape[0], target_dim))
    padded[:, :num_features] = matrix
    tensor = padded.reshape(matrix.shape[0], 1, m, n)
    return tensor, (m, n)