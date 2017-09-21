import numpy as np

def reshape_for_display(image_pixels):
    shape = np.shape(image_pixels)
    if (shape[1] ** 0.5) - int(shape[1] ** 0.5) == 0:
        image = np.reshape(image_pixels, [shape[0], int(shape[1] ** 0.5), int(shape[1] ** 0.5)])
    else:
        image = np.reshape(image_pixels, [shape[0], shape[1], 1])

    return image
