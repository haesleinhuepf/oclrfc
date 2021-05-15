import pyclesperanto_prototype as cle


def generate_feature_stack(image):
    """
    Creates a feature stack from a given image.

    Todo: enable definition of which features.

    Parameters
    ----------
    image : ndarray
        2D or 3D image to generate a feature stack from

    Returns
    -------
    a list of OCLarray images
    """
    image = cle.push(image)
    blurred = cle.gaussian_blur(image, sigma_x=2, sigma_y=2, sigma_z=2)
    edges = cle.sobel(blurred)
    stack = [
        image,
        blurred,
        edges
    ]

    return stack