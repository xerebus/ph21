# Ph21 Set 3
# Aritra Biswas

# edge_detector.py
# Detects edges in an image

import Image
import ImageFilter
from scipy import ndimage
import sys
import matplotlib.pyplot as plotter
import matplotlib.cm as cm
import time

def blur_image(im, res = 1000):
    '''Return a Gaussian-blurred image with a radius calculated
    for a desired stencil size.'''

    blur_radius = max(lx, ly) / res
    return im.filter(ImageFilter.GaussianBlur(blur_radius))

def get_pixel_data(im):
    '''Given a PIL image object, return a 2D array populated with
    RGB tuples for each pixel.'''

    px_list = list(im.getdata()) # 1d list of pixels
    return [px_list[(n*lx):((n + 1)*lx)] for n in range(ly)]

def get_brightness_data(px_data, show = False):
    '''Given a 2D array of RGB tuples, return the brightness array.'''

    bright_data = [[sum(pixel) for pixel in row] for row in px_data]

    if show:
        plotter.imshow(bright_data)
        plotter.show()

    return bright_data

def get_edges(bright_data, sig = 0.5, show = False):
    '''Given a 2D array of brightnesses, compute the gradient magnitude
    to identify the edges. Either show the result or save it.'''

    edge_data = ndimage.filters.gaussian_gradient_magnitude(bright_data, sig)
    fig = plotter.imshow(edge_data, cmap = cm.binary)

    
    if show:
        plotter.show()
    else:
        plotter.axis("off")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plotter.savefig("sig%s_%s.png" % (sig,
            time.strftime("%Y%m%d%H%M%S")), bbox_inches = "tight",
            pad_inches = 0)
        plotter.clf()

    return edge_data

if __name__ == "__main__":

    try:
        path = sys.argv[1]
    except:
        raise ValueError("Need a single argument to an image file.")

    im = Image.open(path)
    (lx, ly) = im.size

    im = blur_image(im)
    px_data = get_pixel_data(im)
    bright_data = get_brightness_data(px_data)
    for sig in [0.2, 0.5, 1, 3, 5]:
        edge_data = get_edges(bright_data, sig, show = False)

