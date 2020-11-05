import numpy as np

def gray2rgb(im, cmap='gray'):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt  # doesn't work in docker
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, cmap='plasma'):
    depth = 1 / depth.astype('float32')
    depth = depth / np.nanmax(depth)
    # depth = gray2rgb(depth, cmap=cmap)
    return depth * 255