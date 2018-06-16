import numpy as np


def draw_box(image, xmin, ymin, xmax ,ymax):
    """"Draw a box with the given dimensions on the given image"""
    rr, cc = line(xmin, ymin, xmin, ymax-1)
    image[rr, cc] = 1
    rr, cc = line(xmin, ymax-1, xmax-1, ymax-1)
    image[rr, cc] = 1
    rr, cc = line(xmax-1, ymax-1, xmax-1, ymin)
    image[rr, cc] = 1
    rr, cc = line(xmax-1, ymin, xmin, ymin)
    image[rr, cc] = 1
    return image


def check_overlap(bb1, bb2):
    """Do the given bounding boxes overlap?"""
    if np.array_equal(bb1,bb2):
        return False

    # First bounding box, top left corner, bottom right corner
    Amin_y = int(bb1[0])
    Amin_x = int(bb1[1])
    Amax_y = int(bb1[2])
    Amax_x = int(bb1[3])

    # Second bounding box, top left corner, bottom right corner
    Bmin_y = int(bb2[0])
    Bmin_x = int(bb2[1])
    Bmax_y = int(bb2[2])
    Bmax_x = int(bb2[3])

    over_x = any(x in range(Amin_x,Amax_x) for x in range(Bmin_x,Bmax_x))
    over_y = any(y in range(Amin_y,Amax_y) for y in range(Bmin_y,Bmax_y))
    return over_x and over_y

def merge_overlapping(overlapping_bbs):
    maxes = np.max(overlapping_bbs,axis=0)
    mins = np.min(overlapping_bbs,axis=0)
    minY = mins[0]
    minX = mins[1]
    maxY = maxes[2]
    maxX = maxes[3]
    return [minY,minX,maxY,maxX]
