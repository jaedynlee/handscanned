import math

import numpy as np
from skimage import exposure
from skimage import filters
from skimage import img_as_uint
from skimage import io
from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import transform
from skimage import util

import config
from page_utils import check_overlap, draw_box, merge_overlapping


def preprocess(filename):
    """Perform image processing to isolate characters from page background"""
    # Read in image
    filename_wo_extension = filename.split('.')[0]
    page = io.imread(filename, as_grey=True)

    # Blur the image
    page_blur = filters.gaussian(page)

    # Logarithmic correction to even background color
    page_logarithmic_corrected = exposure.adjust_log(page_blur, 1)

    # Increase contrast
    p3, p98 = np.percentile(page_logarithmic_corrected, (3, 98))
    page_contrast = exposure.rescale_intensity(page_logarithmic_corrected, in_range=(p3, p98))

    # Separate characters from background and remove anythind touching the edge
    page_filter = page_contrast < filters.threshold_minimum(page_contrast)
    page_filter = segmentation.clear_border(page_filter)

    # Remove specks
    page_clean = morphology.remove_small_objects(page_filter, 30)

    # Dilate characters
    page_dilate = morphology.binary_dilation(page_clean);
    page_dilate = morphology.binary_dilation(page_dilate);
    page_dilate = morphology.binary_dilation(page_dilate);
    page_dilate = morphology.binary_dilation(page_dilate);

    return page_dilate


def get_bbox(page):
    """Returns a list of bounding boxes for every connected component on the given page"""
    # Identify connected components
    page_label = measure.label(page, background=0)
    blob_info = measure.regionprops(page_label)
    bboxes = np.zeros((len(blob_info), 4))

    i = 0
    for blob in blob_info:
        bbox = blob.bbox
        bbox = [int(max(bbox[0] - config.Y_EXPANSION, 0)), int(max(bbox[1] - config.X_EXPANSION, 0)), int(min(bbox[2] + config.Y_EXPANSION, page.shape[0])), int(min(bbox[3] + config.X_EXPANSION, page.shape[1]))]
        bboxes[i] = bbox
        i = i + 1

    return bboxes


def merge(expanded_bbs):
    """Given a list of bounding boxes, merges any that are overlapping"""
    i = 0
    blacklist = []
    for bb1 in expanded_bbs:
        if any(np.array_equal(row, bb1) for row in blacklist):
            continue

        has_overlaps = False
        overlapping_bbs = []
        for bb2 in expanded_bbs:
            if any(np.array_equal(row, bb2) for row in blacklist):
                continue
            if check_overlap(bb1, bb2):
                if not has_overlaps: overlapping_bbs = bb2
                else: overlapping_bbs = np.vstack((overlapping_bbs, bb2))
                has_overlaps = True
        if has_overlaps:  # if the box is overlapping others
            overlapping_bbs = np.vstack((overlapping_bbs, bb1))
            merged_bb = merge_overlapping(overlapping_bbs)
            if i == 0: blacklist = overlapping_bbs
            else: blacklist = np.vstack((blacklist, overlapping_bbs))
        else:  # if the box isn't overlapping others
            merged_bb = bb1
            if i == 0: blacklist = bb1
            else: blacklist = np.vstack((blacklist, bb1))
        if i == 0: merged_bbs = merged_bb
        else: merged_bbs = np.vstack((merged_bbs, merged_bb))
        i = i + 1
    # Shrink every bounding box by the amount we previously expanded it by
    for i in range(0, merged_bbs.shape[0]):
        bbox = merged_bbs[i]
        merged_bbs[i] = [bbox[0] + config.Y_EXPANSION, bbox[1] + config.X_EXPANSION, bbox[2] - config.Y_EXPANSION, bbox[3] - config.X_EXPANSION]
    return merged_bbs


def sort_bbox(bboxes):
    """Order the merged bounding boxes by rounded y, then x value"""
    indices = [0]
    for i in range(1, 52):
        indices = np.vstack((indices,i))
    indices_and_bbs = np.hstack((indices, bboxes))
    # Round yn coordinates up to nearest 400 so everything's on the same line
    indices_and_bbs[:, 3] = [int(math.ceil(y0 / 400.0)) * 400 for y0 in indices_and_bbs[:, 3]]
    i = np.lexsort((indices_and_bbs[:, 2], indices_and_bbs[:, 3]))
    indices_and_bbs = indices_and_bbs[i]
    indices = indices_and_bbs[:, 0]
    sorted_bboxes = bboxes[int(indices[0]), :]
    for i in indices[1:]:
        bbox = bboxes[int(i), :]
        sorted_bboxes = np.vstack((sorted_bboxes, bbox))
    return sorted_bboxes


def resize(page, bboxes):
    """
    Take the characters outlined by each merged bounding box and resize to
    20x20 with 8 total pixels of vertical and horizontal padding

    :return: a list of flattened character images
    """
    char_dimension = 20.0
    padding_dimension = 8.0
    characters = []
    count = 0
    # For each of the bounding boxes
    for bbox in bboxes:
        # The desired dimensions for each cropped character
        character_sized = np.zeros((int(char_dimension + 8), int(char_dimension + 8)))

        # Crop to bounding box
        character_cropped = page[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]

        # Resize to 20x20.
        blob_height = bbox[3] - bbox[1]
        blob_width = bbox[2] - bbox[0]

        scale_factor = char_dimension / max(blob_height, blob_width)
        character_cropped = transform.rescale(character_cropped, scale_factor)
        crop_height = character_cropped.shape[0]
        crop_width = character_cropped.shape[1]

        # The number of blank rows to put at the top and bottom of the character.
        # (desired total height - actual character height) * 0.5, to center the character
        num_pad_rows = int(np.floor(((char_dimension + padding_dimension) - crop_height) * 0.5))
        # The number of blank pixels to the left of the character
        # (desired total width - actual character width) * 0.5, to center the character
        pad_left = np.zeros((1, int(np.floor((char_dimension + padding_dimension) - crop_width) * 0.5)))
        # The number of blank pixels to the right of the character
        # (desired total width - actual character width - pad_left), to ensure it is 28 pixels wide
        pad_right = np.zeros((1, int(char_dimension + padding_dimension - len(pad_left[0]) - crop_width)))

        # Traverse the rows of the eventual final image.
        for i in range(0, int(char_dimension + padding_dimension)):
            # If it is a row that the character inhabits,
            if i in range(int(num_pad_rows), int(crop_height + num_pad_rows)):
                # Append the left padding, character, and right padding
                new_row = np.hstack((pad_left, [character_cropped[i - num_pad_rows]], pad_right))
                # Replace the default black with the new_row
                character_sized[i] = new_row
            # If it is not a row that the character inhabits, the row stays all black.
        # Flatten the image into one row
        character_flattened = np.reshape(character_sized, -1)
        # Put the values in the range of 0-255, like the training data
        character = np.multiply(character_flattened, 255.0 / max(character_flattened))
        character[character > 170] = 255
        # Save to array
        characters = character if characters == [] else np.vstack((characters, character))
        count = count + 1
    print('Identified ' + str(count) + ' characters.')
    return characters
