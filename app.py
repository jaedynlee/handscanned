import numpy as np

from classifier import classify
from page import preprocess, get_bbox, merge, sort_bbox, resize


def main(filename):
    print('Processing image...')
    page = preprocess(filename) # read in, clean image
    bboxes = get_bbox(page)  # get expanded bounding boxes around characters
    bboxes = merge(bboxes)  # merge expanded bounding boxes
    bboxes = sort_bbox(bboxes)  # sort characters to order that you'd read them in
    characters = resize(page, bboxes)  # resize to 28x28, centered in 20x20

    # Make each image take a single row in the big batch image by flattening the
    # width (2nd) and height (3rd) dimension.
    characters = np.reshape(characters, (len(characters), -1))

    print('Classifying characters...')
    # Return classified images as a string of ASCII characters.
    predictions = classify(characters)
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    text = []
    for pred in predictions:
        text.append(letters[int(pred)])
    text = ''.join(text)
    print('Writing to file...')
    # Write out to file
    base_filename = filename.split('.')[0]
    f = open('{}.txt'.format(base_filename), 'w+')
    f.write(text)
    f.close()


main('classifyAZ.jpg')
