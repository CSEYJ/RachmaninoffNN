'''
Reimplementation of visualize.py (https://github.com/calclavia/DeepJ/blob/icsc/visualize.py) with PyTorch.

Credit to the original implementation:

MIT License

Copyright (c) 2018 Calclavia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import torch as tf
import numpy as np
import os
from keras import backend as K

from util import *
from constants import *

# Visualize using:
# http://projector.tensorflow.org/
def main():
    models = build_or_load()
    style_layer = models[0].get_layer('style')

    print('Creating input')
    style_in = tf.placeholder(tf.float32, shape=(NUM_STYLES, NUM_STYLES))
    style_out = style_layer(style_in)

    # All possible styles
    all_styles = np.identity(NUM_STYLES)

    with K.get_session() as sess:
        embedding = sess.run(style_out, { style_in: all_styles })

    print('Writing to out directory')
    np.savetxt(os.path.join(OUT_DIR, 'style_embedding_vec.tsv'), embedding, delimiter='\t')

    labels = [[g] * len(styles[i]) for i, g in enumerate(genre)]
    # Flatten
    labels = [y for x in labels for y in x]

    # Retreive specific artists
    styles_labels = [y for x in styles for y in x]

    styles_labels = np.reshape(styles_labels, [-1, 1])
    labels = np.reshape(labels, [-1, 1])
    labels = np.hstack([labels, styles_labels])

    # Add metadata header
    header = ['Genre', 'Artist']
    labels = np.vstack([header, labels])

    np.savetxt(os.path.join(OUT_DIR, 'style_embedding_labels.tsv'), labels, delimiter='\t', fmt='%s')

if __name__ == '__main__':
    main()
