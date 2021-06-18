import numpy as np
import os
import argparse
import random
import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.python.keras.layers import Input, Convolution2D, Flatten, Dense, Activation, MaxPooling2D, add, Dropout, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, GlobalAvgPool2D
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential
from math import sqrt

# Allow GPU memory growth
if hasattr(tf, 'GPUOptions'):
    import keras.backend as K
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.tensorflow_backend.set_session(sess)
else:
    # For other GPUs
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def get_map(pdb_path, dist_path, true_npy = False):
    seqy = None
    mypath = dist_path + pdb_path + '.npy'

    if os.path.exists(mypath):
        if not true_npy:
            cb_map = np.load(mypath, allow_pickle = True)
            ly = len(cb_map)
        else:
            (ly, seqy, cb_map) = np.load(mypath, allow_pickle = True)
    else:
        print('Expected distance map file for', pdb_path, 'not found at', dist_path)
        exit(1)
    Y = cb_map

    return Y, ly

'''
***** Calculate LDDT here
'''
# Helpers for metrics calculated using numpy scheme
def get_flattened(dmap):
  if dmap.ndim == 1:
    return dmap
  elif dmap.ndim == 2:
    return dmap[np.triu_indices_from(dmap, k=1)]
  else:
    assert False, "ERROR: the passes array has dimension not equal to 2 or 1!"

def get_separations(dmap):
  t_indices = np.triu_indices_from(dmap, k=1)
  separations = np.abs(t_indices[0] - t_indices[1])
  return separations
  
# return a 1D boolean array indicating where the sequence separation in the
# upper triangle meets the threshold comparison
def get_sep_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = get_flattened(dmap)
  separations = get_separations(dmap)
  if comparator == 'gt':
    threshed = separations > thresh
  elif comparator == 'lt':
    threshed = separations < thresh
  elif comparator == 'ge':
    threshed = separations >= thresh
  elif comparator == 'le':
    threshed = separations <= thresh

  return threshed

# return a 1D boolean array indicating where the distance in the
# upper triangle meets the threshold comparison
def get_dist_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = get_flattened(dmap)
  if comparator == 'gt':
    threshed = dmap_flat > thresh
  elif comparator == 'lt':
    threshed = dmap_flat < thresh
  elif comparator == 'ge':
    threshed = dmap_flat >= thresh
  elif comparator == 'le':
    threshed = dmap_flat <= thresh
  return threshed

# Calculate lDDT using numpy scheme
def get_LDDT(true_map, pred_map, R=15, sep_thresh=6, T_set=[0.5, 1, 2, 4], precision=4):
    '''
    Mariani V, Biasini M, Barbato A, Schwede T.
    lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests.
    Bioinformatics. 2013 Nov 1;29(21):2722-8.
    doi: 10.1093/bioinformatics/btt473.
    Epub 2013 Aug 27.
    PMID: 23986568; PMCID: PMC3799472.
    '''
    
    # Helper for number preserved in a threshold
    def get_n_preserved(ref_flat, mod_flat, thresh):
        err = np.abs(ref_flat - mod_flat)
        n_preserved = (err < thresh).sum()
        return n_preserved
    
    # flatten upper triangles
    true_flat_map = get_flattened(true_map)
    pred_flat_map = get_flattened(pred_map)
    
    # Find set L
    S_thresh_indices = get_sep_thresh_b_indices(true_map, sep_thresh, 'gt')
    R_thresh_indices = get_dist_thresh_b_indices(true_flat_map, R, 'lt')
    
    L_indices = S_thresh_indices & R_thresh_indices
    
    true_flat_in_L = true_flat_map[L_indices]
    pred_flat_in_L = pred_flat_map[L_indices]
    
    # Number of pairs in L
    L_n = L_indices.sum()
    
    # Calculated lDDT
    preserved_fractions = []
    for _thresh in T_set:
        _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)
        _f_preserved = _n_preserved / L_n
        preserved_fractions.append(_f_preserved)
        
    lDDT = np.mean(preserved_fractions)
    if precision > 0:
        lDDT = round(lDDT, precision)
    return lDDT

def trrosetta_probindex2dist(index):
    d = 1.75
    for k in range(1, 37):
        d += 0.5
        if index == k:
            return d
    return d

def trrosetta2maps(a):
    if len(a[0, 0, :]) != 37:
        print('ERROR! This does not look like a trRosetta prediction')
        return
    D = np.full((len(a), len(a)), 21.0)
    for i in range(len(a)):
        for j in range(len(a)):
            maxprob_value = 0.0
            for k in range(37):
                if maxprob_value < a[i, j, k]:
                    maxprob_value = a[i, j, k]
                    D[i, j] = trrosetta_probindex2dist(k)
    return D

def get_feature_channels(npz_path, npy_path):

    if os.path.exists(npz_path) and os.path.exists(npy_path):
        x1 = np.load(npz_path)
        a = x1['dist']

        y_pred = trrosetta2maps(a)

        x2 = np.load(npy_path)
        b = x2[0]
        
        c = np.dstack((a, b, y_pred))

        return c
    else:
        print('File does not exist')
        exit()


def get_input_features(npz_path, npy_path, xy_dimension, expected_n_channels):
    X_input = np.zeros((1, xy_dimension, xy_dimension, expected_n_channels))
    
    x = get_feature_channels(npz_path, npy_path)
    
    l, w, c = x.shape
    
    if l <= xy_dimension:
        X_input[0, 0: l, 0: l, :] = x
    else:
        rx = random.randint(0, l - xy_dimension)
        ry = rx
        assert rx + xy_dimension <= l
        assert ry + xy_dimension <= l
        
        X_input[0, :, :, :] = x[rx:rx+xy_dimension, ry:ry+xy_dimension, :]

    return X_input


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f1', type=str, required=True,
                            dest='npz_file',    help="target file with .npz extension")
    parser.add_argument('-f2', type=str, required=True,
                            dest='npy_file',    help="target file with .npy extension")
    args = parser.parse_args()
    
    return args

args = get_args()

npz_path = args.npz_file
npy_path = args.npy_file

if npz_path:
    path_list = npz_path.split('/')
    pdb_id = (path_list[-1]).split('.')[0]

    target_dir = ''
    for i in range(len(path_list)-1):
        target_dir = os.path.join(target_dir, path_list[i])

window_dimen = 512
expected_n_channels = 564

print('')
print ('Build Model')


base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(window_dimen, window_dimen, expected_n_channels)
)

model = Sequential()
model.add(BatchNormalization(input_shape=(window_dimen, window_dimen, expected_n_channels)))

model.add(base_model)
model.add(BatchNormalization())
model.add(Convolution2D(1, 3, padding = 'same', activation="relu"))
model.add(GlobalAveragePooling2D())



x_input = get_input_features(npz_path, npy_path, window_dimen, expected_n_channels) 

file_weights = 'model_weights512.hdf5'

model.load_weights(file_weights)
P = model.predict_generator(x_input, max_queue_size=10, verbose=1)
P = P.flatten()

np.set_printoptions(formatter = {'float': '{: 0.3f}'.format})

result_path = os.path.join(target_dir, 'results.txt')
fo = open(result_path, 'w')

fo.write("lDDT score for " + pdb_id + '\n')
fo.write('lDDT score: ')
for item in P:
    print ("Predicted lDDT: %.3f" % item)
    fo.write(str(round(item, 3)) + "\n")

fo.close()
