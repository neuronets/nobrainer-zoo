
"""
Implementation of voxelmorph
G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag and A. V. Dalca, "VoxelMorph: A Learning Framework for Deformable Medical Image Registration," 
in IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1788-1800, Aug. 2019, doi: 10.1109/TMI.2019.2897538.

@author: Dhritiman Das


libraries:
 - voxelmorph
 - tf (2.1.x)
 - neurite
 - numpy
 - pystrum

"""

import os, sys



# third party imports
import numpy as np
import subprocess
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'Tensorflow 2.0+'

import voxelmorph as vxm
import neurite as ne

subprocess.call('wget https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz -O data.tar.gz', shell=True)

subprocess.call('tar -xzvf data.tar.gz', shell=True)

npz = np.load('tutorial_data.npz')
x_train = npz['train']
x_val = npz['validate']


# Define the volume shape for the pretrained weights which will be loaded
vol_shape = (160, 192, 224)

# Feature shape for the encoder, decoder

nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16] 
]

vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0);

# losses and loss weights
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
lambda_param = 0.01 #smoothness
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

#loading a pretrained 3D model

vxm_model.load_weights('brain_3d.h5')

#preparing validation dataset

val_volume_1 = np.load('subj1.npz')['vol']
seg_volume_1 = np.load('subj1.npz')['seg']
val_volume_2 = np.load('subj2.npz')['vol']
seg_volume_2 = np.load('subj2.npz')['seg']

val_input = [
    val_volume_1[np.newaxis, ..., np.newaxis],
    val_volume_2[np.newaxis, ..., np.newaxis]
]

# registration
val_pred = vxm_model.predict(val_input)

#first component 'val_pred[0]' is the moving image warped using the displacement tensor
#second tensor 'val_pred[1]' is the displacement
moved_pred = val_pred[0].squeeze()
pred_warp = val_pred[1]

# plotting the mid-slices 

mid_slices_fixed = [np.take(val_volume_2, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(moved_pred, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)

mid_slice_plot = ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[2,3]);

# to analyze segmentations
warp_model = vxm.networks.Transform(vol_shape, interp_method='nearest')
warped_seg = warp_model.predict([seg_volume_1[np.newaxis,...,np.newaxis], pred_warp])

ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['jet'], do_colorbars=True, grid=[2,3]);