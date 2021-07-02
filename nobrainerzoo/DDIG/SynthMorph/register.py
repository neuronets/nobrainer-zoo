#!/usr/bin/env python

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at:
# 
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


# reference
ref = '''
If you find this code useful, please cite:

    Learning MRI Contrast-Agnostic Registration.
    Hoffmann M, Billot B, Iglesias JE, Fischl B, Dalca AV.
    IEEE International Symposium on Biomedical Imaging (ISBI), pp 899-903, 2021.
    https://doi.org/10.1109/ISBI48211.2021.9434113
'''


# parse command line
p = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=f'''
Register two image volumes with a SynthMorph model. For accurate deformable
registration, the source and target input images are expected to be aligned
within the affine space used at training.

Example:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.h5 
        --out_moved moved.nii.gz --out-warp warp.nii.gz
{ref}
''')

p.add_argument('--moving', required=True, help='path to moving (source) image')
p.add_argument('--fixed', required=True, help='path to fixed (target) image')
p.add_argument('--model', required=True, help='path to H5 model file')
p.add_argument('--out-moved', help='output warped image path')
p.add_argument('--out-warp', help='output deformation field path')
p.add_argument('--gpu', help='ID of GPU to use, defaults to CPU')
arg = p.parse_args()


# tensorflow device handling
vxm.tf.utils.setup_device(arg.gpu)


def load(f):
    '''Load and normalize image volume.'''
    im, aff = vxm.py.utils.load_volfile(
        f, add_batch_axis=True, add_feat_axis=True, ret_affine=True,
    )
    im = np.asarray(im, np.float32)
    im -= np.min(im)
    im /= np.max(im)
    return im, aff


# load images
moving, _ = load(arg.moving)
fixed, affine = load(arg.fixed)
in_shape = moving.shape[1:-1]


# load model and predict
model = vxm.networks.VxmDense.load(arg.model, input_model=None)
warp = model.register(moving, fixed)
moved = vxm.networks.Transform(in_shape).predict([moving, warp])


# save outputs
if arg.out_moved:
    vxm.py.utils.save_volfile(np.squeeze(moved), arg.out_moved, affine)
if arg.out_warp:
    vxm.py.utils.save_volfile(np.squeeze(warp), arg.out_warp, affine)


print('\nThank you for using SynthMorph!', ref.lstrip())
