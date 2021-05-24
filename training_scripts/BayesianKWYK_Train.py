
# This script helps to understand hwo to load data and train a multi-class segmentation model
import nobrainer
import tensorflow as tf

#Download sample data and prepare tfrecprds 
csv_of_filepaths = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)
train_paths = filepaths[:9]
evaluate_paths = filepaths[9:]

#Convert medical images to TFRecords
invalid = nobrainer.io.verify_features_labels(train_paths, num_parallel_calls=2)
assert not invalid
invalid = nobrainer.io.verify_features_labels(evaluate_paths)
assert not invalid

import os
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'data')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

nobrainer.tfrecord.write(
    features_labels=train_paths,
    filename_template='data/data-train_shard-{shard:03d}.tfrec',
    examples_per_shard=3)

nobrainer.tfrecord.write(
    features_labels=evaluate_paths,
    filename_template='data/data-evaluate_shard-{shard:03d}.tfrec',
    examples_per_shard=1)

# functions to load a multiclass dataset
import glob
import pandas as pd
import numpy as np

def _to_blocks(x, y,block_shape):
    """Separate `x` into blocks and repeat `y` by number of blocks."""
    print(x.shape)
    x = nobrainer.volume.to_blocks(x, block_shape)
    y = nobrainer.volume.to_blocks(y, block_shape)
    return (x, y)

def get_dict(n_classes):
    print('Conversion into {} segmentation classes from freesurfer labels to 0-{}'.format(n_classes,n_classes-1))
    if n_classes == 50: 
        tmp = pd.read_csv('50-class-mapping.csv', header=0,usecols=[1,2],dtype=np.int32) # if 50 classes mapping file required
        tmp = tmp.iloc[1:,:] # removing the unknown class
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        return mydict
    elif n_classes == 115:
        tmp = pd.read_csv('115-class-mapping.csv', header=0,usecols=[0,1],dtype=np.int32)# if 115 classes mapping file required
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        del tmp
        return mydict
    else: raise(NotImplementedError)
    
def process_dataset(dset,batch_size,block_shape,n_classes,one_hot_label= False,training= True):
    # Standard score the features.
    dset = dset.map(lambda x, y: (nobrainer.volume.standardize(x), nobrainer.volume.replace(y,get_dict(n_classes))))
    
    # Separate features into blocks.
    dset = dset.map(lambda x, y:_to_blocks(x,y,block_shape))
    if one_hot_label:
        dset= dset.map(lambda x,y:(x, tf.one_hot(y,n_classes)))
    dset = dset.unbatch()
    if training:
        dset = dset.shuffle(buffer_size=100)
    # Add a grayscale channel to the features.
    dset = dset.map(lambda x, y: (tf.expand_dims(x, -1), y))
    # Batch features and labels.
    dset = dset.batch(batch_size, drop_remainder=True)
    return dset

def get_dataset(file_pattern,volume_shape,batch_size,block_shape,n_classes,one_hot_label= False,training = True):

    dataset = nobrainer.dataset.tfrecord_dataset(
        file_pattern=glob.glob(file_pattern),
        volume_shape=volume_shape,
        shuffle=False,
        scalar_label=False,
        compressed=True)
    dataset = process_dataset(dataset,batch_size,block_shape,n_classes, one_hot_label= one_hot_label ,training = training)
    return dataset

# Hyperparameters   
n_classes = 50
block_shape = (32, 32, 32)
batch_size = 2
volume_shape = (256, 256, 256)
n_epochs = None
augment = True
shuffle_buffer_size = 10
num_parallel_calls = 2

# Dataloaders 
                
dataset_train = get_dataset(
    file_pattern="data/data-train_shard-*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    one_hot_label=True)

dataset_evaluate = get_dataset(
    file_pattern="data/data-evaluate_shard-*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    training = False,
    one_hot_label=True)

steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(train_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size)

steps_per_epoch

validation_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(evaluate_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size)

validation_steps

# Initialize and Compile Model
from nobrainer.models.bayesian_meshnet import variational_meshnet
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = variational_meshnet(n_classes=50, input_shape=(32, 32, 32, 1), filters=96, dropout="concrete", receptive_field=37, is_monte_carlo=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy() ,metrics=[nobrainer.metrics.dice])
 
model.fit(
    dataset_train,
    epochs= 10,
    steps_per_epoch=steps_per_epoch, 
    validation_data=dataset_evaluate, 
    validation_steps=validation_steps)

model.save_weights('weights_kwyk_nokld.hdf5')
