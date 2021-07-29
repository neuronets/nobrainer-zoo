
#### general settings
name: unet_brainy
is_train: true
#use_visdom: false # for visualization
#visdom_port: 8067  
model: cnn
device: cuda:0

#### datasets
dataset:
  n_classes: 1
  train:
    name: sample_MGH
    data_location: data/
    shuffle_buffer_size: 10
    block_shape: 32
    volume_shape: 256  
    batch_size: 2  # per GPU
    augment: False
    num_parallel_calls: 2 # keeping same as batch size
  test:     #  test params may differ from train params
    name: sample_MGH
    data_location: data/
    shuffle_buffer_size: 0
    block_shape: 32
    volume_shape: 256
    batch_size: 1
    num_parallel_calls: 1
    augment: False

#### network structures
network:
  model: unet
  batchnorm: True
#### training settings: learning rate scheme, loss
train:
  epoch: 5
  lr: .00001  # adam
  loss: dice
  metric: dice

#### logger
logger:
  ckpt_path: ckpts/
  
path:
  save_model: model/
  pretrained_model: none
