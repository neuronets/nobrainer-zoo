#### general setting
is_train: true
use_visdom: false
visdom_port: 8067
model: cnn
device: cuda:0

#### Dataset Setting
dataset:
  read: ram # disk
  scale: 2
  train:
    name: DIV2K
    data_location: data/datasets/DIV2k/
    shuffle: true
    n_workers: 1  # per GPU
    batch_size: 40
    lr_size: 48
    repeat: 2
  test:
    name: Set14
    data_location: data/datasets/Set14/
    shuffle: false
    n_workers: 1  # per GPU
    batch_size: 1
    repeat: 1

####  Architecture Setting
network_G:
  model: UNET

#### training settings: learning rate scheme, loss
train:
  epoch: 1
  cl_train: false
  lr_G: !!float 1e-4
  weight_decay_G: 0
  beta1_G: 0.9
  beta2_G: 0.99

  lr_scheme: MultiStepLR
  lr_step: [200, 400, 600, 800]
  lr_gamma: 0.5
  val_freq: 5 # epoch

#### logger
logger:
  print_freq: 6 # epoch
  chkpt_freq: 100 # epoch
  img_freq: 100
  path: experiments/
