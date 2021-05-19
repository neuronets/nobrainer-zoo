import nobrainer
import tensorflow as tf

#Load sample Data--- inputs and labels 
csv_of_filepaths = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)
train_paths = filepaths[:9]
evaluate_paths = filepaths[9:]

#Convert medical images to TFRecords
invalid = nobrainer.io.verify_features_labels(train_paths, num_parallel_calls=2)
assert not invalid
invalid = nobrainer.io.verify_features_labels(evaluate_paths)
assert not invalid

nobrainer.tfrecord.write(
    features_labels=train_paths,
    filename_template='data/data-train_shard-{shard:03d}.tfrec',
    examples_per_shard=3)

nobrainer.tfrecord.write(
    features_labels=evaluate_paths,
    filename_template='data/data-evaluate_shard-{shard:03d}.tfrec',
    examples_per_shard=1)

# Set parameters
n_classes = 1
batch_size = 2
volume_shape = (256, 256, 256)
block_shape = (32, 32, 32)
n_epochs = None
augment = True
shuffle_buffer_size = 100
num_parallel_calls = 2

# Create and Load Datasets for training and validation
dataset_train = nobrainer.dataset.get_dataset(
    file_pattern="data/data-train_shard-*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    augment=augment,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

dataset_evaluate = nobrainer.dataset.get_dataset(
    file_pattern="data/data-evaluate_shard-*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=1,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=1,
)

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

model = variational_meshnet(n_classes=50, input_shape=(32, 32, 32, 1), filters=96, dropout="concrete", receptive_field=37, is_monte_carlo=True)
weights_path = tf.keras.utils.get_file(fname="nobrainer_spikeslab_32iso_weights.h5",
    origin="https://dl.dropbox.com/s/rojjoio9jyyfejy/nobrainer_spikeslab_32iso_weights.h5")
model.load_weights(weights_path)

new_model = tf.keras.Sequential()
for layer in model.layers[:22]:
  new_model.add(layer)
new_model.add(tfp.layers.Convolution3DFlipout(filters=1, 
                                kernel_size = 1, 
                                dilation_rate= (1,1,1),
                                padding = 'SAME',
                                activation=tf.nn.softmax, 
                                name="classification/Mennin3D"))
new_model.compile(tf.keras.optimizers.Adam(lr=1e-02),loss=nobrainer.losses.jaccard,
        metrics=[nobrainer.metrics.dice])

model.fit(
    dataset_train,
    epochs= 20,
    steps_per_epoch=steps_per_epoch, 
    validation_data=dataset_evaluate, 
    validation_steps=validation_steps)

model.save_weights('weights_BAMS_meshnet.hdf5')
