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

# Set parameters
n_classes = 1
batch_size = 2
volume_shape = (256, 256, 256)
block_shape = (128, 128, 128)
n_epochs = None
augment = False
shuffle_buffer_size = 10
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

# Compile model
model = nobrainer.models.unet(n_classes=n_classes, input_shape=(*block_shape, 1),batchnorm=True,)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
model.compile(optimizer=optimizer,loss=nobrainer.losses.dice,metrics=[nobrainer.metrics.dice, nobrainer.metrics.jaccard],)

# Training Model
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

model.fit(
    dataset_train,
    epochs= 20,
    steps_per_epoch=steps_per_epoch, 
    validation_data=dataset_evaluate, 
    validation_steps=validation_steps)

#save model
model.save_weights('weights_brainy_unet.hdf5')
