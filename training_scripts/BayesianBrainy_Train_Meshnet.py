import nobrainer
import tensorflow as tf
import os

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
block_shape = (64, 64, 64)
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
from nobrainer.models.bayesian_meshnet import variational_meshnet
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = variational_meshnet(n_classes=n_classes,input_shape=block_shape+(1,), filters=21, dropout=None, is_monte_carlo=True,receptive_field=37)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
    model.compile(optimizer=optimizer,loss=nobrainer.losses.jaccard,metrics=[nobrainer.metrics.dice, nobrainer.metrics.jaccard],)

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

# Model Training
for e in range(1, 20):
    model.fit(
        dataset_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset_evaluate,
        validation_steps=validation_steps,
        epochs=e+1,
        initial_epoch=e)
model.save_weights('BayesianBrainy_meshnet.hdf5')
