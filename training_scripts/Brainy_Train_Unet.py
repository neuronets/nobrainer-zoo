import nobrainer
import tensorflow as tf
import os
import argparse
import yaml

def main(config):
    # Set parameters
    n_classes = config['dataset']['n_classes']
    batch_size = config['dataset']['train']['batch_size']
    v =  config['dataset']['train']['volume_shape']
    b = config['dataset']['train']['block_shape']
    volume_shape = (v, v, v)
    block_shape = (b, b, b)
    n_epochs = config['train']['epoch']
    
    if config['dataset']['train']['name'] == 'sample_MGH':
        #Load sample Data--- inputs and labels 
        csv_of_filepaths = nobrainer.utils.get_data()
        filepaths = nobrainer.io.read_csv(csv_of_filepaths)
        train_paths = filepaths[:9]
        evaluate_paths = filepaths[9:]
        
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
    else: # TODO  for any datase6
        raise ValueError("cant take any random dataset yet")
    
    # Create and Load Datasets for training and validation
    dataset_train = nobrainer.dataset.get_dataset(
        file_pattern="data/data-train_shard-*.tfrec",
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_epochs=n_epochs,
        augment=config['dataset']['train']['augment'],
        shuffle_buffer_size=config['dataset']['train']['shuffle_buffer_size'],
        num_parallel_calls=config['dataset']['train']['num_parallel_calls'],
    )
    
    dataset_evaluate = nobrainer.dataset.get_dataset(
        file_pattern="data/data-evaluate_shard-*.tfrec",
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_epochs=1,
        augment=config['dataset']['test']['augment'],
        shuffle_buffer_size=config['dataset']['test']['shuffle_buffer_size'],
        num_parallel_calls=config['dataset']['test']['num_parallel_calls'],
    )
    
    # Compile model
    model = nobrainer.models.unet(n_classes=n_classes, input_shape=(*block_shape, 1),batchnorm=config['network']['batchnorm'],)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['train']['lr'])
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
        epochs= n_epochs,
        steps_per_epoch=steps_per_epoch, 
        validation_data=dataset_evaluate, 
        validation_steps=validation_steps)
    
    #save model
    model.save_weights(os.path.join(config['path']['save_model'],'weights_brainy_unet.hdf5' ))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, help='Path to config YAML file.')
	args = parser.parse_args()
	with open(args.config, 'r') as stream:
		config = yaml.safe_load(stream)
	main(config)
