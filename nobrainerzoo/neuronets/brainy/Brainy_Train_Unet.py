import nobrainer
import tensorflow as tf
import os
import json
import click

_option_kwds = {"show_default": True}

@click.command()
@click.argument("data_train_pattern")
@click.argument("data_evaluate_pattern")
@click.option(
    "-v",
    "--volume-shape",
    default=256,
    type=int,
    nargs=1,
    help="Shape of volumes for training data.",
    **_option_kwds,
)
@click.option(
    "-b",
    "--block-shape",
    default=32,
    type=int,
    nargs=1,
    help="Shape of sub-volumes for training data.",
    **_option_kwds,
)
@click.option(
    "--n-classes",
    default=1,
    type=int,
    nargs=1,
    help="Number of classes in labels",
    **_option_kwds,
)
@click.option(
    "--shuffle_buffer_size",
    default=10,
    type=int,
    nargs=1,
    help="Value to fill the buffer for shuffling",
    **_option_kwds,
)
@click.option(
    "--batch_size",
    default=2,
    type=int,
    nargs=1,
    help="Size of batches",
    **_option_kwds,
)
@click.option(
    "--augment",
    default=False,
    is_flag=True,
    help="Apply augmentation to data",
    **_option_kwds,
)
@click.option(
    "--n_train",
    default= None,
    type=int,
    nargs=1,
    help="Number of train samples",
    **_option_kwds,
)
@click.option(
    "--n_valid",
    default= None,
    type=int,
    nargs=1,
    help="Number of validation samples",
    **_option_kwds,
)
@click.option(
    "--num_parallel_calls",
    default= 1,
    type=int,
    nargs=1,
    help="Number of parallel calls",
    **_option_kwds,
)
@click.option(
    "--batchnorm",
    default=True,
    is_flag=True,
    help="Apply batch normalization",
    **_option_kwds,
)
@click.option(
    "--n_epochs",
    default= 1,
    type=int,
    nargs=1,
    help="Number of epochs for training",
    **_option_kwds,
)
@click.option(
    "--lr",
    default= 0.00001,
    type=float,
    nargs=1,
    help="Value for learning rate",
    **_option_kwds,
)
@click.option(
    "--loss",
    type=str,
    required=True,
    help="Loss function",
    **_option_kwds,
    )
@click.option(
    "--metrics",
    type=list,
    required=True,
    help="list of metrics",
    **_option_kwds,
    )
@click.option(
    "--check_point_path",
    type=str,
    help="Path to save training checkpoints",
    **_option_kwds,
    )
@click.option(
    "--save_history",
    type=str,
    help="Path to save training results",
    **_option_kwds,
    )
@click.option(
    "--save_model",
    required=True,
    type=str,
    help="Path to save model weights",
    **_option_kwds,
    )
def main(
        data_train_pattern = "sample_MGH",
        data_evaluate_pattern = "sample_MGH",
        volume_shape = 256,
        block_shape = 32,
        n_classes = 1,
        shuffle_buffer_size = 10,
        batch_size = 2,
        augment = False,
        n_train_data = None,
        n_valid_data = None,
        num_parallel_calls = 1,
        batchnorm = True,
        n_epochs = 1,
        lr = 0.00001,
        loss = nobrainer.losses.dice,
        metrics = [nobrainer.metrics.dice, nobrainer.metrics.jaccard],
        check_point_path = None,
        save_history = None,
        save_model = None,
        pretrained_model = None,
    ):
    """
    Train the brainy model
    """
    # Set parameters
    n_classes = n_classes
    batch_size = batch_size
    v = volume_shape 
    b = block_shape
    volume_shape = (v, v, v)
    block_shape = (b, b, b)
    n_epochs = n_epochs
    
    if data_train_pattern == 'sample_MGH':
        #Load sample Data--- inputs and labels 
        csv_of_filepaths = nobrainer.utils.get_data()
        filepaths = nobrainer.io.read_csv(csv_of_filepaths)
        train_paths = filepaths[:9]
        n_train_data = len(train_paths)
        evaluate_paths = filepaths[9:]
        n_valid_data = len(evaluate_paths)
        
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
        
        data_train_pattern = "data/data-train_shard-*.tfrec"
        
        nobrainer.tfrecord.write(
            features_labels=evaluate_paths,
            filename_template='data/data-evaluate_shard-{shard:03d}.tfrec',
            examples_per_shard=1)
        
        data_evaluate_pattern = "data/data-evaluate_shard-*.tfrec"
        
    else: # TODO: write tfrecords from csv file given by user
        raise ValueError("can't train on non-tfrecord format data." 
                         "convert your data in the form of tfrecords with"
                         "'nobrainer.tfrecord.write'")
    
    # Create and Load Datasets for training and validation
    dataset_train = nobrainer.dataset.get_dataset(
        file_pattern = data_train_pattern,
        n_classes = n_classes,
        batch_size = batch_size,
        volume_shape = volume_shape,
        block_shape = block_shape,
        n_epochs = n_epochs,
        augment = augment,
        shuffle_buffer_size = shuffle_buffer_size,
        num_parallel_calls = num_parallel_calls,
    )
    
    dataset_evaluate = nobrainer.dataset.get_dataset(
        file_pattern = data_evaluate_pattern,
        n_classes = n_classes,
        batch_size = batch_size,
        volume_shape = volume_shape,
        block_shape = block_shape,
        n_epochs = 1,
        augment = augment,
        shuffle_buffer_size = shuffle_buffer_size,
        num_parallel_calls = num_parallel_calls,
    )
    
    # Compile model
    model = nobrainer.models.unet(n_classes=n_classes, input_shape=(*block_shape, 1),batchnorm = batchnorm,)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics,)
    
    # Training Model
    steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_train_data,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size)
    
    print("number of steps per training epoch:", steps_per_epoch)
    
    validation_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_valid_data,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size)
    
    print("number of steps per validation epoch:", validation_steps)
    callbacks = []
    if check_path:
        cpk_call_back = tf.keras.callbacks.ModelCheckpoint(check_path)
        callbacks.append(cpk_call_back)
        
    history = model.fit(
            dataset_train,
            epochs= n_epochs,
            steps_per_epoch=steps_per_epoch, 
            validation_data=dataset_evaluate, 
            validation_steps=validation_steps,
            callbacks=callbacks,
            )
    
    if save_history:
        current_directory = os.getcwd()
        file_name = os.pathjoin(current_directory,f"{save_history}.json")
        with open(file_name,"w") as file:
            json.dump(history, file)
            
    
    saved_model_dir = os.pathjoin(os.getcwd(),r"saved_model")
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    #save model
    model.save_weights(os.path.join(saved_model_dir,f"{save_model}.hdf5" ))
    
    # TODO: Add pretrained model 
        
if __name__ == '__main__':
    
    main()

