# Nobrainer-zoo
Nobrainer-zoo is a toolbox with a collection of deep learning neuroimaging models that eases the use of pretrained 
models for various applications. Nobrainer-zoo provides the required environment with all the dependencies for 
training/inference of models. You need `singularity/apptainer` (>= 3.7.x/1.0.x) or `docker` to run the Nobrainer-zoo.


## Installation
We highly recommend to create a separate environment for the nobrainer-zoo. You can use conda or python to create the environment.

```
conda create -n nobrainer-zoo python=3
conda activate nobrainer-zoo
```

or

```
python3 -m venv /path/to/new/virtual/environment/nobrainer-zoo
source /path/to/new/virtual/environment/nobrainer-zoo/bin/activate
```

Then install the nobrainer-zoo:

```
[Releases]    pip install nobrainer-zoo
[Dev version] pip install https://github.com/neuronets/nobrainer-zoo/archive/refs/heads/main.zip
```

After installation, Nobrainer-zoo should be initialized. It also needs a cache folder to download some helper files based on your needs. By default, it creates a cache folder in your home directory (`~/.nobrainer`). If you do not want the cache folder in your `home` directory, you can setup a different cache location by setting the environmental variable `NOBRAINER_CACHE`. run below command to set it.

```
export NOBRAINER_CACHE=<path_to_your_cache_directory>
```

since this environmental variable will be lost when you close your terminal session you need to run it next time or a better solution is, to add it to your `~/.bashrc` file.
simply open the file with your text editor and add the above line at the end. Restart your terminal or re-run your `bashrc` file by `.~/bashrc` to make this change effective.

To initialize the `nobrainer-zoo` run:

```
nobrainer-zoo init
```

*<font size="1">Note: You need to initialize the nobrainer-zoo only once.

Run help to see the functions and each function's options.

```
nobrainer-zoo --help
nobrainer-zoo ls --help
nobrainer-zoo predict --help
nobrainer-zoo fit --help
nobrainer-zoo register --help
nobrainer-zoo generate --help
```

## Available models

To see the list of available models in the nobrainer-zoo run `nobrainer-zoo ls`

Models are added based on their organization, model name , and version. One model might have different versions. Some models (such as `kwyk` or `SyntSR`) have various types which means there was various training method or dataset that lead to different trained models. You can select the model type with `model_type` option for the train and inference.


List of models which will be added in near future can be find [here](https://github.com/Hoda1394/zoo/blob/add/inference_scripts/models_to_add.md). You can suggest a model [here](https://github.com/neuronets/zoo/issues/new/choose).

*<font size="1">Note: models are distributed under their original license.</font>*

## Inference Example

Inference with default options,

```
nobrainer-zoo predict -m neuronets/brainy/0.1.0 <path_to_input> <path_to_save_output>

nobrainer-zoo register -m DDIG/SynthMorph/1.0.0 --model_type brains <path_to_moving> <path_to_fixed> <path_to_moved>
```

pass the model specific options with `--options` argument to the model.

```
nobrainer-zoo predict -m neuronets/brainy/0.1.0 <path_to_input> <path_to_save_output> --options verbose block_shape=[128,128,128]

nobrainer-zoo predict -m UCL/SynthSeg/0.1 <path_to_input> <path_to_save_output> --options post=<path_to_posteriors>
```

**Note**: Nobrainer-zoo will use the gpu by default. So, if you want to force it to use the cpu while the gpu is available you need to pass `--cpu` flag. If you are using docker without any gpu passing the `--cpu` flag is a must. Otherwise, you will get an error.

**Note**: If you are using docker make sure to use the absolute path for input and putput files.

## Train Example

For training with sample dataset you do not need to pass any dataset pattern.

```
nobrainer-zoo fit -m neuronets/brainy
```

To train the network with your own data pass the dataset pattern in the form of tfrecords.

```
nobrainer-zoo fit -m neuronets/brainy "<data_train_pattern>" "<data_evaluate_pattern>"
```

Other parameters can be changed by providing a spec file or changing them with cli command.

```
nobrainer-zoo fit -m neuronets/brainy --spec_file <path_to_spec_file>
```

```
nobrainer-zoo fit -m neuronets/brainy --train epoch=2
```
