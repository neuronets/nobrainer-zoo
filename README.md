# Nobrainer-zoo
Nobrainer-zoo is a toolbox with a collection of deep learning neuroimaging models that eases the use of pretrained models for various applications. Nobrainer-zoo provides the required environment with all the dependencies for training/inference of models. The only software needed is `singularity` or `Docker`.

To use the Nobrainer-zoo,

```
git clone https://github.com/neuronets/zoo.git
cd zoo
pip install .

```

Models should be refrenced based on their organization, and model name (`neuronets/brainy`). The trained models are version controled and one model might have different version. Therefore for inference, the model version also needs to be specified(`neuronets/brainy/0.1.0`). 
Some models (`kwyk` and `braingen`) also have various types which means there was different structural chracteristic during training that leads to different trained models. Run help to see the functions and each function's options.

```
nobrainer-zoo --help
nobrainer-zoo predict --help
nobrainer-zoo train --help
```

# Available models

- [brainy](https://github.com/neuronets/brainy): 3D U-Net brain extraction model (available for training and inference)
- [ams](https://github.com/neuronets/ams): 3D U-Net meningioma segmentation model (available for training and inference)
- [SynthSeg](https://github.com/BBillot/SynthSeg): Contrast and resolution 3D brain segmentation model (available for inference)


List of models which will be added in near future can be find [here](https://github.com/Hoda1394/zoo/blob/add/inference_scripts/models_to_add.md). You can suggest a model [here]().

*<font size="1">Note: For models, their original license is applied.</font>*

# Inference Example

Inference with default options,

```
nobrainer-zoo predict -m neuronets/brainy/0.1.0 <path_to_input> <path_to_save_output>

nobrainer-zoo predict -m UCL/SynthSeg/0.1 <path_to_input> <path_to_save_output>
```

pass the model specific options with `--options` argument to the model.

```
nobrainer-zoo predict -m neuronets/brainy/0.1.0 <path_to_input> <path_to_save_output> --options verbose block_shape=[128,128,128]

nobrainer-zoo predict -m UCL/SynthSeg/0.1 <path_to_input> <path_to_save_output> --options post=<path_to_posteriors>
```

# Train Example

For training with sample dataset you do not need to pass any dataset pattern.

```
nobrainer-zoo train -m neuronets/brainy
```

To train the network with your own data pass the dataset pattern in the form of tfrecords.

```
nobrainer-zoo train -m neuronets/brainy "<data_train_pattern>" "<data_evaluate_pattern>"
```

Other parameters can be changed by providing a spec file or changing them with cli command.

```
nobrainer-zoo train -m neuronets/brainy --spec_file <path_to_spec_file>
```

```
nobrainer-zoo train -m neuronets/brainy --train epoch=2
```
