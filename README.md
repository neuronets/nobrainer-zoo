# Nobrainer-zoo
Nobrainer-zoo is a deep learning neuroimaging model-zoo toolbox that eases the use of trained deep learning models for varous application. Nobrainer zoo provides and enviroment with all the dependencies required for training or inference from models. The only software needed is `singularity` or `Docker`.

To use the Nobrainer-zoo,

```
clone https://github.com/neuronets/zoo.git
cd zoo
pip install .

```

Models should be refrenced based on their organization, and model name (`neuronets\brainy`). The trained models are version controled and one model might have different version. Therefore for inference, the model version also needs to be specified(`neuronets\brainy\0.1.0`). 
Some models (`kwyk` and `braingen`) also have various types which means there was different structural chracteristic during training that leads to different trained models. Run help to see the functions and each function's options.

```
nobrainer-zoo --help
nobrainer-zoo predict --help
nobrainer-zoo train --help
```

# Inference Example

Inference with default options,

```
nobrainer-zoo predict -m neuronets/brainy/0.1.0 <path_to_input> <path_to_save_output>
```

# Train Exam

For training with sample dataset you do not need to pass any dataset pattern

```
nobrainer-zoo train -m neuronets/brainy
```

To train the network with your own data pass the dataset pattern in the form of tfrecords.

```
nobrainer-zoo train -m neuronets/brainy "<data_train_pattern>" "<data_evaluate_pattern>"
```

Other parameters are also can be changed by providing a spec file or changing them with cli command.

```
nobrainer-zoo train -m neuronets/brainy --spec_file <path_to_spec_file>
```

```
nobrainer-zoo train -m neuronets/brainy --train epoch=2
```
