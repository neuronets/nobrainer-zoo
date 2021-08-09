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
nobrainerzoo predict -m neuronets\brainy\0.1.0 <path_to_input> <path_to_save_output>
```

# Train Example

Training brainy model with sample data,

```
nobrainer-zoo train sample_MGH sample_MGH -m neuronets/brainy
```
