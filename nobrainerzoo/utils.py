import os

def get_model_path(model_name, model_type=None):
    """
    creates the path based on the model_name
    
    model_name: string value indicationg the <org>/<model>/<version>
    mofel_type: model type for braingen and kwyk model

    Returns
    -------
    model_path
    
    """
    # TO DO: model database can be a json file and be updated separately
    neuronet_models = {"neuronets/ams/0.1.0": "meningioma_T1wc_128iso_v1.h5",
              "neuronets/braingen/0.1.0": ["generator_res_8",
                                 "generator_res_16",
                                 "generator_res_32",
                                 "generator_res_64",
                                 "generator_res_128",
                                 "generator_res_256"],
              "neuronets/brainy/0.1.0": "brain-extraction-unet-128iso-model.h5",
              "neuronets/kwyk/0.4.1": ["all_50_wn",
                                      "all_50_bwn_09_multi",
                                      "all_50_bvwn_multi_prior"],
              }
    
    # model type should be entered for braingen and kwyk
    if model_name in ["braingen","kwyk"] and model_type not in neuronet_models[model_name]:
        raise Exception("Model type should be one of {} but it is {}".format(
          neuronet_models[model_name], model_type))
        
    root_path = "nobrainerzoo/trained-models/"
    
    if model_name in ["neuronets/braingen/0.1.0", "neuronets/kwyk/0.4.1"]:
        model_file = os.path.join(neuronet_models[model_name],model_type)
    else:
        model_file = neuronet_models[model_name]
    # create the model path     
    model_path = os.path.join(root_path,model_name,model_file)
    return model_path

def load_model(path):
    """ Returns the model object from file path"""
    
    if "kwyk" in path:
        from nobrainer.pediction import _get_pridictor
        model = _get_pridictor(path)
    else:
        from nobrainer.prediction import _get_model
        model = _get_model(path)
    return model