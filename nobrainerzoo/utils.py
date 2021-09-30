import os

def get_model_path(model_name, model_type=None):
    """
    creates the path based on the model_name
    
    model_name: string value indicationg the <org>/<model>/<version>
    model_type: model type for braingen and kwyk model

    Returns
    -------
    model_path
    
    """
    # TO DO: model database can be a json file and be updated separately
    models = {"neuronets/ams/0.1.0": "neuronets/ams/0.1.0/meningioma_T1wc_128iso_v1.h5",
              "neuronets/braingen/0.1.0": ["neuronets/braingen/0.1.0/generator_res_8",
                                 "neuronets/braingen/0.1.0/generator_res_16",
                                 "neuronets/braingen/0.1.0/generator_res_32",
                                 "neuronets/braingen/0.1.0/generator_res_64",
                                 "neuronets/braingen/0.1.0/generator_res_128",
                                 "neuronets/braingen/0.1.0/generator_res_256"],
              "neuronets/brainy/0.1.0": "neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5",
              "neuronets/kwyk/0.4.1": ["neuronets/kwyk/0.4.1/all_50_wn",
                                      "neuronets/kwyk/0.4.1/all_50_bwn_09_multi",
                                      "neuronets/kwyk/0.4.1/all_50_bvwn_multi_prior"],
              "UCL/SynthSeg/0.1": "models/SynthSeg.h5"
              }
    
    org, model_nm, ver = model_name.split("/")
    # model type should be entered for braingen and kwyk
    if model_name in ["braingen","kwyk"] and model_type not in models[model_name]:
        raise Exception("Model type should be one of {} but it is {}".format(
          models[model_name], model_type))
        
    root_path = "nobrainerzoo/" + org
    
    # create the model path 
    if model_nm in ["neuronets/braingen/0.1.0", "neuronets/kwyk/0.4.1"]:
        model_path = os.path.join(root_path, models[model_name],model_type)
    else:
        model_path = os.path.join(root_path, models[model_name])
    
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