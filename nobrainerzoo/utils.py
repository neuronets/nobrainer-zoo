import os
from pathlib import Path
import subprocess as sp

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
    models = {"neuronets/ams/0.1.0": "meningioma_T1wc_128iso_v1.h5",
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
              "UCL/SynthSeg/0.1": "SynthSeg.h5"
              }
    
    # model type should be given for braingen and kwyk
    if model_name in ["braingen","kwyk"] and model_type not in models[model_name]:
        raise Exception("Model type should be one of {} but it is {}".format(
          models[model_name], model_type))
        
    root_path = "nobrainerzoo/trained-models/"
    
    if model_name in ["neuronets/braingen/0.1.0", "neuronets/kwyk/0.4.1"]:
        model_file = os.path.join(models[model_name],model_type)
    else:
        model_file = models[model_name]
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

def get_repo(org, repo_url):
    """
    downoads the related repo in the org/org_repo.
    org: str, organization name
    
    """
    repo_path = Path(__file__).resolve().parents[0] / org / "org_repo"
    if not repo_path.exists():
        p0 = sp.run(["git", "clone", repo_url, str(repo_path)], stdout=sp.PIPE,
                    stderr=sp.STDOUT ,text=True)
        print(p0.stdout)
        print(f"{org} repository is downloaded")
    else:
        print(f"{org} repository was available locally")