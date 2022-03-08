from pathlib import Path
import subprocess as sp
import json

def get_model_path(model_name, model_type=None):
    """
    creates the path based on the model_name
    
    model_name: string value indicationg the <org>/<model>/<version>
    model_type: model type for braingen and kwyk model

    Returns
    -------
    model_path: path to pretrained model file in trained_models repository
    
    """
    database_path = Path(__file__).resolve().parent / "model_database.json"
    with open(database_path, "r") as fp:
        models=json.load(fp)
    
    org,mdl,ver = model_name.split("/")
        
    root_path = Path(__file__).resolve().parent / "trained-models"
    
    if not model_type:
        return root_path / model_name / models[model_name]
    else:
        return root_path / model_name / models[model_name][model_type]
    

def load_model(path):
    """ Returns the model object from file path"""
    
    if "kwyk" in path:
        from nobrainer.pediction import _get_pridictor
        model = _get_pridictor(path)
    else:
        from nobrainer.prediction import _get_model
        model = _get_model(path)
    return model

def get_repo(org, repo_url, repo_state):
    """
    downoads the related repo in the org/org_repo.
    org: str, organization name
    
    """
    repo_path = Path(__file__).resolve().parents[0] / org / "org_repo"
    if not repo_path.exists():
        p0 = sp.run(["git", "clone", repo_url, str(repo_path)], stdout=sp.PIPE,
                    stderr=sp.STDOUT ,text=True)
        print(p0.stdout)
        p1 = sp.run(["git","-C",str(repo_path),"checkout", repo_state], stdout=sp.PIPE,
                    stderr=sp.STDOUT ,text=True)
        print(p1.stdout)
        print(f"{org} repository is downloaded")
    else:
        print(f"{org} repository is available locally")
        
