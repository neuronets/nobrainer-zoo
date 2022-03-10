from pathlib import Path
import subprocess as sp


def get_model_path(model_db, model_name, model_type=None,):
    """
    returns the path based on model_name and model_type
    
    model_name: string value indicationg the <org>/<model>/<version>
    model_type: str, model type

    Returns
    -------
    model_path: path to pretrained model file in trained_models repository
    
    """
    
    if not model_type:
        return model_db[model_name]
    else:
        return model_db[model_name][model_type]
    

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
        

def get_model_db(models_repo):
    """
    downloads the the trained_model repository to a given path and extracts
    the model database.
    
    models_repo: str, path to where trained_model repository is downloaded or
    should be downloaded.
                    
    """
    
    models_repo = Path(models_repo)
    if not models_repo.exists():
        raise Exception(f"{models_repo} does not exists!")
    
    # create the model database
    model_ext = ("*.h5","*.pb","*.ckpt")
    paths =[]
    for ext in model_ext:
        paths.extend(sorted(Path(models_repo).rglob(ext)))

    model_db={}    
    for pth in paths:
    
        if pth.parts[-6] == 'trained-models':   # if no model_type
            org = pth.parts[-5]
            model_name = pth.parts[-4]
            version = pth.parts[-3]
            print(org+"/"+model_name+"/"+version)
            model = org+"/"+model_name+"/"+version
            # check the model extention
            if not pth.suffix == '.pb':
                model_db[model] = str(pth) # should we add a path object?
            else:
                model_db[model] = str(pth.parents[0])
                
        elif pth.parts[-7] == 'trained-models':    # if there is model type
            org = pth.parts[-6]
            model_name = pth.parts[-5]
            version = pth.parts[-4]
            model_type = pth.parts[-3]
            print(org+"/"+model_name+"/"+version+"/"+model_type)
            model = org+"/"+model_name+"/"+version
            # to avoid deleting previously added model types
            if  not model in model_db:
                model_db[model] = {}
                
            if not pth.suffix == '.pb':
                model_db[model][model_type] = str(pth) # should we add a path object?
            else:
                model_db[model][model_type] = str(pth.parents[0])
                
        else:
            raise Exception(f"The {pth} is not added in proper format and it should be checked!",
                            " It is NOT considered to model database.")
            
    return model_db
        
