from pathlib import Path
import subprocess as sp
import os
import yaml

CACHE_PATH = Path(os.path.expanduser('~')) / ".nobrainer"
if "NOBRAINER_CACHE" in os.environ:
    CACHE_PATH = Path(os.environ["NOBRAINER_CACHE"]).resolve()
MODELS_PATH = CACHE_PATH / "trained-models"


def get_spec(model, model_type):
    org, model_nm, ver = model.split("/")
    
    _check_model_type(model, model_type)
    
    if model_type:
        model_dir = MODELS_PATH / model / model_type
    else:
        model_dir = MODELS_PATH / model
        
    spec_file = model_dir / "spec.yaml"
    
    if not model_dir.exists():
        raise Exception("model directory not found!",
                        "This model does not exist in the zoo or didn't properly added.")
    if not spec_file.exists():
        raise Exception("spec file doesn't exist!",
                        "This model does not exist in the zoo or didn't properly added.")
    
    with spec_file.open() as f:
        spec = yaml.safe_load(f)
    
    return spec


def get_model_path(model_db, model_name, model_type=None):
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

def get_repo(repo_url, destination, repo_state=None):
    """
    downoads the repository to the destination path.
    repo_url: str, url of the git repository
    destination: Path or str, destination on the local file system
    repo_state: optional, git commit
   
    """
    if not destination.exists():
        p0 = sp.run(["git", "clone", repo_url, str(destination)], stdout=sp.PIPE,
                    stderr=sp.STDOUT ,text=True)
        print(p0.stdout)
        if repo_state:
            p1 = sp.run(["git", "-C", str(destination), "checkout", repo_state], stdout=sp.PIPE,
                        stderr=sp.STDOUT, text=True)
            print(p1.stdout)
        print(f"{repo_url} repository is downloaded")
    else:
        print(f"{repo_url} repository is available locally")
        

def get_model_db(models_repo, print_models=True):
    """
    Extracts the model's database from trained_model repository.'
    
    models_repo: Path like object, path to where trained_model repository is downloaded.
    print_models: if True, available models will print out when calling the function.
                    
    """
    
    # create the model database
    model_ext = ("*.h5","*.pb","*.ckpt")
    paths =[]
    for ext in model_ext:
        path_sublist = [path for path in Path(models_repo).rglob(ext) if ".git" not in str(path)]
        paths.extend(sorted(path_sublist))

    model_db={}
    for i, pth in enumerate(paths):
        #breakpoint()
        if pth.parts[-6] == 'trained-models': # if no model_type
            org = pth.parts[-5]
            model_name = pth.parts[-4]
            version = pth.parts[-3]
            if print_models:
                print(org+"/"+model_name+"/"+version)
            model = org+"/"+model_name+"/"+version
            # check the model extention
            if not pth.suffix == '.pb':
                model_db[model] = str(pth) # should we add a path object?
            else:
                model_db[model] = str(pth.parents[0])
                
        elif pth.parts[-7] == 'trained-models': # if there is model type
            org = pth.parts[-6]
            model_name = pth.parts[-5]
            version = pth.parts[-4]
            model_type = pth.parts[-3]
            if print_models:
                print(org+"/"+model_name+"/"+version+"/"+model_type)
            model = org+"/"+model_name+"/"+version
            # to avoid deleting previously added model types
            if  not model in model_db:
                model_db[model] = {}
                
            if not pth.suffix == '.pb':
                model_db[model][model_type] = str(pth) # should we add a path object?
            else:
                model_db[model][model_type] = str(pth.parents[0])
                
        # else:
        #     # TODO: consider if the exception should be raised or not
        #     print(f"{i}: The {pth} is not added in proper format and it should be checked!",
        #                     " It is NOT considered to model database.")
            
    return model_db


def pull_singularity_image(singularity_image, path):
    download_image = Path(path) / singularity_image
    if not download_image.exists():
        image_tag , _ = singularity_image.split("_")[1].split(".")
        print("Downloading the container file. it might take a while...")
        dwnld_cmd = ["singularity", "pull", "--dir", 
                     str(path),
                     # container images are stored in dockerhub neuronets/nobrainer-zoo
                     f"docker://neuronets/nobrainer-zoo:{image_tag}"]
        p0 = sp.run(dwnld_cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        print(p0.stdout)

        
def _check_model_type(model_name, model_type=None):
    
    models = get_model_db(MODELS_PATH, print_models=False)     
    org,mdl,ver = model_name.split("/")

    models_w_types = [m.split("/")[1] for m,v in models.items() if isinstance(v,dict)]
    
    # check if model_types is given and correct
    if mdl in models_w_types and model_type not in models[model_name].keys():
        raise ValueError("Model type should be one of {} but it is {}".format(
          list(models[model_name].keys()), model_type))
    elif mdl not in models_w_types and model_type != None:
        raise ValueError(f"{model_name} does not have model type")
