from pathlib import Path
import subprocess as sp


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
    downoads the related repo to the destination.
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

def pull_singularity_image(image, path):
    download_image = Path(path) / image
    if not download_image.exists():
        print("Downloading the container file. it might take a while...")
        dwnld_cmd = ["singularity", "pull", "--dir", 
                     str(path),
                     # container images are stored in dockerhub neuronets/nobrainer-zoo
                     f"docker://neuronets/nobrainer-zoo:{image}"]
        p0 = sp.run(dwnld_cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        print(p0.stdout)
