import os
from pathlib import Path
import subprocess as sp
import datalad.api
from utils import get_model_path

def get_nobrainer_model(model_path):
    """
    downloads the nobrainer model located in model_path.
    
    """ 
    
    # if trained model is not already cloned
    if not os.path.exists("nobrainerzoo/trained-models"):
        url="git@github.com:neuronets/trained-models.git"
        datalad.api.clone(source=url, path="nobrainerzoo/trained-models")
        
    if not os.path.exists(model_path):
        datalad.api.get(dataset="nobrainerzoo/trained-models",path=model_path)
        
def get_repo(org, model_nm, repo_url):
    """
    downoads the related repo in the org/repo_repo.
    org: str, organization name
    model_nm: str, model name

    """
    repo_path = Path(__file__).resolve().parents[0] / org / "org_repo" / model_nm
    if not repo_path.exists():
        p0 = sp.run(["git", "clone", repo_url, str(repo_path)], stdout=sp.PIPE,
                    stderr=sp.STDOUT ,text=True)
        print(p0.stdout)
        print(f"{model_nm} model is downloaded")
    else:
        raise Exception("f{model_nm} is available locally")
        
       
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        model_name = str(sys.argv[1])
        path = get_model_path(model_name)
    else:
        model_name = str(sys.argv[1])
        model_type = str(sys.argv[2])
        path = get_model_path(model_name, model_type)
        
    get_nobrainer_model(path)


