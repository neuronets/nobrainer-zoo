from pathlib import Path
import subprocess as sp
import datalad.api
from utils import get_model_path

def get_model(model_path):
    """
    downloads the model located in model_path.
    
    """ 
    model_repo = Path(__file__).resolve().parent / "trained-models"
    
    # if trained model is not already cloned
    if not model_repo.exists():
        url="https://github.com/neuronets/trained-models.git"
        datalad.api.clone(source=url, path= model_repo)
        p0=sp.run(["git", "config", "user.name", "nobrainer-zoo"],
               stdout=sp.PIPE, stderr=sp.STDOUT, shell=True) #text=True)
        print(p0.stdout)
        p1=sp.run(["git", "config", "user.email", "nobrainer-zoo"],
               stdout=sp.PIPE, stderr=sp.STDOUT, shell=True) #text=True)
        print(p1.stdout)
        # # set repo config with datalad api
        # datalad.api.x_configuration('set', [('user.name', 'nobrainer-zoo'),
        #                                     ('user.email', 'nobrainer-zoo')])
    
    # leave the decision to datalad run "datalad get" anyway! 
    datalad.api.get(dataset= model_repo,
                         path= model_path,
                         source="osf-storage")    
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        model_name = str(sys.argv[1])
        path = get_model_path(model_name)
    else:
        model_name = str(sys.argv[1])
        model_type = str(sys.argv[2])
        path = get_model_path(model_name, model_type)
    
    get_model(path)


