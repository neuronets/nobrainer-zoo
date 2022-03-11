# script to download and update the trained_models repository
from pathlib import Path
import sys
import subprocess as sp
import datalad.api
from utils import get_model_path, get_model_db

def get_model(model_repo, model_path):
    """
    downloads the model weights.
    
    """ 
    models_repo = Path(model_repo)
    
    # if trained model is not already cloned
    if not models_repo.exists():
        raise Exception("model's repository is not available! run 'nobrainer-zoo init' first!")
        
    # TODO: update should be out of this function, now we suppose that model_repo is updated.
    # # run update to make sure repo is updated
    # datalad.api.update(dataset=models_repo, sibling='origin', how='merge')
    
    # # run git config to suprees the config warning in docker images
    # p0=sp.run(["git", "config", "user.name", "nobrainer-zoo"],
    #           stdout=sp.PIPE, stderr=sp.STDOUT, shell=True) #text=True)
    # print(p0.stdout)
    # p1=sp.run(["git", "config", "user.email", "nobrainer-zoo"],
    #           stdout=sp.PIPE, stderr=sp.STDOUT, shell=True) #text=True)
    # print(p1.stdout)
    # # set repo config with datalad api
    # datalad.api.x_configuration('set', [('user.name', 'nobrainer-zoo'),
    #                                     ('user.email', 'nobrainer-zoo')],
    #                              scope='local',)
    
    
    # leave the decision to datalad run "datalad get" anyway! 
    datalad.api.get(dataset= model_repo,
                         path= model_path,
                         source="osf-storage")
   
# to run the script inside the container  
if __name__ == '__main__':
    # this part needs to be updated based on cli.py changes
    # usage: download.py trained_models_path model_name model_type
    trained_models_path = str(sys.argv[1])
    model_db = get_model_db(trained_models_path)
    if len(sys.argv) == 3:
        model_name = str(sys.argv[2])
        model_path = get_model_path(model_db, model_name)
    elif(sys.argv) == 4:
        model_name = str(sys.argv[2])
        model_type = str(sys.argv[3])
        model_path = get_model_path(model_db, model_name, model_type)
    
    get_model(trained_models_path, model_path)


