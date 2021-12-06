import os
import datalad.api
from utils import get_model_path

def get_model(model_path):
    """
    downloads the model located in model_path.
    
    """ 
    
    # if trained model is not already cloned
    if not os.path.exists("nobrainerzoo/trained-models"):
        url="https://github.com/neuronets/trained-models.git"
        datalad.api.clone(source=url, path="nobrainerzoo/trained-models")
        
    if not os.path.exists(model_path):
        datalad.api.get(dataset="nobrainerzoo/trained-models",
                        path=model_path,
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


