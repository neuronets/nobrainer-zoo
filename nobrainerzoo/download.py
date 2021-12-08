from pathlib import Path
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
        # # set repo config
        # datalad.api.x_configuration('set', [('user.name', 'nobrainer-zoo'),
        #                                     ('user.name', 'nobrainer-zoo')])
        
    if not Path(model_path).exists():
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


