# script to download and update the trained_models repository
from pathlib import Path
import sys

# import subprocess as sp
import datalad.api
import requests


def update_model_db(db_path):
    """Runs datalad update on the model repository."""
    # check the internet connection
    url = "https://github.com/neuronets/trained-models.git"
    timeout = 5
    try:
        requests.get(url, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout):
        print(
            "You are not connected to the internet! check your connectivity and try again."
        )
        raise

    # if the internet connection is Ok run update
    datalad.api.update(dataset=db_path, sibling="origin", merge=True)


def get_model(model_repo, model_path):
    """
    downloads the model weights.

    """
    models_repo = Path(model_repo)

    # if trained model is not already cloned
    if not models_repo.exists():
        raise Exception(
            "model database is not available! run 'nobrainer-zoo init' first!"
        )

    update_model_db(model_repo)

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
    datalad.api.get(dataset=model_repo, path=model_path, source="osf-storage")


# to run the script inside the container
if __name__ == "__main__":

    model_db_path = str(sys.argv[1])
    model_path = str(sys.argv[2])
    get_model(model_db_path, model_path)
