from zoo.utils import get_model_path
import subprocess as sp
import click
import sys

_option_kwds = {"show_default": True}

@click.group()
def cli():
    """A collection of neuro imaging deep learning models."""
    return

@cli.command()
@click.argument("infile")
@click.argument("outfile")
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="model name. should be in  the form of <org>/<model>/<version>",
    **_option_kwds,
)
@click.option(
    "--model_type",
    default=None,
    type=str,
    help="Type of model for kwyk and braingen model",
    **_option_kwds,
)
@click.option(
    "-b",
    "--block-shape",
    default=(128, 128, 128),
    type=int,
    nargs=3,
    help="Shape of sub-volumes on which to predict.",
    **_option_kwds,
)
@click.option(
    "-r",
    "--resize-features-to",
    default=(256, 256, 256),
    type=int,
    nargs=3,
    help="Resize features to this size before taking blocks and predicting.",
    **_option_kwds,
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.3,
    help=(
        "Threshold used to binarize model output. Only used in binary prediction and"
        " must be in (0, 1)."
    ),
    **_option_kwds,
)
@click.option(
    "-l",
    "--largest-label",
    is_flag=True,
    help=(
        "Zero out all values not connected to the largest contiguous label (not"
        " including 0 values). This remove false positives in binary prediction."
    ),
    **_option_kwds,
)
@click.option(
    "--rotate-and-predict",
    is_flag=True,
    help=(
        "Average the prediction with a prediction on a rotated (and subsequently"
        " un-rotated) volume. This can produce a better overall prediction."
    ),
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress bar.", **_option_kwds
)
def predict(
    *,
    infile,
    outfile,
    model,
    model_type,
    block_shape,
    resize_features_to,
    threshold,
    largest_label,
    rotate_and_predict,
    verbose
):
    """
    get the prediction from model.

    Saves the output file to the path defined by outfile.

    """
    
    # set the docker/singularity image
    org=model.split("/")[0]
    if org=="neuronets":
        img = "env/nobrainer-zoo_test.sif"
    else:
        raise NotImplementedError
        
    #create model_path
    model_path=get_model_path(model, model_type)
    
    if model_type == None:
        cmd0 = ["singularity","run",img,"python3","download.py",model]
    else:
        cmd0 = ["singularity","run",img,"python3","download.py",model,model_type]
    
    # download the model using container
    p0=sp.run(cmd0,stdout=sp.PIPE, stderr=sp.STDOUT ,text=True)
    print(p0.stdout)
         
    # create the command 
    options =["--nv","-B","$(pwd):/data","-W", "/data"]
    # ToDo add generate function for GAN models
    
    nb_command = ["predict"]
    data = [infile,outfile]
        
    model_options=["-b", str(block_shape[0]),str(block_shape[1]),str(block_shape[2]),
                   "-r", str(resize_features_to[0]),str(resize_features_to[1]),str(resize_features_to[2]),
                   "-t", str(threshold)]
    
    # add flag-type options
    if largest_label:
        model_options = model_options+["-l"]
    
    if rotate_and_predict:
        model_options = model_options+["--rotate-and-predict"]
         
    if verbose:
        model_options = model_options+["-v"]
  
                 
    cmd = ["singularity","run"]+options+[img]+["nobrainer"]+nb_command+["-m"]+[model_path]+data+model_options
    
    # run command
    p1 = sp.run(cmd,stdout=sp.PIPE, stderr=sp.STDOUT ,text=True)            
    print(p1.stdout)
    
@cli.command()
def generate():
    """
    Generate output from GAN models.
    """
    click.echo(
        "Not implemented yet. In the future, this command will be used to generate output from GAN models."
    )
    sys.exit(-2)
    
@cli.command()
def register():
    """
    Creates output for brain registration.
    """
    click.echo(
        "Not implemented yet. In the future, this command will be used for brain registration."
    )
    sys.exit(-2)
    
    
    
    
    
# for debugging purposes    
if __name__ == "__main__":
    cli()
    
    