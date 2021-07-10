from nobrainerzoo.utils import get_model_path
import subprocess as sp
from pathlib import Path
import click
import yaml
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
    "--container_type",
    default="singularity",
    type=str,
    help="Type of the container technology (docker or singularity)",
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
    container_type,
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
    
    # TODO download the image if it is not already downloded
    # set the docker/singularity image
    org, model_nm, ver = model.split("/")
    
    model_dir = Path(__file__).resolve().parents[0] / model
    spec_file = model_dir / "spec.yml"

    if not model_dir.exists():
        raise Exception("model directory not found")
    if not spec_file.exists():
        raise Exception("spec file doesn't exist")

    with spec_file.open() as f:
        spec = yaml.safe_load(f)

    if container_type in ["singularity", "docker"]:
        if container_type == "docker": #TODO
            raise NotImplementedError
        if container_type in spec["image"]:
            image = spec["image"][container_type]
        else:
            raise Exception(f"container name for {container_type} is not "
                            f"provided in the specification")
    else:
        raise Exception(f"container_type should be docker or singularity, "
                        f"but {container_type} provided")

    img = Path(__file__).resolve().parents[0] / f"env/{image}"
    if not img.exists():
        # TODO: we should catch the error and try to download the image
        raise FileNotFoundError(f"the {image} can't be found")


    #create model_path
    model_path = get_model_path(model, model_type)

    cmd0 = ["singularity", "run", img, "python3", "nobrainerzoo/download.py", model]
    if model_type:
        cmd0.append(model_type)

    # download the model using container
    p0 = sp.run(cmd0, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    print(p0.stdout)

    # create the command 
    data_path = Path(infile).parent
    out_path = Path(outfile).parent

    # TODO: this will work for singularity only
    options =["--nv", "-B", str(data_path), "-B", f"{out_path}:/output", "-W", "/output"]

    model_options = ["-b"] + [str(el) for el in block_shape] \
                  + ["-r"] + [str(el) for el in resize_features_to] \
                  + ["-t", str(threshold)]
    
    # add flag-type options
    if largest_label:
        model_options = model_options + ["-l"]
    
    if rotate_and_predict:
        model_options = model_options + ["--rotate-and-predict"]
         
    if verbose:
        model_options = model_options + ["-v"]
  
    # TODO command should be taken from the spec
    cmd = ["singularity","run"] + options + [img, "nobrainer", "predict"]\
        + ["-m"] + [model_path, infile, outfile] + model_options

    # run command
    p1 = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT ,text=True)
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
    
    