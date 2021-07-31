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

    inputs_spec = spec.get("inputs", {})

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

    # reading spec file in order to create oprion for model command
    model_options = []
    for name, in_spec in inputs_spec.items():
        argstr = in_spec.get("argstr", "")
        value = eval(name)
        if in_spec.get("is_flag"):
            if value:
                model_options.append(argstr)
            continue
        elif argstr:
            model_options.append(argstr)

        if in_spec.get("type") == "list":
            model_options.extend([str(el) for el in value])
        else:
            model_options.append(str(value))

    # reading command from the spec file (allowing for f-string)
    try:
        model_cmd = eval(spec["command"])
    except NameError:
        model_cmd = spec["command"]

    cmd = ["singularity","run"] + options + [img] + model_cmd.split() \
        + model_options

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

@cli.command()
@click.argument("data_train_pattern")
@click.argument("data_evaluate_pattern")
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="model name. should be in  the form of <org>/<model>",
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
    "-v",
    "--volume-shape",
    default=256,
    type=int,
    nargs=1,
    help="Shape of volumes for training data.",
    **_option_kwds,
)
@click.option(
    "-b",
    "--block-shape",
    default=32,
    type=int,
    nargs=1,
    help="Shape of sub-volumes for training data.",
    **_option_kwds,
)
@click.option(
    "--n-classes",
    default=1,
    type=int,
    nargs=1,
    help="Number of classes in labels",
    **_option_kwds,
)
@click.option(
    "--shuffle_buffer_size",
    default=10,
    type=int,
    nargs=1,
    help="Value to fill the buffer for shuffling",
    **_option_kwds,
)
@click.option(
    "--batch_size",
    default=2,
    type=int,
    nargs=1,
    help="Size of batches",
    **_option_kwds,
)
@click.option(
    "--augment",
    default=False,
    is_flag=True,
    help="Apply augmentation to data",
    **_option_kwds,
)
@click.option(
    "--n_train",
    default= None,
    type=int,
    nargs=1,
    help="Number of train samples",
    **_option_kwds,
)
@click.option(
    "--n_valid",
    default= None,
    type=int,
    nargs=1,
    help="Number of validation samples",
    **_option_kwds,
)
@click.option(
    "--num_parallel_calls",
    default= 1,
    type=int,
    nargs=1,
    help="Number of parallel calls",
    **_option_kwds,
)
@click.option(
    "--batchnorm",
    default=True,
    is_flag=True,
    help="Apply batch normalization",
    **_option_kwds,
)
@click.option(
    "--n_epochs",
    default= 1,
    type=int,
    nargs=1,
    help="Number of epochs for training",
    **_option_kwds,
)
@click.option(
    "--lr",
    default= 0.00001,
    type=float,
    nargs=1,
    help="Value for learning rate",
    **_option_kwds,
)
@click.option(
    "--loss",
    type=str,
    required=True,
    help="Loss function",
    **_option_kwds,
    )
@click.option(
    "--metrics",
    type=list,
    required=True,
    help="list of metrics",
    **_option_kwds,
    )
@click.option(
    "--check_point_model",
    type=str,
    help="Path to save training checkpoints",
    **_option_kwds,
    )
@click.option(
    "--save_history",
    type=str,
    help="Path to save training results",
    **_option_kwds,
    )
@click.option(
    "--save_model",
    type=str,
    required=True,
    help="Path to save model weights",
    **_option_kwds,
    )
def train(
    *,
    data_train_pattern,
    data_evaluate_pattern,
    model,
    container_type,
    volume_shape,
    block_shape,
    n_classes,
    shuffle_buffer_size,
    batch_size,
    augment,
    batchnorm,
    n_epochs,
    lr,
    loss,
    metrics,
    check_point_model,
    save_history,
    save_model,
):
    """
    Train the model with specified parameters.
    
    Saves the model weights, checkpoints and training history to the path defined by user.
    
    """
    # TODO download the image if it is not already downloded
    # set the docker/singularity image
    org, model_nm = model.split("/")
    
    model_dir = Path(__file__).resolve().parents[0] / model
    spec_file = model_dir / f"{model_nm}_train_spec.yaml"
    
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
                
    # create the command 
    data_train_path = Path(data_train_pattern).parent
    data_valid_path = Path(data_evaluate_pattern).parent    
    out_path = Path(save_model).parent
    
    bind_paths = f"{data_train_path},{data_valid_path},{out_path}"
    
    # TODO: this will work for singularity only
    options =["--nv", "-B", bind_paths, "-B", f"{out_path}:/output", "-W", "/output"]
    
    train_script = model_dir / spec["train_script"]
    
    train_options = [data_train_pattern,
                     data_evaluate_pattern,
                     volume_shape,
                     block_shape,
                     n_classes,
                     shuffle_buffer_size,
                     batch_size,
                     augment,
                     batchnorm,
                     n_epochs,
                     lr,
                     loss,
                     metrics,
                     check_point_model,
                     save_history,
                     save_model
                     ]
    
    cmd = ["singularity","run"] + options + [img] + [str(train_script)] + train_options

    # run command
    p1 = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT ,text=True)
    print(p1.stdout)
        
    
# for debugging purposes    
if __name__ == "__main__":
    
    cli()
    
    