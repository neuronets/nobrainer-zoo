import os
from pathlib import Path
import shutil
import subprocess as sp
import click
import yaml

from .utils import (
    CACHE_PATH,
    MODELS_PATH,
    IMAGES_PATH,
    get_model_db,
    get_model_path,
    get_repo,
    get_spec,
    pull_singularity_image,
    _get_model_file,
)

_option_kwds = {"show_default": True}

DATA_PATH = CACHE_PATH / "data"
REPO_PATH = CACHE_PATH / "repo"


# https://stackoverflow.com/a/48394004/5666087
class OptionEatAll(click.Option):
    """Subclass of `click.Option` that allows for an arbitrary number of options.
    The behavior is similar to `nargs="*"` in argparse.
    """

    def __init__(self, *args, **kwargs):
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            # grab everything up to the next option
            while state.rargs and not done:
                for prefix in self._eat_all_parser.prefixes:
                    if state.rargs[0].startswith(prefix):
                        done = True
                if not done:
                    value.append(state.rargs.pop(0))
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


@click.group()
def cli():
    """A collection of neuroimaging deep learning models."""
    return


@cli.command()
@click.option("-c", "--cache", default=CACHE_PATH)
def init(cache):
    """Initialize ..."""
    # TODO add a clean cache option
    # TODO add a show cache_path option

    cache = Path(cache).resolve()
    global CACHE_PATH

    if "NOBRAINER_CACHE" in os.environ:
        CACHE_PATH = Path(os.environ["NOBRAINER_CACHE"]).resolve() / ".nobrainer"
    elif not cache.samefile(CACHE_PATH):
        CACHE_PATH = cache / ".nobrainer"

    # redefine global variables
    global MODELS_PATH
    global IMAGES_PATH
    global DATA_PATH
    global REPO_PATH
    MODELS_PATH = CACHE_PATH / "trained-models"
    IMAGES_PATH = CACHE_PATH / "images"
    DATA_PATH = CACHE_PATH / "data"
    REPO_PATH = CACHE_PATH / "repo"

    print(
        f"Creating a cache directory in {CACHE_PATH}\n"
        "you can change the cache location by setting the environmental variable `NOBRAINER_CACHE`.\n"
        "run 'export NOBRAINER_CACHE=<path_to_your_location>'.\n"
        "or by the 'nobrainer-zoo init --cache/-c <path_to_your_location>'.\n"
        "Note that NOBRAINER_CACHE variable overrides the --cache/-c option."
    )

    os.makedirs(CACHE_PATH, exist_ok=True)
    # create subdirectory for images, data
    os.makedirs(IMAGES_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    # adding trained_model repository
    model_db_url = "https://github.com/neuronets/trained-models"
    if _container_installed("singularity"):
        # pull the nobrainer image from docker-hub
        download_image = IMAGES_PATH / "nobrainer-zoo_zoo.sif"
        if not download_image.exists():
            dwnld_cmd = [
                "singularity",
                "pull",
                "--dir",
                str(IMAGES_PATH),
                "docker://neuronets/nobrainer-zoo:zoo",
            ]
            p0 = sp.run(dwnld_cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
            print(p0.stdout)
        # For a robust behavior of model_db, we should clone via datalad.
        clone_cmd = [
            "singularity",
            "run",
            "-e",
            "-B",
            CACHE_PATH,
            download_image,
            "datalad",
            "clone",
            model_db_url,
            MODELS_PATH,
        ]
    elif _container_installed("docker"):
        # check output option
        clone_cmd = [
            "docker",
            "run",
            "-v",
            f"{CACHE_PATH}:/cache_dir",
            "-w",
            "/cache_dir",
            "--rm",
            "neuronets/nobrainer-zoo:zoo",
            "datalad",
            "clone",
            model_db_url,
            "/cache_dir/trained-models",
        ]
    else:
        # neither singularity nor docker is found!
        raise Exception(
            "Neither singularuty or docker is installed!",
            "Please install singularity or docker and run 'nobrainer-zoo init' again.",
        )

    if not MODELS_PATH.exists():
        p1 = sp.run(clone_cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        print(p1.stdout)
    # else:
    # update the model_db


@cli.command()
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="model name. should be in  the form of <org>/<model>/<version>",
    **_option_kwds,
)
@click.option(
    "--model_type",
    default=None,
    type=str,
    help="Type of model if there is more than one.",
    **_option_kwds,
)
def ls(model, model_type):
    """lists available models with versions and organizations or
    prints the information about the model given by -m/--model."""

    if not model:

        if not MODELS_PATH.exists():
            raise ValueError(
                "Model's database does not exists. please run 'nobrainer-zoo init'."
            )

        # TODO: update the trained-models repo

        _ = get_model_db(MODELS_PATH)

    else:
        spec = get_spec(model, model_type)

        model_info = spec.get("model", {})
        if not model_info:
            raise Exception("Help is not available for this model.")
        else:
            for key, value in model_info.items():
                print(key + ":")
                print(value, "\n")


@cli.command()
@click.argument("infile", nargs=-1)
@click.argument("outfile", nargs=1)
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
    "--cpu",
    is_flag=True,
    help="force the Nobrainer-zoo to use the cpu when gpu is available",
    **_option_kwds,
)
@click.option(
    "--options",
    type=str,
    cls=OptionEatAll,
    help="Model-specific options",
    **_option_kwds,
)
def predict(infile, outfile, model, model_type, container_type, cpu, options, **kwrg):
    """
    get the prediction from model.

    Saves the output file to the path defined by 'outfile'.

    """

    org, model_nm, ver = model.split("/")
    parent_dir = Path(__file__).resolve().parent

    if cpu:
        cuda_device = dict(os.environ, **{"CUDA_VISIBLE_DEVICES": "-1"})
    else:
        cuda_device = None

    # get the model database
    model_db = get_model_db(MODELS_PATH, print_models=False)

    spec = get_spec(model, model_type)

    # download the container image and set the path
    image = _container_check(
        container_type=container_type, image_spec=spec.get("image")
    )

    model_path = Path(get_model_path(model_db, model, model_type=model_type))
    if model_path.is_dir():
        model_avail = model_path / "saved_model.pb"
    else:
        model_avail = model_path

    if not model_avail.exists():
        # get the model file
        _get_model_file(model_path, container_type=container_type, )

    # download the model repository if needed
    if spec["repository"]["repo_download"]:
        repo_info = spec.get("repository")
        repo_dest = REPO_PATH / f"{model_nm}-{ver}"
        get_repo(repo_info["repo_url"], repo_dest, repo_info["committish"])

    spec = spec["inference"]
    # check the input data
    data_path = _check_input(_name(infile=infile), infile, spec)
    out_path = Path(outfile).resolve().parent
    bind_paths = data_path + [str(out_path)] + [str(CACHE_PATH)]
    # reading spec file in order to create options for model command
    options_spec = spec.get("options", {})

    model_options = []
    if eval("options") is not None:
        val_l = eval(eval("options"))
        val_dict = {}
        for el in val_l:
            if "=" in el:
                key, val = el.split("=")
                val_dict[key] = val
            else:
                val_dict[el] = None

        # updating command with the argument provided in the command line
        for name, in_spec in options_spec.items():
            if name in val_dict.keys():
                argstr = in_spec.get("argstr", "")
                value = val_dict[name]
                if in_spec.get("is_flag"):
                    model_options.append(argstr)
                    continue
                elif argstr:
                    model_options.append(argstr)

                if in_spec.get("type") == "list":
                    model_options.extend([str(el) for el in eval(value)])
                else:
                    model_options.append(str(value))

    # reading command from the spec file (allowing for f-string)
    try:
        model_cmd = eval(spec["command"])
    except NameError:
        model_cmd = spec["command"]
    # breakpoint()
    if container_type == "singularity":
        bind_paths = ",".join(bind_paths)
        cmd_options = [
            # "-e",
            "--nv",
            "-B",
            bind_paths,
            "-B",
            f"{out_path}:/output",
            "-W",
            "/output",
        ]
        cmd = (
                ["singularity", "run"]
                + cmd_options
                + [image]
                + model_cmd.split()
                + model_options
        )
    elif container_type == "docker":
        bind_paths_docker = []
        for el in bind_paths:
            bind_paths_docker += ["-v", f"{el}:{el}"]
        cmd_options = bind_paths_docker + [
            "-v",
            f"{out_path}:/output",
            "-w",
            "/output",
            "--rm",
        ]
        if cpu:
            docker_cmd = ["docker", "run"]
        else:
            docker_cmd = ["docker", "run", "--gpus", "all"]
        cmd = (
                docker_cmd
                + cmd_options
                + [image]
                + model_cmd.split()
                + model_options
        )
    else:
        raise ValueError(f"unknown container type: {container_type}")

    # run command
    p1 = sp.run(cmd,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                env=cuda_device)
    print(p1.stdout)


@cli.command()
@click.argument("outfile", nargs=1)
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
    "--cpu",
    is_flag=True,
    help="force the Nobrainer-zoo to use the cpu when gpu is available",
    **_option_kwds,
)
@click.option(
    "--options",
    type=str,
    cls=OptionEatAll,
    help="Model-specific options",
    **_option_kwds,
)
def generate(outfile, model, model_type, container_type, cpu, options, **kwrg):
    """
    Generate output from GAN models.
    """
    org, model_nm, ver = model.split("/")
    parent_dir = Path(__file__).resolve().parent

    if cpu:
        cuda_device = dict(os.environ, **{"CUDA_VISIBLE_DEVICES": "-1"})
    else:
        cuda_device = None

    # get the model database
    model_db = get_model_db(MODELS_PATH, print_models=False)

    spec = get_spec(model, model_type)

    # download the model-required docker/singularity image and set the path
    image = _container_check(
        container_type=container_type, image_spec=spec.get("image")
    )

    model_path = Path(get_model_path(model_db, model, model_type=model_type))
    if model_path.is_dir():
        model_avail = model_path / "saved_model.pb"
    else:
        model_avail = model_path

    if not model_avail.exists():
        # get the model file
        _get_model_file(model_path, container_type=container_type, )

    # download the model repository if needed
    if spec["repository"]["repo_download"]:
        repo_info = spec.get("repository")
        repo_dest = REPO_PATH / f"{model_nm}-{ver}"
        get_repo(repo_info["repo_url"], repo_dest, repo_info["committish"])

    spec = spec["inference"]
    # check the input data
    # data_path = _check_input(_name(infile=infile), infile, spec)
    out_path = Path(outfile).resolve().parent
    bind_paths = [str(out_path)] + [str(CACHE_PATH)]
    # reading spec file in order to create options for model command
    options_spec = spec.get("options", {})

    model_options = []
    if eval("options") is not None:
        val_l = eval(eval("options"))
        val_dict = {}
        for el in val_l:
            if "=" in el:
                key, val = el.split("=")
                val_dict[key] = val
            else:
                val_dict[el] = None

        # updating command with the argument provided in the command line
        for name, in_spec in options_spec.items():
            if name in val_dict.keys():
                argstr = in_spec.get("argstr", "")
                value = val_dict[name]
                if in_spec.get("is_flag"):
                    model_options.append(argstr)
                    continue
                elif argstr:
                    model_options.append(argstr)

                if in_spec.get("type") == "list":
                    model_options.extend([str(el) for el in eval(value)])
                else:
                    model_options.append(str(value))

    # reading command from the spec file (allowing for f-string)
    try:
        model_cmd = eval(spec["command"])
    except NameError:
        model_cmd = spec["command"]

    if container_type == "singularity":
        bind_paths = ",".join(bind_paths)
        cmd_options = [
            "-e",
            "--nv",
            "-B",
            bind_paths,
            "-B",
            f"{out_path}:/output",
            "-W",
            "/output",
        ]
        cmd = (
                ["singularity", "run"]
                + cmd_options
                + [image]
                + model_cmd.split()
                + model_options
        )
    elif container_type == "docker":
        bind_paths_docker = []
        for el in bind_paths:
            bind_paths_docker += ["-v", f"{el}:{el}"]
        cmd_options = bind_paths_docker + [
            "-v",
            f"{out_path}:/output",
            "-w",
            "/output",
            "--rm",
        ]
        if cpu:
            docker_cmd = ["docker", "run"]
        else:
            docker_cmd = ["docker", "run", "--gpus", "all"]
        cmd = (
                docker_cmd
                + cmd_options
                + [image]
                + model_cmd.split()
                + model_options
        )
    else:
        raise ValueError(f"unknown container type: {container_type}")

    # run command
    p1 = sp.run(cmd,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                env=cuda_device)
    print(p1.stdout)


@cli.command()
@click.argument("moving", nargs=1, type=click.Path(exists=True))
@click.argument("fixed", nargs=1, type=click.Path(exists=True))
@click.argument("moved", nargs=1, type=click.Path())
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
    help="Type of model if there is more than one.",
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
    "--cpu",
    is_flag=True,
    help="force the Nobrainer-zoo to use the cpu when gpu is available",
    **_option_kwds,
)
@click.option(
    "--options",
    type=str,
    cls=OptionEatAll,
    help="Model-specific options",
    **_option_kwds,
)
def register(moving, fixed, moved, model, model_type, container_type, cpu, options, **kwrg):
    """
    registers the MOVING image to the FIXED image.

    output saves in the path defined by MOVED argument.

    """

    org, model_nm, ver = model.split("/")
    parent_dir = Path(__file__).resolve().parent

    if cpu:
        cuda_device = dict(os.environ, **{"CUDA_VISIBLE_DEVICES": "-1"})
    else:
        cuda_device = None

    # get the model database
    model_db = get_model_db(MODELS_PATH, print_models=False)

    spec = get_spec(model, model_type)

    # set the docker/singularity image
    image = _container_check(
        container_type=container_type, image_spec=spec.get("image")
    )

    model_path = Path(get_model_path(model_db, model, model_type=model_type))
    if model_path.is_dir():
        model_avail = model_path / "saved_model.pb"
    else:
        model_avail = model_path

    if not model_avail.exists():
        # get the model file
        _get_model_file(model_path, container_type=container_type, )

    # download the model repository if needed
    if spec["repository"]["repo_download"]:
        repo_info = spec.get("repository")
        repo_dest = REPO_PATH / f"{model_nm}-{ver}"
        get_repo(repo_info["repo_url"], repo_dest, repo_info["committish"])

    spec = spec["inference"]
    # check the input variables
    moving_path = _check_input(_name(moving=moving), moving, spec)
    fixed_path = _check_input(_name(fixed=fixed), fixed, spec)
    out_path = Path(moved).resolve().parent
    bind_paths = moving_path + fixed_path + [str(out_path)] + [str(CACHE_PATH)]

    # reading spec file in order to create options for model command
    options_spec = spec.get("options", {})

    model_options = []
    if eval("options") is not None:
        val_l = eval(eval("options"))
        val_dict = {}
        for el in val_l:
            if "=" in el:
                key, val = el.split("=")
                val_dict[key] = val
            else:
                val_dict[el] = None

        # updating command with the argument provided in the command line
        for name, in_spec in options_spec.items():
            if name in val_dict.keys():
                argstr = in_spec.get("argstr", "")
                value = val_dict[name]
                if in_spec.get("is_flag"):
                    model_options.append(argstr)
                    continue
                elif argstr:
                    model_options.append(argstr)

                if in_spec.get("type") == "list":
                    model_options.extend([str(el) for el in eval(value)])
                else:
                    model_options.append(str(value))

    # reading command from the spec file (allowing for f-string)
    try:
        model_cmd = eval(spec["command"])
    except NameError:
        model_cmd = spec["command"]

    if container_type == "singularity":
        bind_paths = ",".join(bind_paths)
        cmd_options = [
            "-e",
            "--nv",
            "-B",
            bind_paths,
            "-B",
            f"{out_path}:/output",
            "-W",
            "/output",
        ]
        cmd = (
                ["singularity", "run"]
                + cmd_options
                + [image]
                + model_cmd.split()
                + model_options
        )
    elif container_type == "docker":
        bind_paths_docker = []
        for el in bind_paths:
            bind_paths_docker += ["-v", f"{el}:{el}"]
        cmd_options = bind_paths_docker + [
            "-v",
            f"{out_path}:/output",
            "-w",
            "/output",
            "--rm",
        ]
        if cpu:
            docker_cmd = ["docker", "run"]
        else:
            docker_cmd = ["docker", "run", "--gpus", "all"]
        cmd = (
                docker_cmd
                + cmd_options
                + [image]
                + model_cmd.split()
                + model_options
        )
    else:
        raise ValueError(f"unknown container type: {container_type}")

    # run command
    p1 = sp.run(cmd,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                env=cuda_device)
    print(p1.stdout)


@cli.command()
@click.argument("data_train_pattern", required=False)
@click.argument("data_evaluate_pattern", required=False)
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="model name. should be in  the form of <org>/<model>/<version>",
    **_option_kwds,
)
@click.option(
    "--spec_file",
    type=str,
    help="path to a spec file (if not set, the default spec is used)",
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
    "--n-classes",
    type=int,
    nargs=1,
    help="Number of classes in labels",
    **_option_kwds,
)
@click.option(
    "--dataset_train",
    type=str,
    cls=OptionEatAll,
    help="info about training dataset",
    **_option_kwds,
)
@click.option(
    "--dataset_test",
    type=str,
    cls=OptionEatAll,
    help="info about testing dataset",
    **_option_kwds,
)
@click.option(
    "--train",
    type=str,
    cls=OptionEatAll,
    help="training options",
    **_option_kwds,
)
@click.option(
    "--network",
    type=str,
    cls=OptionEatAll,
    help="network options",
    **_option_kwds,
)
@click.option(
    "--path",
    type=str,
    cls=OptionEatAll,
    help="paths for saving results",
    **_option_kwds,
)
def fit(
        model,
        spec_file,
        container_type,
        n_classes,
        dataset_train,
        dataset_test,
        train,
        network,
        path,
        data_train_pattern,
        data_evaluate_pattern,
        **kwrg,
):
    """
    Train the model with specified parameters.

    Saves the model weights, checkpoints and training history to the path defined by user.

    """

    # set the docker/singularity image
    org, model_nm, ver = model.split("/")

    if spec_file:
        spec_file = Path(spec_file).resolve()
    else:
        # if no spec_file provided we are using the default spec from the model repo
        spec_file = MODELS_PATH / model / "spec.yaml"

    if not spec_file.exists():
        raise Exception("model directory not found")
    if not spec_file.exists():
        raise Exception("spec file doesn't exist")

    with spec_file.open() as f:
        spec = yaml.safe_load(f)

    train_spec = spec.get("train")

    # updating specification with the argument provided in the command line
    for arg_str in ["n_classes"]:
        if eval(arg_str) is not None:
            train_spec[arg_str] = eval(arg_str)

    for arg_dict in ["dataset_train", "dataset_test", "training", "network", "path"]:
        if eval(arg_dict) is not None:
            val_l = eval(eval(arg_dict))
            val_dict = {}
            for el in val_l:
                key, val = el.split("=")
                # if the field is in the spec file, the type from the spec
                # is used to convert the value provided from cli
                if key in train_spec[arg_dict]:
                    tp = type(train_spec[arg_dict][key])
                    val_dict[key] = tp(val)
                else:
                    val_dict[key] = val
            train_spec[arg_dict].update(val_dict)
        elif arg_dict == "path":
            # set the cache dir to save the model
            save_dir = DATA_PATH / f"{model_nm}-{ver}" / "model"
            train_spec["path"]["save_model"] = str(save_dir)

    if data_train_pattern and data_evaluate_pattern:
        data_train_path = Path(data_train_pattern).resolve().parent
        train_spec["data_train_pattern"] = str(Path(data_train_pattern).resolve())
        data_valid_path = Path(data_evaluate_pattern).resolve().parent
        train_spec["data_valid_pattern"] = str(Path(data_evaluate_pattern).resolve())
        bind_paths = [str(data_train_path), str(data_valid_path)]
    elif data_train_pattern or data_evaluate_pattern:
        raise Exception(
            "please provide both data_train_pattern and data_evaluate_pattern",
            " or neither if you want to use the sample data",
        )
    else:  # if data_train_pattern not provided, the sample data is used
        data_train_path = DATA_PATH / f"{model_nm}-{ver}"
        train_spec["dataset_train"]["data_location"] = str(data_train_path)
        data_valid_path = DATA_PATH / f"{model_nm}-{ver}"
        train_spec["dataset_test"]["data_location"] = str(data_valid_path)
        bind_paths = [str(DATA_PATH)]

    # out_path = Path(".").resolve().parent
    out_path = DATA_PATH / f"{model_nm}-{ver}"

    train_script = MODELS_PATH / model / "train.py"
    bind_paths.append(str(train_script.resolve().parent))

    spec_file_updated = spec_file.parent / "spec_updated.yaml"
    with spec_file_updated.open("w") as f:
        yaml.dump(train_spec, f)

    cmd = ["python", str(train_script)] + ["-config", str(spec_file_updated)]

    image = _container_check(
        container_type=container_type, image_spec=spec.get("image")
    )
    if container_type == "singularity":
        bind_paths = ",".join(bind_paths)
        options = [
            "-e",
            "--nv",
            "-B",
            bind_paths,
            "-B",
            f"{out_path}:/output",
            "-W",
            "/output",
        ]
        cmd_cont = ["singularity", "run"] + options + [image] + cmd
    elif container_type == "docker":
        bind_paths_docker = []
        for el in bind_paths:
            bind_paths_docker += ["-v", f"{el}:{el}"]
        options = bind_paths_docker + [
            "-v",
            f"{out_path}:/output",
            "-w",
            "/output",
            "--rm",
        ]
        cmd_cont = ["docker", "run"] + options + [image] + cmd

    # run command
    print("training the model ........")
    p1 = sp.run(cmd_cont, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)

    # removing the file with updated spec
    spec_file_updated.unlink()
    print(p1.stdout)


def _container_check(container_type, image_spec):
    """downloads the container image and returns its path"""
    if image_spec is None:
        raise Exception("image not provided in the specification")

    # check the installation of singularity or docker
    if not _container_installed(container_type):
        raise Exception(f"{container_type} is not installed.")

    if container_type == "singularity":
        if container_type in image_spec:
            pull_singularity_image(image_spec[container_type], IMAGES_PATH)
            image = IMAGES_PATH / image_spec[container_type]
        else:
            raise Exception(
                f"container name for {container_type} is not "
                f"provided in the specification, "
                f"try using container_type=docker"
            )
    elif container_type == "docker":
        if container_type in image_spec:
            image = image_spec[container_type]
        else:
            raise Exception(
                f"container name for {container_type} is not "
                f"provided in the specification, "
                f"try using container_type=singularity"
            )

    else:
        raise Exception(
            f"container_type should be docker or singularity, "
            f"but {container_type} provided"
        )

    return image


def _check_input(infile_name, infile, spec):
    """Checks the infile path and returns the binding path for the container"""
    # TODO: check if the infile is a dir
    if isinstance(infile, tuple):
        n_infile = len(infile)
    else:
        n_infile = 1
        infile = (infile,)
    n_inputs = spec["data_spec"][f"{infile_name}"]["n_files"]
    if n_inputs != "any" and n_infile != n_inputs:
        raise ValueError(
            f"This model needs {n_inputs} input files but {n_infile} files are given."
        )

    return [str(Path(file).resolve().parent) for file in infile]


def _name(**variables):
    """Extracts the variable name.
    Usage: _name(variables=variables)
    """
    # https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
    return [x for x in variables][0]


def _container_installed(container_type):
    """checks singularity or docker is installed."""

    if container_type == "singularity":
        if shutil.which("singularity") is None:
            return False
        else:
            return True

    elif container_type == "docker":
        if shutil.which("docker") is None or sp.call(["docker", "info"]):
            return False
        else:
            return True
    else:
        raise Exception(
            f"container_type should be docker or singularity, "
            f"but {container_type} is provided."
        )


# for debugging purposes
if __name__ == "__main__":
    cli()
