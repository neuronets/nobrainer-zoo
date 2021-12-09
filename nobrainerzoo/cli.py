from nobrainerzoo.utils import get_model_path, get_repo
import subprocess as sp
from pathlib import Path
import click
import yaml
import sys, shutil

_option_kwds = {"show_default": True}


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
    "--options",
    type=str,
    cls=OptionEatAll,
    help="Model-specific options",
    **_option_kwds,
)
def predict(
    infile,
    outfile,
    model,
    model_type,
    container_type,
    options,
    **kwrg
):
    """
    get the prediction from model.

    Saves the output file to the path defined by outfile.

    """
    
    org, model_nm, ver = model.split("/")
    parent_dir = Path(__file__).resolve().parent
    model_dir = parent_dir / model
    spec_file = model_dir / "spec.yaml"
    
    if not model_dir.exists():
        raise Exception("model directory not found!",
                        "This model does not exist in the zoo or didn't properly added.")
    if not spec_file.exists():
        raise Exception("spec file doesn't exist!",
                        "This model does not exist in the zoo or didn't properly added.")
  
    with spec_file.open() as f:
        spec = yaml.safe_load(f)
        
    # set the docker/singularity image    
    image = _container_check(container_type=container_type, image_spec=spec.get("image"))
    
    if container_type == "singularity":
        
        download_image = parent_dir / "env/nobrainer-zoo_nobrainer.sif"
        if not download_image.exists():
            dwnld_cmd = ["singularity", "pull", "--dir", 
                       str(parent_dir/ "env"),
                       "docker://neuronets/nobrainer-zoo:nobrainer"]
            p = sp.run(dwnld_cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
            print(p.stdout)
        cmd0 = ["singularity", "run", download_image, "python3", 
                str(parent_dir/ "download.py"), model]
        
    elif container_type == "docker":
        path = str(parent_dir)+":"+str(parent_dir)
        loader = str(parent_dir / "download.py")
        # check output option
        cmd0 = ["docker", "run","-v",path,"-v",f"{parent_dir}:/output","-w","/output",
                "--rm","neuronets/nobrainer-zoo:nobrainer", 
                "python3", loader, model]
    else:
        raise ValueError(f"unknown container type: {container_type}")
    if model_type:
            cmd0.append(model_type)
            
    # download the model using container
    p0 = sp.run(cmd0, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    print(p0.stdout)

    # download the model repository
    if not org == "neuronets": # neuronet models do not need the repo download
      repo_info = spec.get("repo")
      get_repo(org, repo_info["repo_url"], repo_info["commitish"])
                
    
    data_path = Path(infile).resolve().parent
    out_path = Path(outfile).resolve().parent
    bind_paths = [str(data_path), str(out_path)]
    
    # reading spec file in order to create options for model command
    options_spec = spec.get("options", {})
    
    # create model_path
    # it is used by neuronets models 
    model_path = get_model_path(model, model_type)
    
    # TODO: sould we check if an option is mandatory?
    model_options = []
    if eval("options") is not None:
        val_l = eval(eval("options"))
        val_dict = {}
        for el in  val_l:
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
        cmd_options = ["--nv", "-B", bind_paths, "-B", f"{out_path}:/output", "-W", "/output"]
        cmd = ["singularity", "run"] + cmd_options + [image] + model_cmd.split() \
            + model_options
    elif container_type == "docker":
        bind_paths_docker = []
        for el in bind_paths:
            bind_paths_docker += ["-v", f"{el}:{el}"]
        cmd_options = bind_paths_docker + ["-v", f"{out_path}:/output", "-w", "/output", "--rm"]
        cmd = ["docker", "run"] + cmd_options + [image] + model_cmd.split() \
            + model_options
    else:
        raise ValueError(f"unknown container type: {container_type}")
    
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
@click.argument("data_train_pattern", required=False)
@click.argument("data_evaluate_pattern", required=False)
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="model name. should be in  the form of <org>/<model>",
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
def train(model, spec_file, container_type, n_classes, dataset_train, dataset_test,
          train, network, path, data_train_pattern, data_evaluate_pattern, **kwrg):
    """
    Train the model with specified parameters.
    
    Saves the model weights, checkpoints and training history to the path defined by user.
    
    """
    
    # set the docker/singularity image
    org, model_nm = model.split("/")
    model_dir = Path(__file__).resolve().parents[0] / model
    if spec_file:
        spec_file = Path(spec_file).resolve()
    else:
        # if no spec_file provided we are using the default spec from the model repo
        spec_file = model_dir / f"{model_nm}_train_spec.yaml"

    if not spec_file.exists():
        raise Exception("model directory not found")
    if not spec_file.exists():
        raise Exception("spec file doesn't exist")

    with spec_file.open() as f:
        spec = yaml.safe_load(f)
    
    # updating specification with the argument provided in the command line
    for arg_str in ["n_classes"]:
        if eval(arg_str) is not None:
            spec[arg_str] = eval(arg_str)

    for arg_dict in ["dataset_train", "dataset_test", "train", "network", "path"]:
        if eval(arg_dict) is not None:
            val_l = eval(eval(arg_dict))
            val_dict = {}
            for el in val_l:
                key, val = el.split("=")
                # if the field is in the spec file, the type from the spec
                # is used to convert the value provided from cli
                if key in spec[arg_dict]:
                    tp = type(spec[arg_dict][key])
                    val_dict[key] = tp(val)
                else:
                    val_dict[key] = val
            spec[arg_dict].update(val_dict)
            
    if data_train_pattern and data_evaluate_pattern:
        data_train_path = Path(data_train_pattern).resolve().parent
        spec["data_train_pattern"] = str(Path(data_train_pattern).resolve())
        data_valid_path = Path(data_evaluate_pattern).resolve().parent
        spec["data_valid_pattern"] = str(Path(data_evaluate_pattern).resolve())
        bind_paths = [str(data_train_path), str(data_valid_path)]
    elif data_train_pattern or data_evaluate_pattern:
        raise Exception(f"please provide both data_train_pattern and data_evaluate_pattern,"
                        f" or neither if you want to use the sample data")
    else: # if data_train_pattern not provided, the sample data is used
        data_path = Path(__file__).resolve().parents[0] / "data"
        bind_paths = [str(data_path)]

    out_path = Path(".").resolve().parent

    train_script = model_dir / spec["train_script"]
    bind_paths.append(str(train_script.resolve().parent))

    spec_file_updated = spec_file.parent / "spec_updated.yaml"
    with spec_file_updated.open("w") as f:
        yaml.dump(spec, f)

    cmd = ["python", str(train_script)] + ["-config", str(spec_file_updated)]

    image = _container_check(container_type=container_type, image_spec=spec.get("image"))
    if container_type == "singularity":
        bind_paths = ",".join(bind_paths)
        options = ["--nv", "-B", bind_paths, "-B", f"{out_path}:/output", "-W", "/output"]
        cmd_cont = ["singularity", "run"] + options + [image] + cmd
    elif container_type == "docker":
        bind_paths_docker = []
        for el in bind_paths:
            bind_paths_docker += ["-v", f"{el}:{el}"]
        options = bind_paths_docker + ["-v", f"{out_path}:/output", "-w", "/output", "--rm"]
        cmd_cont = ["docker", "run"] + options + [image] + cmd

    # run command
    print("training the model ........")
    p1 = sp.run(cmd_cont, stdout=sp.PIPE, stderr=sp.STDOUT ,text=True)

    # removing the file with updated spec
    spec_file_updated.unlink()
    print(p1.stdout)



def _container_check(container_type, image_spec, docker_ok=True):
    if image_spec is None:
        raise Exception("image not provided in the specification")
        
    if container_type == "singularity":
        if shutil.which("singularity") is None:
            raise Exception("singularity is not installed")
        if container_type in image_spec:
            image = image_spec[container_type]
            image = Path(__file__).resolve().parents[0] / f"env/{image}"
            if not image.exists():
                # pull image from dockerhub
                print("Container does not exists locally!")
                print("Downloading the container file. it might take a while...")
                pull_image = "docker://"+image_spec["docker"]
                path = Path(__file__).resolve().parents[0] / "env/"
                pull_cmd = ["singularity","pull","--dir", path, pull_image]
                pr = sp.run(pull_cmd, stdout=sp.PIPE, stderr=sp.STDOUT ,text=True)
                print(pr.stdout)
                
        else:
            raise Exception(f"container name for {container_type} is not "
                            f"provided in the specification, "
                            f"try using container_type=docker")
    elif container_type == "docker":
        if not docker_ok:
            raise NotImplementedError("the command is not implemented to work with "
                                      "docker, try singularity")
        if shutil.which("docker") is None or sp.call(["docker", "info"]):
            raise Exception("docker is not installed")
        if container_type in image_spec:
            image = image_spec[container_type]
        else:
            raise Exception(f"container name for {container_type} is not "
                            f"provided in the specification, "
                            f"try using container_type=singularity")

    else:
        raise Exception(f"container_type should be docker or singularity, "
                        f"but {container_type} provided")

    return image


# for debugging purposes    
if __name__ == "__main__":
    cli()