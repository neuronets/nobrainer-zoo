from nobrainerzoo.utils import get_model_path, get_repo, get_model_db
from nobrainerzoo.utils import pull_singularity_image
import subprocess as sp
from pathlib import Path
import click
import yaml
import os, sys, shutil

_option_kwds = {"show_default": True}

if "NOBRAINER_CACHE" in os.environ:
    CACHE_PATH = Path(os.environ["NOBRAINER_CACHE"]).resolve() / ".nobrainer"
else:
    CACHE_PATH = Path(os.path.expanduser('~')) / ".nobrainer"
MODELS_PATH = CACHE_PATH / "trained-models"
IMAGES_PATH = CACHE_PATH / "images"
DATA_PATH = CACHE_PATH / "data"


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
    # get the model database
    model_db = get_model_db(MODELS_PATH, print_models=False)
    # check model type
    _check_model_type(model, model_type)
    
    if model_type:
        model_dir = MODELS_PATH / model / model_type
    else:
        model_dir = MODELS_PATH / model
        
    spec_file = model_dir / "spec.yaml"
    
    if not model_dir.exists():
        raise Exception("model directory not found!",
                        "This model does not exist in the zoo or didn't properly added.")
    if not spec_file.exists():
        raise Exception("spec file doesn't exist!",
                        "This model does not exist in the zoo or didn't properly added.")
      
    with spec_file.open() as f:
        spec = yaml.safe_load(f)
        
    # download the model-required docker/singularity image and set the path
    image = _container_check(container_type=container_type, image_spec=spec.get("image"))
    
    model_path = get_model_path(model_db, model, model_type=model_type)
    # get the model file
    if not Path(model_path).exists():
        loader = str(parent_dir / "download.py")
        if container_type == "singularity":
            download_image = IMAGES_PATH / "nobrainer-zoo_nobrainer.sif"
            if not download_image.exists():
                raise Exception("'nobrainer' singularity image is missing! ",
                                "Please run 'nobrainer-zoo init'.")
            
            # mount CACHE_PATH to /cache_dir, I will be using that path in some functions
            cmd0 = ["singularity", "run",
                    "-B", str(CACHE_PATH),
                    "-B", f"{CACHE_PATH}:/cache_dir",
                    download_image, "python3",
                    loader, MODELS_PATH, model_path]
                    #str( parent_dir / "download.py"), "/cache_dir/trained-models", model]    
        elif container_type == "docker":
            path = str(parent_dir)+":"+str(parent_dir)
            # check output option
            cmd0 = ["docker", "run","-v", path, 
                    "-v", f"{CACHE_PATH}:{CACHE_PATH}",
                    "-w", f"{MODELS_PATH}",
                    "--rm", "neuronets/nobrainer-zoo:nobrainer", 
                    "python3", loader, f"{MODELS_PATH}", model_path]
        else:
            raise ValueError(f"unknown container type: {container_type}")
    
        # download the model using container
        p0 = sp.run(cmd0, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        # TODO: we should be catching the errors (instead of only printing)
        print(p0.stdout)
    
    # download the model repository if needed
    if spec["repository"]["repo_download"]:
        repo_info = spec.get("repository")
        # UCL organization has separate repositories for different models
        # if org == "UCL":  
        #     org = org + "/" + model_nm
        # repo_dest = CACHE_PATH / org / "org_repo"
        repo_dest = CACHE_PATH
        get_repo(repo_info["repo_url"], repo_dest, repo_info["commitish"])
           
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
def init():
    """ Initialize ..."""
    print(f"Creating a cache directory in {CACHE_PATH}, if you want " 
          "to change the location you can point environmental variable  NOBRAINER_CACHE "
          "to the location where .nobrainer directory will be created. "
          "run 'export NOBRAINER_CACHE=<path_to_your_location>")

    os.makedirs(CACHE_PATH, exist_ok=True)
    #create subdirectory for images, data
    os.makedirs(IMAGES_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    # adding trained_model repository
    model_db_url = "https://github.com/neuronets/trained-models"
    if _container_installed("singularity"):
        # pull the nobrainer image from docker-hub
        download_image = IMAGES_PATH / "nobrainer-zoo_nobrainer.sif"
        if not download_image.exists():
            dwnld_cmd = ["singularity", "pull", "--dir", 
                         str(IMAGES_PATH),
                       "docker://neuronets/nobrainer-zoo:nobrainer"]
            p0 = sp.run(dwnld_cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
            print(p0.stdout)
        # For a robust behavior of model_db, we should clone via datalad.
        clone_cmd = ["singularity", "run", "-B", CACHE_PATH, download_image, "datalad",
                     "clone", model_db_url, MODELS_PATH]
        
    elif _container_installed("docker"):
        # check output option
        clone_cmd = ["docker", "run", "-v", f"{CACHE_PATH}:/cache_dir",
                     "-w", "/cache_dir",
                     "--rm", "neuronets/nobrainer-zoo:nobrainer", 
                     "datalad", "clone", model_db_url, "/cache_dir/trained-models"]
    else:
        # neither singularity or docker is found!
        raise Exception("Neither singularuty or docker is installed!",
                        "Please install singularity or docker and run 'nobrainer-zoo init' again.")
        
    if not MODELS_PATH.exists():
        p1 = sp.run(clone_cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        print(p1.stdout)    
    #else:
        # update the model_db

@cli.command()        
def ls():
    """lists available models with versions and organizations."""
    
    if not MODELS_PATH.exists():
       raise ValueError("Model's database does not exists. please run 'nobrainer-zoo init'.")
       
    _ = get_model_db(MODELS_PATH)

    
        
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
    "--options",
    type=str,
    cls=OptionEatAll,
    help="Model-specific options",
    **_option_kwds,
)
def register(
    moving,
    fixed,
    moved,
    model,
    model_type,
    container_type,
    options,
    **kwrg
):
    """
    registers the MOVING image to the FIXED image.

    output saves in the path defined by MOVED argumet.

    """
    
    org, model_nm, ver = model.split("/")
    parent_dir = Path(__file__).resolve().parent
    
    # check model type
    _check_model_type(model, model_type)
    
    if model_type:
        model_dir = parent_dir / model / model_type
    else:
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
        
        nb_image = IMAGES_PATH / "nobrainer-zoo_nobrainer.sif"
        if not nb_image.exists():
            raise Exception("nobrainer-zoo container image not found! ",
                            "please run 'nobrainer-zoo init'.")
               
        cmd0 = ["singularity", "run", nb_image, "python3", 
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
    
    # download the model repository if needed
    if spec["repo"]["repo_download"]:
        repo_info = spec.get("repo")
        # UCL organization has separate repositories for different models
        if org == "UCL":
            org = org + "/" + model_nm
        repo_dest = CACHE_PATH / org / "org_repo"
        get_repo(repo_info["repo_url"], repo_dest, repo_info["commitish"])
              
    # check the input variables
    moving_path = _check_input(_name(moving=moving), moving, spec)
    fixed_path = _check_input(_name(fixed=fixed), fixed,spec)
    out_path = Path(moved).resolve().parent
    bind_paths = moving_path + fixed_path + [str(out_path)]
    
    # reading spec file in order to create options for model command
    options_spec = spec.get("options", {})
    
    # create model_path
    # it is used by some organizations like neuronet and voxelmorph 
    model_path = get_model_path(model, model_type=model_type)
    
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
            raise Exception(f"container name for {container_type} is not "
                            f"provided in the specification, "
                            f"try using container_type=docker")        
    elif container_type == "docker":
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

def _check_model_type(model_name, model_type=None):
    
    models = get_model_db(MODELS_PATH, print_models=False)     
    org,mdl,ver = model_name.split("/")

    models_w_types = [m.split("/")[1] for m,v in models.items() if isinstance(v,dict)]
    
    # check if model_types is given and correct
    if mdl in models_w_types and model_type not in models[model_name].keys():
        raise ValueError("Model type should be one of {} but it is {}".format(
          list(models[model_name].keys()), model_type))
    elif mdl not in models_w_types and model_type != None:
        raise ValueError(f"{model_name} does not have model type")
        
def _check_input(infile_name,infile, spec):
    """Checks the infile path and returns the binding path for the container"""
    # TODO: check if the infile is a dir
    if isinstance(infile, tuple):
        n_infile = len(infile)
    else:
        n_infile = 1
        infile = (infile,)
    n_inputs = spec["data_spec"][f"{infile_name}"]["n_files"]
    if n_inputs != "any" and n_infile != n_inputs:
        raise ValueError(f"This model needs {n_inputs} input files but {n_infile} files are given.")
   
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
        raise Exception(f"container_type should be docker or singularity, "
                        f"but {container_type} is provided.")
        
# for debugging purposes    
if __name__ == "__main__":
    cli()
