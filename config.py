# config.py
import argparse
import logging
import pickle
from pathlib import Path
import yaml

# Constants
C = 32
M = 3


# We only specify the yaml file from argparse and handle rest
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--config_file", default="configs/defaults.yaml", help="configuration file path")
parser.add_argument("-l", "--log_file", default="log", help="log file path")
parser.add_argument("-v", "--verbose", action='store_true', help="set logging level to DEBUG")
ARGS, etc = parser.parse_known_args()

# Set up logging
if ARGS.verbose: logging.basicConfig(filename=ARGS.log_file, level=logging.DEBUG)
else: logging.basicConfig(filename=ARGS.log_file, level=logging.INFO)
    
# Stop filelock from clogging up the log
logging.getLogger("filelock").setLevel(logging.ERROR)

# Load the yaml file 
with open(ARGS.config_file, 'r') as f:
    config = yaml.safe_load(f)
    print(f"Loaded configuration file {ARGS.config_file}")

# override config with any remaining valid arguments from parser 
# (useful for managing wandb sweeps)

def recurse_keys(d, key_list):
    # return innermost dict of nested dict d
    if len(key_list) == 2: return d[key_list[0]]
    else: return recurse_keys(d[key_list[0]], key_list[1:])

for arg in etc:
    if arg.startswith(("--")):
        # strip --
        # first two elements must define file for parameter injection
        a = arg[2:].split('.')
        a = [a[0] + '.' + a[1]] + a[2:]
        
        # cast value type as appropriate
        a[-1], val = a[-1].split('=')
        val = type(recurse_keys(config, a)[a[-1]])(val)
        
        # update innermost key with passed parameter
        logging.debug(f"Overriding arg {arg[2:]} from yaml with passed cmd value")
        recurse_keys(config, a).update({a[-1]: val})
        

logging.debug(f"Config parameterization: {config}")

def extyaml(func):
    """Wraps keyword arguments from configuration."""
    def wrapper(*args, **kwargs):
        """Injects configuration keywords."""
        # We get the file name in which the function is defined, ex: train.py
        fname = Path(func.__globals__['__file__']).name
        # Then we extract arguments corresponding to the function name
        # ex: train.py -> load_data
        conf = config[fname][func.__name__]
        # And update the keyword arguments with any specified arguments
        # if it isn't specified then the default values still hold
        conf.update(kwargs)
        return func(*args, **conf)
    return wrapper

def save_checkpoint(fn, model, trace):
    with open(f'{fn}.pickle', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace, 'config': config}, buff)
        logging.debug(f"Model checkpoint pickled to: {fn}")
        
def load_checkpoint(fn):
    with open(fn, 'rb') as buff:
        data = pickle.load(buff)
        logging.debug(f"Model checkpoint loaded from: {fn}")
    return data