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
ARGS = parser.parse_args()

# Set up logging
if ARGS.verbose:
    logging.basicConfig(filename=ARGS.log_file, level=logging.DEBUG)
else:
    logging.basicConfig(filename=ARGS.log_file, level=logging.INFO)
    
# Stop filelock from clogging up the log
logging.getLogger("filelock").setLevel(logging.ERROR)

# Let's load the yaml file here
with open(ARGS.config_file, 'r') as f:
    config = yaml.safe_load(f)
        
print(f"Loaded configuration file {ARGS.config_file}")

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
        
def load_checkpoint(fn):
    with open(fn, 'rb') as buff:
        data = pickle.load(buff)
    return data