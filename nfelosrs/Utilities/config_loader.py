import json
import pathlib

## package path ##
package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()

def get_package_dir():
    ## returns the package root directory ##
    return pathlib.Path(__file__).parent.parent.parent.resolve()

def load_config(file, config_route):
    ## load config from path ##
    config = None
    with open('{0}/{1}'.format(package_dir,file)) as fp:
        config = json.load(fp)
    ## loop path to get to proper level ##
    for route in config_route:
        config = config[route]
    ## return ##
    return config
