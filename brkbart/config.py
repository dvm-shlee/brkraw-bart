import os
import shutil
from datetime import date
import configparser

from pkg_resources import find_distributions
from .utils import *

if os.name == 'nt':
    config_file = 'brkbart.ini'
else:
    config_file = '.brkbartrc'


#%% Set config
def create_config_file(cfg, path, search_path, search_depth):
    # Config for BART TOOLBOX
    if 'TOOLBOX_PATH' not in os.environ.keys():
        print('BART TOOLBOX not found...Searching...')
        toolbox_path = search_bart_installed_location(path=search_path, depth=search_depth)
    else:
        toolbox_path = os.environ['TOOLBOX_PATH']
        
    cfg['Default'] = dict(toolbox_path=toolbox_path)

    with open(path, 'w') as configfile:
        cfg.write(configfile)

def rescan_bart(search_path, search_depth):
    global cfg_path
    if os.path.exists(cfg_path):
        dirname, filename = os.path.split(cfg_path)
        shutil.copy(cfg_path, os.path.join(dirname, '{}_{}'.format(filename, date.today().strftime("%y%m%d"))))
        os.unlink(cfg_path)
    new_config = configparser.RawConfigParser()
    create_config_file(new_config, cfg_path, search_path, search_depth)


#%% Load config
cfg_path = os.path.join(os.path.expanduser("~"), config_file)
config = configparser.ConfigParser()

if not os.path.exists(cfg_path):
    try:
        create_config_file(config, cfg_path, search_path='/usr', search_depth=4)
        config.read(cfg_path)
    except:
        pass
else:
    config.read(cfg_path)

if __name__ == '__main__':
    pass