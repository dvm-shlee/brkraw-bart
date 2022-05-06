import os
import re
import paralexe
import numpy as np


def get_path_depth(root, target):
    root_depth = len([p for p in root.split(os.sep) if len(p) > 0])
    target_depth = len([p for p in target.split(os.sep) if len(p) > 0])
    return target_depth - root_depth

def find_filename(filename, path, depth):
    cache = []
    for p, d, f in os.walk(path):
        if get_path_depth(path, p) > depth:
            del d[:]
        else:
            if filename in f:
                cache.append(p)
            else:
                del f[:]
    return cache

def search_bart_installed_location(path, depth):
    paths = find_filename('bart', path, depth)
    
    # filter if python module not exists
    paths = [p for p in paths if os.path.isfile(os.path.join(p, 'python', 'bart.py'))]
    
    if len(paths) == 0:
        return None
    elif len(paths) > 1:
        return get_newest_version(paths)
    else:
        return paths[0]

def check_bart_version(exe_path):
    ex = paralexe.Executor(f'{exe_path} version')
    ex.run()
    found = re.findall('\d+\.\d+\.\d+', ex.stdout.read().decode('utf-8').split('\n')[0])
    if found > 0:
        return found[0]
    else:
        raise Exception('Cannot find BART version')

def get_newest_version(paths):
    versions = np.array([check_bart_version(os.path.join(p, 'bart')).split('.') for p in paths]).astype(int).T
    for v_str in versions:
        if not all(v_str == v_str[0]):
            return paths[v_str.argmax()]