"""
Collection of tools for handling loading and saving arrays in the
"arrays" sibling folder.
"""
import numpy as np
import os
import hashlib

fileDir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(fileDir, '../arrays/')


def check(name):
    """
    Check if an array already exists
    """
    try:
        f = open(path+name+'.npy', 'rb')
    except FileNotFoundError:
        return False
    return True


def load(name, size):
    """
    Load content of a saved array
    """
    with open(path+name+'.npy', 'rb') as f:
        content = []
        for _ in range(size):
            content.append(np.load(f, allow_pickle=True))
    return content


def save(name, content):
    """
    Save an array
    """
    with open(path+name+'.npy', 'wb') as f:
        for array in content:
            np.save(f, array)
    return


def hash(var):
    """
    Returns a unique hash of any variable that can be cast to a string.
    Use for unique naming.
    """
    return hashlib.md5(str(var).encode('utf-8')).hexdigest()


def makehash(var):
    """
    Returns a unique hash of any variable that can be cast to a string.
    Use for unique naming.

    Edit: changed function name, since 'hash' is in the standard library
    """
    return hashlib.md5(str(var).encode('utf-8')).hexdigest()


def make_dict_name(var, max_list_len=10):
    string = ''
    for value in list(var.values()):
        if type(value) == dict:
            string += makehash(value) + '_'
        elif type(value) == np.ndarray or type(value) == list or type(value) == tuple:
            if len(value) >= max_list_len:
                string += makehash(value) + '_'
            else:
                for el in value:
                    string += str(el) + '_'
        else:
            string += str(value) + '_'
    return string


def makename_old(var):
    string = ''
    for value in list(var.values()):
        if type(value) == dict:
            string += makehash(value) + '_'
        elif type(value) == np.ndarray or type(value) == list:
            if len(value) >= 10:
                string += makehash(value) + '_'
            else:
                for el in value:
                    string += str(el) + '_'
        else:
            string += str(value) + '_'
    return string


def makename(var, max_list_len=10):
    """
    Returns a whitespace and special character free name containing all
    variables readably.
    """
    string = '_'
    if type(var) == dict:
        string += make_dict_name(var, max_list_len)
    else:
        for val in var:
            if type(val) == dict:
                string += make_dict_name(val, max_list_len) + '_'
            elif type(val) == np.ndarray or type(val) == list or type(val) == tuple:
                if len(val) >= max_list_len:
                    string += makehash(val) + '_'
                else:
                    for el in val:
                        string += str(el) + '_'
            else:
                string += str(val) + '_'
    return string


def autoscale_y(ax, margin=0.1):
    """Rescale y-axis to match the data range within current x-limits."""
    xlim = ax.get_xlim()
    
    if isinstance(margin, (int, float)):
        margin_lower, margin_upper = margin, margin
    else:
        margin_lower, margin_upper = margin
    
    y_min, y_max = float('inf'), float('-inf')
    
    for line in ax.get_lines():
        x_data, y_data = line.get_data()
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        
        # Skip vertical lines (axvline): all x-values are identical
        if len(x_data) > 0 and np.allclose(x_data, x_data[0]):
            continue
        
        mask = (x_data >= xlim[0]) & (x_data <= xlim[1])
        
        if mask.any():
            y_visible = y_data[mask]
            y_min = min(y_min, np.nanmin(y_visible))
            y_max = max(y_max, np.nanmax(y_visible))
    
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_range = y_max - y_min
        ax.set_ylim(bottom=y_min - margin_lower * y_range, 
                    top=y_max + margin_upper * y_range)
        ax.autoscale(enable=False, axis='y')


if __name__ == '__main__':
    name = 'test'
    print(check(name))
    testarray1 = np.zeros(3)
    testarray2 = np.ones(4)
    save(name, [testarray1, testarray2])
    content1, content2 = load(name, 2)
    print(content1)
    print(content2)

    a = {'a': 5., 'b': 123, 'c': {'d': 12, 'e': 13}, 'd': np.linspace(1, 2, 9), 'e':[1, 2]}
    print(makename(a))
