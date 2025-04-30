from collections.abc import MutableMapping
import h5py

def print_attrs(name, obj):
    print(f"Object: {name}")
    for key, value in obj.attrs.items():
        print(f"    Attribute - {key}: {value}")


def print_dict(dict, name=None):
    if name:
        print(f"In {name}: ")
    else:
        print(f"In Dictionary: ")
    for key, value in dict.items():
        print(f"    {key}: {value}")


def is_in_args(args, name, default):
    """Checks if the parammeter is specified in the args Namespace
    If not, attributes him the default value
    """
    if name in args:
        para = getattr(args, name)
    else:
        para = default
    return para


def convert_flatten(d, parent_key="", sep="_"):
    """
    Flattens a nested dict.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(convert_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def assert_identity(i1, i2):
    """assert_identity
    asserts that all indices are in the same sequence in the res list.
    """
    assert list(i1) == list(
        i2
    ), "the sequence of images are different between several models"
    return i2

def get_features(path):
    with h5py.File(path, "r") as f:
        attrs = dict(f["features"].attrs)
        feats = f["features"][:]
    return attrs, feats