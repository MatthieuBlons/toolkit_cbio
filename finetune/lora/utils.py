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
