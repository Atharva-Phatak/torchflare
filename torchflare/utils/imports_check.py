from importlib.util import find_spec


def module_available(module_path: str) -> bool:
    """Function to check whether the package is available or not.

    Args:
        module_path : The name of the module.
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False
