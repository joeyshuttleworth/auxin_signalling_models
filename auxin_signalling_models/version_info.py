from importlib import metadata

from ._version import __commit_id__, __version__, __version_tuple__


def get_version_info():

    if __name__ == "__main__":
        raise RuntimeError("auxin_signaling_models.version_info ran as script")

    else:
        _module_names = __name__.split(".")
        module_name = _module_names[0]

    info_dict = metadata.metadata(module_name)
    info_dict["Version_tuple"] = __version_tuple__
    info_dict["Commit_id"] = __commit_id__
    info_dict["Version"] = __version__

    return info_dict
