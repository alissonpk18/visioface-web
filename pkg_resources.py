import os
import importlib.util

def resource_filename(package_name, resource_name):
    # Hack to replace pkg_resources for face_recognition_models
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Cannot find package {package_name}")
    pkg_dir = os.path.dirname(spec.origin)
    return os.path.join(pkg_dir, resource_name)
