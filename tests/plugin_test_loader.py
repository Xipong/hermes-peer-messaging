"""Load the checkout's package independent of its directory name."""
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_plugin():
    name = "peer-messaging"
    if name not in sys.modules:
        root = Path(__file__).resolve().parents[1]
        spec = spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
        package = module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)