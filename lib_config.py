import dataclasses
from typing import Any, Dict, List

from pathlib import Path
import numpy as np

try:
    import tomllib
except Exception:
    import tomli as tomllib

# ------------------------------ Constants ------------------------------
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent # scripting directory
else:
    BASE_DIR = Path.cwd()

CONFIG_PATH: Path = BASE_DIR / "_config.toml"
RUNS_DIR: Path = BASE_DIR / "..."

# ------------------------------ Configs ------------------------------

@dataclasses.dataclass
class IOConfig:
    filename_input: str
    filename_output: str
    filename_temp: str

@dataclasses.dataclass
class GaussianProcessConfig:
    kernel_type: str
    length_scale: float
    noise_std: float
    noise_var: float


#@dataclasses.dataclass
#class HFSSConfig:
#    n_simulation: int
#    n_repeats: int
#    n_init: int
#    n_params: int
#    lower_bounds: float
#    upper_bounds: float
#    param_names: str
#    param_units: float
#    filename_models: str

@dataclasses.dataclass
class HFSSConfig:
    n_simulation: int
    n_repeats: int
    n_init: int
    n_params: int
    lower_bounds: List[float]
    upper_bounds: List[float]
    param_names: List[str]
    param_units: List[str]
    filename_models: List[str]
    param_groups: Dict[str, Dict[str, Any]]
    group_order: List[str] | None = None

@dataclasses.dataclass
class SyntheticTestConfig:
    n_simulation: int
    n_repeats: int
    n_init: int
    n_params: int
    lower_bounds: float
    upper_bounds: float
    param_names: str


@dataclasses.dataclass
class Environment:
    dir_base: Path

@dataclasses.dataclass
class Depends:
    n_gp: Path

@dataclasses.dataclass
class AppConfig:
    io: IOConfig
    opt: GaussianProcessConfig
    hfss: HFSSConfig
    test: SyntheticTestConfig
    env: Environment

    @staticmethod
    def fromDict(config: dict) -> "AppConfig":
        io = config["io"]; opt=config["opt"]; hfss = config["hfss"]; test = config["test"]

        hfss = flatten_hfss_param_groups(hfss) # flatten param groups if they exist

        dir_base = BASE_DIR
        env = Environment(
            dir_base=dir_base
        )

        return AppConfig(
            io=IOConfig(**io),
            opt = GaussianProcessConfig(**opt),
            hfss = HFSSConfig(**hfss),
            test = SyntheticTestConfig(**test),
            env = env,
        )

# ------------------------------ App ------------------------------
def _loadConfig(path: Path = None) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)

def printConfig(cfg: AppConfig):
    for name, data_obj in cfg.__dict__.items():
        if not dataclasses.is_dataclass(data_obj):
            continue
        header = name.replace('_', ' ').title()
        print(f"\n[{header}]")
        for field in dataclasses.fields(data_obj):
            key = field.name
            value = getattr(data_obj, key)
            if isinstance(value, np.ndarray):
                print(f"  {key:<20}: ndarray(shape={value.shape}, dtype={value.dtype})")
            else:
                if isinstance(value, float):
                    print(f"  {key:<20}: {value:.6f}")
                else:
                    print(f"  {key:<20}: {value}")

def initParams(_config: dict, debug: bool = True, runs_dir: Path | str = RUNS_DIR) -> None:
    cfg = AppConfig.fromDict(_config)
    if debug:
        printConfig(cfg)
    return cfg

def flatten_hfss_param_groups(hfss: Dict[str, Any]) -> Dict[str, Any]:

    if "param_groups" not in hfss:
        return hfss

    pg = hfss["param_groups"]

    order: List[str] = hfss.get("group_order") or list(pg.keys())

    lower: List[float] = []
    upper: List[float] = []
    names: List[str] = []
    units: List[str] = []

    for gname in order:
        g = pg[gname]
        names.extend(list(g["param_names"]))
        units.extend(list(g["param_units"]))
        lower.extend(list(g["lower_bounds"]))
        upper.extend(list(g["upper_bounds"]))

    hfss["param_names"] = names
    hfss["param_units"] = units
    hfss["lower_bounds"] = lower
    hfss["upper_bounds"] = upper
    hfss["n_params"] = len(names)

    return hfss