"""RSL-RL API adapters shared by training and playback."""

from importlib.metadata import version
from inspect import signature

from packaging.version import Version


def prepare_agent_cfg(cfg):
    """Migrate the Isaac Lab runner config when the installed release requires it."""
    installed_version = version("rsl-rl-lib")
    if Version(installed_version) < Version("3.0.1"):
        raise RuntimeError(f"RSL-RL >= 3.0.1 is required; found {installed_version}.")
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    except ImportError:
        return cfg
    return handle_deprecated_rsl_rl_cfg(cfg, installed_version)


def load_weights(runner, checkpoint):
    """Load a native checkpoint without restoring training state for playback."""
    if "load_cfg" in signature(runner.load).parameters:
        runner.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
    else:
        runner.load(checkpoint, load_optimizer=False)
