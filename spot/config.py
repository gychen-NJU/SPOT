# -*- coding: utf-8 -*-
"""
spot.config — configuration dictionary utilities
==================================================

The package is configured through a single Python dict.  The defaults
live in :mod:`spot.default`, and a user config only needs to specify
the keys that differ; nested dicts are merged recursively by
:func:`merge_config` (scalars/sequences overwrite, dicts recurse).

The merged configuration can be frozen into an immutable nested dict
(:func:`freeze_config`) so that downstream code can rely on the exact
set of keys.
"""

from copy import deepcopy

__all__ = ["merge_config", "update_config", "freeze_config", "ConfigError"]


class ConfigError(ValueError):
    """Raised when a user configuration is not a dict or contains a bad key."""


def merge_config(user, base=None):
    """
    Recursively merge ``user`` over ``base`` (defaults).

    Parameters
    ----------
    user : dict
        User-provided configuration.  May be partial.
    base : dict or None
        Default configuration.  If None, :data:`spot.default.DEFAULT_CONFIG`
        is used.

    Returns
    -------
    dict
        A *new* deep-copied dict: ``base`` keys not present in ``user``
        keep their default value, scalars/sequences are overwritten,
        nested dicts are merged recursively.  Neither input is modified.
    """
    if base is None:
        from .default import DEFAULT_CONFIG
        base = DEFAULT_CONFIG
    if not isinstance(user, dict):
        raise ConfigError(
            f"configuration must be a dict, got {type(user).__name__}")
    merged = deepcopy(base)
    for key, value in user.items():
        if key not in merged:
            raise ConfigError(
                f"unknown configuration key {key!r}; "
                f"known top-level keys: {sorted(merged)}")
        if isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = merge_config(value, merged[key])
        else:
            merged[key] = deepcopy(value)
    return merged


def update_config(user, base=None):
    """
    Alias of :func:`merge_config` (mutating-style name kept for
    convenience; the returned dict is still a fresh object).
    """
    return merge_config(user, base)


def freeze_config(config):
    """
    Return an immutable (mappingproxy) view of a merged configuration.

    Nested dicts are converted to :class:`types.MappingProxyType`, so
    accidental mutation raises ``TypeError``.  Used internally by the
    Synthesis/Inversion classes to guarantee config stability.
    """
    from types import MappingProxyType

    if isinstance(config, dict):
        return MappingProxyType({k: freeze_config(v) for k, v in config.items()})
    return config
