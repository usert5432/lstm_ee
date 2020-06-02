"""
A collection of helper functions for `Args` construction.
"""

import hashlib
import json
import os
import re

def modify_vars(vars_orig, var_modifiers):
    """Return a copy of `vars_orig` modified according to `var_modifiers`.

    Parameters
    ----------
    vars_orig : list of str
        A list of variable names.
    var_modifiers : list of str or None, optional
        A list of variable modifiers. Each variable modifier is a string
        either "+var_name" or "-var_name".
        If `var_modifiers` has "+var_name", then "var_name" will be added
        to the result.
        If `var_modifiers` has "-var_name", then "var_name" will be removed
        from  the result.
        If `var_modifiers` is None, then a simple copy of `vars_orig` will be
        returned.
        Default: None.

    Returns
    -------
    list of str
        A copy of `vars_orig` modified according to `var_modifiers`.
    """

    if var_modifiers is None:
        return vars_orig

    result = vars_orig[:]

    regexp = re.compile(r'([+-])(.*)')

    for vm in var_modifiers:

        res = regexp.match(vm)
        if not res:
            raise ValueError('Failed to parse column modifier %s' % vm)

        mod = res.group(1)
        val = res.group(2)

        if mod == '+':
            if val not in result:
                result.append(val)
        else:
            result.remove(val)

    return result

def update_kwargs(kwargs, extra_kwargs):
    """Modify kwargs dict inplace by items from extra_kwargs.

    This function overrides/updates items from `kwargs` by the items from
    `extra_kwargs`.

    Notes
    -----
    If `extra_kwargs` is None, then `kwargs` will not be modified.

    If `extra_kwargs` contains an item (`key`, `value`) and `kwargs` also
    contains an item (`key`, `orig_value`) and both `value` and `orig_value`
    are dict themselves, then `update_kwargs` will be called recursively
    on `orig_value`. That is `update_kwargs`('orig_value`, `value`) will be
    called.

    Otherwise, for all other (`key`, `value`) pairs in the `extra_kwargs` the
    following will be executed: `kwargs`[`key`] = `value`
    """

    if extra_kwargs is None:
        return

    for k,v in extra_kwargs.items():
        if (
                isinstance(v, dict)
            and k in kwargs
            and isinstance(kwargs[k], dict)
        ):
            update_kwargs(kwargs[k], v)
        else:
            kwargs[k] = v

def join_dicts(*dicts_list):
    """Return a dict obtained by joining dicts_list with `update_kwargs`."""

    base_dict = {}

    for d in dicts_list:
        update_kwargs(base_dict, d)

    return base_dict

def calc_savedir(parent, name, config, extra_kwargs):
    """Calculate save directory for a given training.

    This function calculates and creates(under `path`) a save directory for a
    training, specified by `config` and `extra_kwargs`.
    The save directory is calculated using the following schematic repr:
    return `name`_`extra_kwargs`_hash(hash of `config`).

    Parameters
    ----------
    parent : str
        Root directory under which save directory will be created.
    name : str
        Prefix of the last component of a path of the save directory.
    config : `Config`
        Training configuration.
    extra_kwargs : dict or None
        Extra training parameters used to override `config`.

    Returns
    -------
    str
        Path of the save directory relative to `parent`.

    Notes
    -----
    hash of `config` is defined as sha1 hash of the str representation of
    `config`.

    If expression `name`_`extra_kwargs`_hash(hash of `config`) results in a
    too long name for a filesystem, then it will be truncated to just
    `name`_hash(hash of `config`)
    """

    extra_kwargs    = extra_kwargs or {}
    relevant_extras = {
        k : v for k,v in extra_kwargs.items() if k in config.__slots__
    }

    if relevant_extras:
        basename = json.dumps(
            relevant_extras, sort_keys = True, separators = (',', ':')
        )
    else:
        basename = ""

    basename = basename.replace('"', '').replace('/', ':')
    basename = basename.replace('{', '(').replace('}', ')')

    digest = bytes(str(config), 'utf-8')
    digest = hashlib.sha1(digest).hexdigest()

    basename = '%s%s_hash(%s)' % (name, basename, digest)

    if len(basename) >= 255:
        basename = '%s_hash(%s)' % (name, digest)

    savedir = os.path.join(parent, basename)
    os.makedirs(savedir, exist_ok = True)

    return savedir

