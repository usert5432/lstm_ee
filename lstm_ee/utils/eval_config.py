"""
Definition of `EvalConfig` that holds parameters of evaluation.
"""

import json

def build_eval_subdir(list_of_label_value):
    """Create evaluation subdir name from a list of evaluation parameters"""
    tokens = []

    for label,value in list_of_label_value:
        if value is None:
            continue

        tokens.append('%s(%s)' % (label, value))

    result = "_".join(tokens)

    if len(result) == 0:
        result = 'eval'

    result = result.replace('/', '::')

    return result

def modify_args_value(args, attr, eval_value, dtype = None):
    """Set `args` attribute `attr` to `eval_value`.

    This function modifies `args` attribute `attr`. It sets it to `eval_value`
    paying attention to the special values of `eval_value`:
      - if `eval_value` is None, then the `attr` value won't be modified.
      - if `eval_value` == 'none', then the `attr` value will be set to None.
      - otherwise, `attr` = `eval_value`
    """
    if eval_value is None:
        return

    if eval_value.lower() == 'none':
        setattr(args, attr, None)
    else:
        if dtype:
            setattr(args, attr, dtype(eval_value))
        else:
            setattr(args, attr, eval_value)

def modify_args_value_from_conf(args, attr, eval_value):
    """Load `args` attribute `attr` from a json file `eval_value`.

    This function modifies `args` attribute `attr`. It loads its value from
    a json file `eval_value` paying attention to the special values of
    `eval_value`:
      - if `eval_value` is None, then the `attr` value won't be modified.
      - if `eval_value` == 'none', then the `attr` value will be set to None.
      - otherwise, `attr` = json.load(`eval_value`)
    """
    if eval_value is None:
        return

    if eval_value.lower() == 'none':
        setattr(args, attr, None)
    else:
        with open(eval_value, 'rt') as f:
            setattr(args, attr, json.load(f))

class EvalConfig:
    """Configuration of an `lstm_ee` network evaluation.

    Parameters of the `EvalConfig` will be used to modify the corresponding
    parameters of the `Args` of the trained network. The rules of such
    modification are as follows:
        - If (parameter == "same") or (parameter is None) then the value of
          `Args` will not be modified.
        - If parameter == "none" then the value of `Args` will be set to None.
        - otherwise `Args` parameter will be either directly set to parameter
          from `EvalConfig` or will be loaded from a file specified by a
          parameter of `EvalConfig`.

    Parameters
    ----------
    data : str or None,
        Name of the evaluation dataset.
    noise : str or None,
        JSON file name with the noise config that will be used during
        evaluation.  C.f. `Config.noise` for the configuration spec.
    preset : str or None,
        Name of the evaluation preset. C.f. `PRESETS_EVAL`.
    prong_sorter : str or None,
        JSON file name with the prong sorting config that will be used during
        evaluation.  C.f. `Config.prong_sorter` for the configuration spec.
    test_size : int or float or None
        Size of the dataset that will be used for evaluation.
        C.f. `Config.test_size`.
    weights : str or None
        Weights specification that will be used during the evaluation.
        C.f. `Config.weights`.
    """

    @staticmethod
    def _recognize_same(value):
        if isinstance(value, str) and (value == 'same'):
            return None

        return value

    @staticmethod
    def from_cmdargs(cmdargs):
        """Construct `EvalConfig` from parameters from `argparse.Namespace`"""
        return EvalConfig(
            cmdargs.data,
            cmdargs.noise,
            cmdargs.preset,
            cmdargs.prong_sorter,
            cmdargs.test_size,
            cmdargs.weights,
        )

    def __init__(self, data, noise, preset, prong_sorter, test_size, weights):
        self.data         = EvalConfig._recognize_same(data)
        self.noise        = EvalConfig._recognize_same(noise)
        self.preset       = EvalConfig._recognize_same(preset)
        self.prong_sorter = EvalConfig._recognize_same(prong_sorter)
        self.test_size    = EvalConfig._recognize_same(test_size)
        self.weights      = EvalConfig._recognize_same(weights)

    def get_eval_subdir(self):
        """Create eval subdir that is unique for this evaluation config."""
        return build_eval_subdir([
            ('data',    self.data),
            ('noise',   self.noise),
            ('preset',  self.preset),
            ('psort',   self.prong_sorter),
            ('tsize',   self.test_size),
            ('weights', self.weights),
        ])

    def modify_eval_args(self, args):
        """Modify parameters of `args` using values from `self`"""
        modify_args_value(args.config, 'dataset',   self.data)
        modify_args_value(args.config, 'test_size', self.test_size, float)
        modify_args_value(args.config, 'weights',   self.weights)

        modify_args_value_from_conf(args.config, 'noise', self.noise)
        modify_args_value_from_conf(
            args.config, 'prong_sorters', self.prong_sorter
        )

