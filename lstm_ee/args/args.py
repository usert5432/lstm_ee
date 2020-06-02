"""
Definition of the `Args` object that holds runtime training configuration.
"""

import json
import os
import copy

from lstm_ee.consts import ROOT_DATADIR, ROOT_OUTDIR, DEF_SEED

from .config import Config
from .funcs import calc_savedir, modify_vars, update_kwargs

class Args:
    """Runtime training configuration.

    `Args` contains an instance of `Config` that defines the training, plus a
    number of options that are not necessary for training reproduction.

    Parameters
    ----------
    save_best : bool, optional
        Flag that controls whether save best model according to the validation
        loss, or simply the last model. If True then save the best model,
        otherwise will save the last model. Default: True.
    outdir : str
        Parent directory under `root_outdir` where model directory will be
        created.
    vars_mod_png2d : list of str or None, optional
        A list of strings that define how 2d prong inputs from `Config` should
        be modified. E.g. "+var_name" will add "var_name" to the 2d prong input
        list and  "-var_name" will remove "var_name" from the 2d prong input
        list. If None then `config` 2d prong inputs will not be modified.
        Default: None.
    vars_mod_png3d : list of str or None, optional
        A list of strings that define how 3d prong inputs from `Config` should
        be modified. C.f. `vars_mod_png2d` parameter.
    vars_mod_slice : list of str or None, optional
        A list of strings that define how slice inputs from `Config` should be
        modified. C.f. `vars_mod_png2d` parameter.
    cache : bool, optional
        If True data batches will be cached in RAM. Default: False.
        If `cache` is False and `concurrency` is not None, then the internal
        `keras` concurrent data generation will be used.
        Otherwise, data cache will be filled in parallel and keras will be
        run without concurrent data generation.
    disk_cache : bool, optional
        If True data batches will be cached in on a disk. Default: False.
        Caches are stored under "`root_outdir`/.cache" and should be
        cleaned manually.
    concurrency : { 'process', 'thread', None}, optional
        Type of the parallel data batch generation to use.
        If `concurrency` is "process" then will spawn several parallel
        processes for the data batch generation (may eat all your RAM).
        If "thread" then will spawn several parallel threads, mostly
        ineffective due to GIL.
        The number of parallel threads or processes is controlled by the
        `workers` parameter.
        If None then will not use parallelized data batch generation.
        Default: None.
    workers : int or None, optional
        Number of parallel workers to spawn for the purpose of data batch
        generation. If None then no parallelization will be used.
    **kwargs : dict
        Parameters to be passed to the `Config` constructor.
    extra_kwargs : dict or None, optional
        Specifies extra arguments that will be used to modify `kwargs` above.
        This `extra_kwargs` will be saved in a separate file and will be
        used to determine model `savedir`.

    Attributes
    ----------
    config : Config
        Training configuration.
    savedir : str
        Directory under `root_outdir` where trained model and its config
        will be saved.  It is calculated based on the `outdir` and
        `extra_kwargs` parameters following the pattern:
        `savedir` = `outdir`/model_`extra_kwargs`_hash(hash of `config`).
    root_data : str
        Parent directory where all data is saved.
        Unless set explicitly it is equal to "${LSMT_EE_DATADIR}".
    root_outdir : str
        Parent directory where all trained models are saved.
        Unless set explicitly it is equal to "${LSMT_EE_OUTDIR}".

    Notes
    -----
    The `extra_kwargs` parameter is useful for example when you would like to
    train multiple networks with the same configuration but say different
    `batch_size`. In such case you would pass the basic configuration through
    the **kwargs, but the `batch_size` parameter through the `extra_kwargs` = {
    'batch_size' : N }. After that networks with different `batch_size` will be
    saved in different directories, human friendly named according to the
    `batch_size`.  C.f. `savedir` attribute

    Keep in mind that `Args` class has a bit weird constructor. Maybe I will
    simplify it in the future. But for now, note that due to the way `Args` is
    constructed you can also pass a number of arguments defined in the
    `Attributes` section using the `kwargs` dict parameter. Do *NOT* do that
    unless you know what you are doing.
    """

    # pylint: disable=access-member-before-definition
    __slots__ = (
        'config',
        'savedir',
        'save_best',
        'outdir',

        'vars_mod_png2d',
        'vars_mod_png3d',
        'vars_mod_slice',

        'root_datadir',
        'root_outdir',

        'cache',
        'disk_cache',
        'concurrency',
        'workers',

        'extra_kwargs',
    )

    def __init__(
        self, loaded = False, extra_kwargs = None, **kwargs
    ):
        for k in self.__slots__:
            setattr(self, k, None)

        self.extra_kwargs = extra_kwargs

        kwargs = copy.deepcopy(kwargs)
        update_kwargs(kwargs, extra_kwargs)

        self.config = Config(**kwargs)

        for k,v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v)
            else:
                if not k in self.config.__slots__:
                    raise ValueError(
                        "Unknown Parameter '%s = %s'" % (k, v)
                    )

        self._init_default_values()

        if not loaded:
            self._modify_variables()
            self._init_savedir()

    @staticmethod
    def load(savedir):
        """Load `Args` from the directory `savedir`"""
        # pylint: disable=attribute-defined-outside-init

        config = Config.load(savedir)
        result = Args(loaded = True)

        result.config  = config
        result.savedir = savedir

        result.outdir = os.path.normpath(
            os.path.join(result.savedir, os.path.pardir)
        )

        try:
            with open("%s/extra.json" % (savedir), 'rt') as f:
                result.extra_kwargs = json.load(f)
        except IOError:
            pass

        return result

    def _init_default_values(self):
        """Initialize unspecified `Args` attributes"""
        if self.root_datadir is None:
            self.root_datadir = ROOT_DATADIR

        if self.root_outdir is None:
            self.root_outdir = ROOT_OUTDIR

        if self.config.seed is None:
            self.config.seed = DEF_SEED

        if self.save_best is None:
            self.save_best = True

    def _modify_variables(self):
        """Modify input variables.

        This function modifies slice, 2d and 3d prong input variables according
        to the rules defined by the `vars_mod_slice`, `vars_mod_png2d`,
        `vars_mod_png3d` parameters.
        C.f. `Args` constructor for their description.
        """

        self.config.vars_input_slice = modify_vars(
            self.vars_input_slice, self.vars_mod_slice
        )

        self.config.vars_input_png2d = modify_vars(
            self.vars_input_png2d, self.vars_mod_png2d
        )

        self.config.vars_input_png3d = modify_vars(
            self.vars_input_png3d, self.vars_mod_png3d
        )

    def _init_savedir(self):
        """Create training directory and save `Args` there."""
        self.savedir = calc_savedir(
            os.path.join(self.root_outdir, self.outdir),
            'model', self.config, self.extra_kwargs
        )

        self.config.save(self.savedir)

        with open("%s/extra.json" % (self.savedir), 'wt') as f:
            json.dump(self.extra_kwargs, f, sort_keys = True, indent = 4)

    def __getattr__(self, name):
        """Get attribute `name` from `Args.config`.

        This function is invoked when one has called `Args.name`, but the
        `Args` itself does not have `name` attribute. In such case it will
        return `Args.config.name`.
        """
        return getattr(self.config, name)

    def __getitem__(self, name):
        """Get `Args` or `Args.config` attribute specified by address `name`.

        This function tries to return an attribute of either `Args` or
        `Args.config` that is encoded in address `name`.

        Parameters
        ----------
        name : str of list of str
            If `name` is str, then first it is converted to a list of str by
            splitting it using ':' as delimiter. Say the resulting list is
            [ "attr", "addr1", "addr2" ]. Then this function will return
            value of the expression: getattr(`self`, "attr")["addr1"]["addr2"].

            If `name` is list of str, then it will return a list of
            [ self[x] for x in `name` ]

        Returns
        -------
        str or list of str
            Value(s) specified by `name`.
        """

        # NOTE: Address is needed to access keys of the nested dictionaries.
        #       e.g. say config.opt_kwargs = { 'lr' : 0.001 }
        #       To access learning rate 'lr' directly one has to call:
        #       args['opt_kwargs:lr']

        if isinstance(name, list):
            return [self[n] for n in name]

        address = name.split(':')

        result = getattr(self, address[0])

        for addr_part in address[1:]:
            result = result[addr_part]

        return result

