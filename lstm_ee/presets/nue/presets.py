"""Presets for the NuE energy estimator training/eval"""

from lstm_ee.presets.nue.specs import (
    plot_specs_nue_rel_vs_true,
    plot_specs_nue_hist,
    plot_specs_nue_binstats_abs,
    plot_specs_nue_binstats_rel,
    plot_specs_nue_fom,
    base_map,
    name_map,
    units_map,
)


preset_nue_base = {
    'var_target_total'   : [ 'trueE'    ],
    'var_target_primary' : [ 'trueLepE' ],
}

preset_nue_v3 = {
    **preset_nue_base,
    'vars_input_slice': [
        "calE",
        "remPngCalE",
        "nHit",
        "orphCalE",
        "coarseTiming",
        "lowGain",
    ],
    'vars_input_png2d' : [
        "png2d.dir.x",
        "png2d.dir.y",
        "png2d.dir.z",
        "png2d.len",
        "png2d.weightedCalE",
        "png2d.calE",
        "png2d.nhit",
        "png2d.nhitx",
        "png2d.nhity",
        "png2d.nplane",
        "png2d.start.x",
        "png2d.start.y",
        "png2d.start.z",
    ],
    'vars_input_png3d' : [
        "png.dir.x",
        "png.dir.y",
        "png.dir.z",
        "png.start.x",
        "png.start.y",
        "png.start.z",
        "png.cvnpart.muonid",
        "png.cvnpart.electronid",
        "png.cvnpart.pionid",
        "png.cvnpart.protonid",
        "png.cvnpart.photonid",
        "png.bpf[0].energy",
        "png.bpf[0].overlapE",
        "png.bpf[0].momentum.x",
        "png.bpf[0].momentum.y",
        "png.bpf[0].momentum.z",
        "png.bpf[1].energy",
        "png.bpf[1].overlapE",
        "png.bpf[1].momentum.x",
        "png.bpf[1].momentum.y",
        "png.bpf[1].momentum.z",
        "png.bpf[2].energy",
        "png.bpf[2].overlapE",
        "png.bpf[2].momentum.x",
        "png.bpf[2].momentum.y",
        "png.bpf[2].momentum.z",
        "png.len",
        "png.nhit",
        "png.nhitx",
        "png.nhity",
        "png.nplane",
        "png.weightedCalE",
        "png.calE",
    ],
}

def add_train_nue_presets(presets):
    """Add nue training presets to the `presets` dict"""
    presets['nue_v3'] = preset_nue_v3

def add_eval_nue_presets(presets):
    """Add nue eval presets to the `presets` dict"""
    presets['nue'] = {
        'rel_vs_true' : plot_specs_nue_rel_vs_true,
        'hist'        : plot_specs_nue_hist,
        'binstats_abs': plot_specs_nue_binstats_abs,
        'binstats_rel': plot_specs_nue_binstats_rel,
        'fom'         : plot_specs_nue_fom,
        'base_map'    : base_map,
        'name_map'    : name_map,
        'units_map'   : units_map,
    }

