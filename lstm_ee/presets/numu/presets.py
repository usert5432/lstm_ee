"""Presets for the NuE energy estimator training/eval"""

from lstm_ee.presets.numu.specs import get_numu_eval_presets

preset_numu_base = {
    'var_target_total'   : 'trueE',
    'var_target_primary' : 'trueLepE',
}

preset_numu_v2 = {
    **preset_numu_base,
    'vars_input_slice': [
        "calE",
        "remPngCalE",
    ],
    'vars_input_png2d' : None,
    'vars_input_png3d' : [
        "png.dir.x",
        "png.dir.y",
        "png.dir.z",
        "png.cvnpart.muonid",
        "png.cvnpart.electronid",
        "png.cvnpart.pionid",
        "png.cvnpart.protonid",
        "png.cvnpart.neutronid",
        "png.cvnpart.pizeroid",
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
        "png.bpf[2].pid",
        "png.bpf[2].energy",
        "png.bpf[2].overlapE",
        "png.bpf[2].momentum.x",
        "png.bpf[2].momentum.y",
        "png.bpf[2].momentum.z",
        "png.len",
        "png.weightedCalE",
        "png.calE",
    ],
}

preset_numu_v3 = {
    **preset_numu_base,
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

preset_numu_slice_linear = {
    **preset_numu_base,
    'vars_input_slice': [
        "trkLen",
        "hadCalE",
        "hadTrkE",
    ],
    'vars_input_png2d' : None,
    'vars_input_png3d' : None,
}

def add_train_numu_presets(presets):
    """Add numu training presets to the `presets` dict"""
    presets['numu_v2'] = preset_numu_v2
    presets['numu_v3'] = preset_numu_v3
    presets['numu_slice-simple'] = preset_numu_slice_linear

def add_eval_numu_presets(presets):
    """Add numu eval presets to the `presets` dict"""
    presets['numu_5GeV'] = get_numu_eval_presets((0, 5))
    presets['numu_7GeV'] = get_numu_eval_presets((0, 7))

