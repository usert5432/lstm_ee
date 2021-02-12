"""Presets for the NuE energy estimator training/eval"""

from lstm_ee.presets.numu.specs import (
    get_numu_eval_presets, LABEL_PRIMARY, LABEL_TOTAL
)

BASE_MAP_DUNE = {
    LABEL_PRIMARY : "numue.lepE",
    LABEL_TOTAL   : "numue.nuE",
}

preset_dune_numu_base = {
    'var_target_total'   : 'mc.nuE',
    'var_target_primary' : 'mc.lepE',
}

preset_dune_numu_v1 = {
    **preset_dune_numu_base,
    'vars_input_slice': [
        "event.calE",
        "event.charge",
        "event.nHits",
    ],
    'vars_input_png3d' : [
        "particle.is_shower",
        "particle.length",
        "particle.start.x",
        "particle.start.y",
        "particle.start.z",
        "particle.dir.x",
        "particle.dir.y",
        "particle.dir.z",
        "particle.energy",
        "particle.calE",
        "particle.charge",
        "particle.nHit",
    ],
}

def get_dune_numu_eval_presets(energy_range):
    """Construct DUNE numu eval presets from NOvA preset"""
    result = get_numu_eval_presets(energy_range)
    result['base_map'] = BASE_MAP_DUNE

    return result

def add_train_dune_numu_presets(presets):
    """Add DUNE numu training presets to the `presets` dict"""
    presets['dune_numu_v1'] = preset_dune_numu_v1

def add_eval_dune_numu_presets(presets):
    """Add DUNE numu eval presets to the `presets` dict"""
    presets['dune_numu_5GeV'] = get_dune_numu_eval_presets((0, 5))

