"""Basic definitions for the NuMu energy estimator training/eval"""

from lstm_ee.presets.specs import (
    get_plot_spec_rel_res_vs_true,
    get_plot_spec_hist,
    get_plot_spec_binstat_abs,
    get_plot_spec_binstat_rel,
    get_plot_spec_fom
)
from lstm_ee.consts import LABEL_TOTAL, LABEL_PRIMARY, LABEL_SECONDARY

name_map = {
    LABEL_PRIMARY   : 'Muon Energy',
    LABEL_SECONDARY : 'Hadronic Energy',
    LABEL_TOTAL     : 'Neutrino Energy',
}

units_map = {
    LABEL_PRIMARY   : 'GeV',
    LABEL_SECONDARY : 'GeV',
    LABEL_TOTAL     : 'GeV',
}

base_map = {
    LABEL_PRIMARY   : "numuRecoMuonE",
    LABEL_TOTAL     : "numuRecoE",
}

def get_numu_eval_presets(energy_range):
    """Get NuMu eval presets for a given range of TrueE"""
    plot_specs_numu_rel_vs_true = {
        LABEL_PRIMARY : get_plot_spec_rel_res_vs_true(
            "Muon", range_y = (-0.1, 0.1)
        ),
        LABEL_SECONDARY : get_plot_spec_rel_res_vs_true(
            "Hadronic", range_x = energy_range,
        ),
        LABEL_TOTAL : get_plot_spec_rel_res_vs_true(
            "Neutrino", range_y = (-0.25, 0.25)
        ),
    }

    plot_specs_numu_hist = {
        LABEL_PRIMARY   : get_plot_spec_hist("Muon",     range = energy_range),
        LABEL_SECONDARY : get_plot_spec_hist("Hadronic", range = energy_range),
        LABEL_TOTAL     : get_plot_spec_hist("Neutrino", range = energy_range),
    }

    plot_specs_numu_binstats_abs = {
        LABEL_PRIMARY : get_plot_spec_binstat_abs(
            "Muon", range = energy_range, bins = 50
        ),
        LABEL_SECONDARY : get_plot_spec_binstat_abs(
            "Hadronic", range = energy_range, bins = 50
        ),
        LABEL_TOTAL : get_plot_spec_binstat_abs(
            "Neutrino", range = energy_range, bins = 50
        )
    }

    plot_specs_numu_binstats_rel = {
        LABEL_PRIMARY : get_plot_spec_binstat_rel(
            "Muon", range = energy_range, bins = 50
        ),
        LABEL_SECONDARY : get_plot_spec_binstat_rel(
            "Hadronic", range = energy_range, bins = 50
        ),
        LABEL_TOTAL : get_plot_spec_binstat_rel(
            "Neutrino", range = energy_range, bins = 50
        )
    }

    plot_specs_numu_fom = {
        LABEL_PRIMARY   : get_plot_spec_fom("Muon"),
        LABEL_SECONDARY : get_plot_spec_fom("Hadronic"),
        LABEL_TOTAL     : get_plot_spec_fom("Neutrino"),
    }

    return {
        'rel_vs_true' : plot_specs_numu_rel_vs_true,
        'hist'        : plot_specs_numu_hist,
        'binstats_abs': plot_specs_numu_binstats_abs,
        'binstats_rel': plot_specs_numu_binstats_rel,
        'fom'         : plot_specs_numu_fom,
        'base_map'    : base_map,
        'name_map'    : name_map,
        'units_map'   : units_map,
    }

