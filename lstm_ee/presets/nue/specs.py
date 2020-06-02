"""Basic definitions for the NuE energy estimator training/eval"""

from lstm_ee.presets.specs import (
    get_plot_spec_rel_res_vs_true,
    get_plot_spec_hist,
    get_plot_spec_binstat_abs,
    get_plot_spec_binstat_rel,
    get_plot_spec_fom
)
from lstm_ee.consts import LABEL_TOTAL, LABEL_PRIMARY, LABEL_SECONDARY

name_map = {
    LABEL_PRIMARY   : 'Electron Energy',
    LABEL_SECONDARY : 'Hadronic Energy',
    LABEL_TOTAL     : 'Neutrino Energy',
}

units_map = {
    LABEL_PRIMARY   : 'GeV',
    LABEL_SECONDARY : 'GeV',
    LABEL_TOTAL     : 'GeV',
}

base_map = {
    LABEL_PRIMARY   : "nueRecoLepE",
    LABEL_SECONDARY : "nueRecoHadE",
    LABEL_TOTAL     : "nueRecoE",
}

plot_specs_nue_rel_vs_true = {
    LABEL_PRIMARY   : get_plot_spec_rel_res_vs_true(
        "Electron", range_y = (-0.1, 0.1)
    ),
    LABEL_SECONDARY : get_plot_spec_rel_res_vs_true(
        "Hadronic", range_x = (0, 5)
    ),
    LABEL_TOTAL     : get_plot_spec_rel_res_vs_true(
        "Neutrino", range_y = (-0.25, 0.25)
    ),
}

plot_specs_nue_hist = {
    LABEL_PRIMARY   : get_plot_spec_hist("Electron", range = (0, 5)),
    LABEL_SECONDARY : get_plot_spec_hist("Hadronic", range = (0, 5)),
    LABEL_TOTAL     : get_plot_spec_hist("Neutrino", range = (0, 5))
}

plot_specs_nue_binstats_abs = {
    LABEL_PRIMARY   : get_plot_spec_binstat_abs(
        "Electron", range = (0, 3), bins = 50
    ),
    LABEL_SECONDARY : get_plot_spec_binstat_abs(
        "Hadronic", range = (0, 2), bins = 20
    ),
    LABEL_TOTAL     : get_plot_spec_binstat_abs(
        "Neutrino", range = (0, 5), bins = 50
    ),
}

plot_specs_nue_binstats_rel = {
    LABEL_PRIMARY   : get_plot_spec_binstat_rel(
        "Electron", range = (0, 3), bins = 50
    ),
    LABEL_SECONDARY : get_plot_spec_binstat_rel(
        "Hadronic", range = (0, 2), bins = 20
    ),
    LABEL_TOTAL     : get_plot_spec_binstat_rel(
        "Neutrino", range = (0, 5), bins = 50
    ),
}

plot_specs_nue_fom = {
    LABEL_PRIMARY   : get_plot_spec_fom("Electron"),
    LABEL_SECONDARY : get_plot_spec_fom("Hadronic"),
    LABEL_TOTAL     : get_plot_spec_fom("Neutrino"),
}

