import os

localdir = os.path.dirname(os.path.abspath(__file__))

onnx_model_dict = {
    "spanet": "",
    "vbf_ggf_dnn": "",
    "bkg_morphing_dnn": "",
    "sig_bkg_dnn": "",
    "bkg_morphing_spread_dnn": "",
}

# Loading default parameters
# HIGGS_PARTON_MATCHING = False
# VBF_PARTON_MATCHING = False
# TIGHT_CUTS = False
# CLASSIFICATION = False
# SAVE_CHUNK = False
# VBF_PRESEL = False
# SEMI_TIGHT_VBF = False
# DNN_VARIABLES = False
# RUN2 = False
# BLIND = True if onnx_model_dict["SIG_BKG_DNN"] else False
# VR1 = False
# RANDOM_PT = True
# DELTA_PROB=False


config_options_dict = {
    "higgs_parton_matching": False,
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "classification": False,
    "save_chunk": False,
    "vbf_presel": False,
    "semi_tight_vbf": True,
    "dnn_variables": False,
    "run2": False,
    "vr1": False,
    "random_pt": True,
    "rand_type": 0.3,
    "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "sig_bkg_dnn_input_variables": None,
    "bkg_morphing_dnn_input_variables": None,
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": True,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
} | onnx_model_dict
    