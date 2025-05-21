import os

localdir = os.path.dirname(os.path.abspath(__file__))

common_params = f"{localdir}/../HH4b_common/params_common/"

onnx_model_dict = {
    "SPANET": "",
    "VBF_GGF_DNN": "",
    "BKG_MORPHING_DNN": "",
    "SIG_BKG_DNN": "",
    "BKG_MORPHING_SPREAD_DNN": "",
}


onnx_model_dict  |= {
    "SPANET": "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx",
    "VBF_GGF_DNN": "",
    "BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/SPANET_ptFlat_20_runs_postEE/best_models/ratio/average_model_from_onnx.onnx", # --> trained on postEE only
    #"SIG_BKG_DNN": "",
    "SIG_BKG_DNN": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/SPANET_ptflat_norm_e5drop75_postEE/state_dict/model_best_epoch_23.onnx",
}

# Loading default parameters
HIGGS_PARTON_MATCHING = False
VBF_PARTON_MATCHING = False
TIGHT_CUTS = False
CLASSIFICATION = False
SAVE_CHUNK = False
VBF_PRESEL = False
SEMI_TIGHT_VBF = True
DNN_VARIABLES = True
RUN2 = False
BLIND = True if onnx_model_dict["SIG_BKG_DNN"] else False
VR1 = False
RANDOM_PT = False
DELTA_PROB=False