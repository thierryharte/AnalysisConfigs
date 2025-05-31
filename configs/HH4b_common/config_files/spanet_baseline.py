from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
)


onnx_model_dict = {
    "spanet": "",
    "vbf_ggf_dnn": "",
    "bkg_morphing_dnn": "",
    "sig_bkg_dnn": "",
    "bkg_morphing_spread_dnn": "",
}


onnx_model_dict  |= {
    "spanet": "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s160_btag.onnx",
    # "spanet": "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx",
    #"bkg_morphing_dnn": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx",
    #"bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/AN_1e-2_noDropout_e20lrdrop95/state_dict/ratio/average_model_from_onnx.onnx",
    #"bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/SPANET_minDelta1em5_LRdropout/state_dict/ratio/average_model_from_onnx.onnx",
    #"bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_morphing/SPANET_baseline_20_runs_fixed/best_models/ratio/average_model_from_onnx.onnx",
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/SPANET_baseline_20_runs_postEE/best_models/ratio/average_model_from_onnx.onnx", # --> trained on postEE only
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/SPANET_baseline_norm_e5drop75_postEE/state_dict/model_best_epoch_18.onnx",
    # "sig_bkg_dnn": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_SvsB/model_fold0.onnx",
}

# Loading default parameters
# HIGGS_PARTON_MATCHING = False
# VBF_PARTON_MATCHING = False
# TIGHT_CUTS = False
# CLASSIFICATION = False
# SAVE_CHUNK = False
# VBF_PRESEL = False
# SEMI_TIGHT_VBF = True
# DNN_VARIABLES = True
# RUN2 = False
# BLIND = True if onnx_model_dict["SIG_BKG_DNN"] else False
# VR1 = False
# RANDOM_PT = False
# DELTA_PROB=False



config_options_dict = {
    "higgs_parton_matching": False,
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "classification": False,
    "save_chunk": False,
    "vbf_presel": False,
    "semi_tight_vbf": True,
    "dnn_variables": True,
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": True,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
} | onnx_model_dict