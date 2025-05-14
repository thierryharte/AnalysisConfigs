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
    "SPANET": "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s160_btag.onnx",
    # "SPANET": "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx",
    #"BKG_MORPHING_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx",
    #"BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/AN_1e-2_noDropout_e20lrdrop95/state_dict/ratio/average_model_from_onnx.onnx",
    #"BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/SPANET_minDelta1em5_LRdropout/state_dict/ratio/average_model_from_onnx.onnx",
    #"BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/bkg_morphing/SPANET_baseline_20_runs_fixed/best_models/ratio/average_model_from_onnx.onnx",
    "BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/SPANET_baseline_20_runs_postEE/best_models/ratio/average_model_from_onnx.onnx", # --> trained on postEE only
    "SIG_BKG_DNN": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/SPANET_baseline_norm_e5drop75_postEE/state_dict/model_best_epoch_18.onnx",
    # "SIG_BKG_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_SvsB/model_fold0.onnx",
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
