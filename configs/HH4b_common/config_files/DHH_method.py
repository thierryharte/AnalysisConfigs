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
    # "VBF_GGF_DNN":"/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx",
    #"BKG_MORPHING_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx",
    #"BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/AN_1e-2_noDropout_e20lrdrop95/state_dict/ratio/average_model_from_onnx.onnx",
    #"BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/bkg_morphing/DHH_method_20_runs_1e-3_e20drop75_minDelta1em5/best_models/ratio/average_model_from_onnx.onnx",
    "BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/average_model_from_onnx.onnx", # --> training on postEE
    "BKG_MORPHING_SPREAD_DNN": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/all_ratios_model_onnx.onnx", # --> training on postEE
    #"SIG_BKG_DNN": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DHH_method_norm_e5drop75_fixed/state_dict/model_best_epoch_18.onnx",
    #"SIG_BKG_DNN": "",
    "SIG_BKG_DNN": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DHH_method_norm_e5drop75_postEE/state_dict/model_best_epoch_18.onnx",
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
RUN2 = True
BLIND = True if onnx_model_dict["SIG_BKG_DNN"] else False
VR1 = False
RANDOM_PT = False
DELTA_PROB=False