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
    "SPANET": "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx", # spanet pt vary 0.3, 1.7
    "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE_DeltaProb/best_models/average_model_from_onnx.onnx",  # DeltaProb, only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary 
}

HIGGS_PARTON_MATCHING=False
VBF_PARTON_MATCHING = False
TIGHT_CUTS = False
CLASSIFICATION = False
SAVE_CHUNK = False
VBF_PRESEL = False
SEMI_TIGHT_VBF = True
DNN_VARIABLES = True
RUN2 = False
VR1 = False
RANDOM_PT = False
BLIND = True if onnx_model_dict["SIG_BKG_DNN"] else False
DELTA_PROB=True