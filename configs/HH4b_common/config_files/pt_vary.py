import os

localdir = os.path.dirname(os.path.abspath(__file__))

onnx_model_dict = {
    "SPANET": "",
    "VBF_GGF_DNN": "",
    "BKG_MORPHING_DNN": "",
    "SIG_BKG_DNN": "",
    "BKG_MORPHING_SPREAD_DNN": "",
}

# Loading default parameters
HIGGS_PARTON_MATCHING = False
VBF_PARTON_MATCHING = False
TIGHT_CUTS = False
CLASSIFICATION = False
SAVE_CHUNK = False
VBF_PRESEL = False
SEMI_TIGHT_VBF = False
DNN_VARIABLES = False
RUN2 = False
BLIND = True if onnx_model_dict["SIG_BKG_DNN"] else False
VR1 = False
RANDOM_PT = True
DELTA_PROB=False