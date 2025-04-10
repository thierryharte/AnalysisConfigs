import os

localdir = os.path.dirname(os.path.abspath(__file__))

common_params = f"{localdir}/../HH4b_common/params_common/"

onnx_model_dict = {
    # "SPANET": f"{common_params}/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx", # pt vary 0.3, 1.7
    "SPANET": "",

    # "SPANET": "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx",
    "VBF_GGF_DNN": "",
    # "VBF_GGF_DNN":"/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx",
    "BKG_MORPHING_DNN": "",
    # "BKG_MORPHING_DNN": f"{common_params}/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_noEarlyStopping_20folds_average_model_from_onnx.onnx",  # 20 k-folds, early stopping, 1e-5 minDelta, spanet
    # "BKG_MORPHING_DNN": "/t3home/mmalucch/ML_pytorch/out/DNN_AN_VR1_1e-3_e20drop75_minDelta1em5_SPANet_oversample_split/state_dict/average_model_from_onnx.onnx",  # VR1 train, early stopping, 1e-5 minDelta, spanet
    # "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_run2/state_dict/average_model_from_onnx.onnx",  # Run2 CR train, early stopping, 1e-5 minDelta
    # "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_minDelta1em5/batch06/best_models/average_model_from_onnx.onnx",  # 20 k-folds, early stopping, 1e-5 minDelta, spanet
    # "BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/AN_1e-2_noDropout_e20lrdrop95/state_dict/ratio/average_model_from_onnx.onnx",
    # "BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/test_multiple_coffea/state_dict/model_40_state_dict.onnx", # thierry's model trained on 22C-22D-22E
    # "BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/batch01/best_models/output/average_model_from_onnx.onnx", # thierry's model trained on 22E
    # "BKG_MORPHING_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx", # soumya's model
    "SIG_BKG_DNN": "",
    # "SIG_BKG_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_SvsB/model_fold0.onnx",
}