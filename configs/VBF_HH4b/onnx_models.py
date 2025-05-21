import os

localdir = os.path.dirname(os.path.abspath(__file__))

common_params = f"{localdir}/../HH4b_common/params_common/"

onnx_model_dict = {
    "SPANET": "",
    "VBF_GGF_DNN": "",
    "BKG_MORPHING_DNN": "",
    "BKG_MORPHING_SPREAD_DNN": "",
    "SIG_BKG_DNN": "",
}


onnx_model_dict  |= {
    "SPANET": f"{common_params}/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx", # spanet pt vary 0.3, 1.7
    # "SPANET": "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s160_btag.onnx", # thierry's model, spanet baseline
    # "SPANET": "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx", # ruth's model
    #
    # "VBF_GGF_DNN":"/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx",
    #
    ### SPANET pt vary 0.3, 1.7
    # "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_oversample_split_20folds/batch06/best_models/average_model_from_onnx.onnx",  # full 2022, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary
    "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE/best_models/average_model_from_onnx.onnx",  # only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary 
    # "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_VR1_1e-3_e20drop75_minDelta1em5_SPANet_oversample_split/state_dict/average_model_from_onnx.onnx",  # VR1 train, early stopping, 1e-5 minDelta, spanet pt vary
    ### Run2
    # "BKG_MORPHING_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx", # soumya's model
    # "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_run2/state_dict/average_model_from_onnx.onnx",  # Run2 CR train, early stopping, 1e-5 minDelta
    # "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_postEE_matteo/best_models/average_model_from_onnx.onnx",  #  only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, run2
    ### SPANET baseline
    # "BKG_MORPHING_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_baseline_postEE_matteo/best_models/average_model_from_onnx.onnx",  #  only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet baseline
    
    ### SPANET pt vary 0.3, 1.7
    "BKG_MORPHING_SPREAD_DNN": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE/best_models/all_ratios_model_onnx.onnx",  # only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet, all ration spread model
    
    # "SIG_BKG_DNN": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75/state_dict/model_best_epoch_25.onnx", # morphing full 2022, FixedPairingMorphing, FixedModelMorphing
    # "SIG_BKG_DNN": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_postEE/state_dict/model_best_epoch_13.onnx", # only 2022postEE, FixedPairingMorphing, FixedModelMorphing
    "SIG_BKG_DNN": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_postEE_allklambda/state_dict/model_best_epoch_16.onnx", # allklambda, only 2022postEE, FixedPairingMorphing, FixedModelMorphing
    # "SIG_BKG_DNN": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_WrongMorphing/state_dict/model_best_epoch_21.onnx", # pytorch training, FixedPairingMorphing, WrongModelMorphing
    # "SIG_BKG_DNN": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_WrongPairingMorphing/state_dict/model_best_epoch_14.onnx", # pytorch training, WrongPairingMorphing
    # "SIG_BKG_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_SvsB/model_fold0.onnx", # soumya's model
}