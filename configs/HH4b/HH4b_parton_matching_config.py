# for spanet evaluation: pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py
import os
import sys

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
# from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.weights.common.common import common_weights

from workflow import HH4bbQuarkMatchingProcessor

from configs.HH4b_common.custom_cuts_common import (
    hh4b_presel,
    hh4b_presel_tight,
    hh4b_4b_region,
    hh4b_2b_region,
    hh4b_signal_region,
    hh4b_control_region,
    hh4b_signal_region_run2,
    hh4b_control_region_run2,
)

from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
    bkg_morphing_dnn_weightRun2,
)
from configs.HH4b_common.configurator_options import (
    get_variables_dict,
    get_columns_list,
    create_DNN_columns_list,
)
from configs.HH4b_common.dnn_input_variables import bkg_morphing_dnn_input_variables

from configs.HH4b_common.configurator_options import DEFAULT_COLUMNS

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

CLASSIFICATION = False
TIGHT_CUTS = False
RANDOM_PT = False
SAVE_CHUNK = False
DNN_VARIABLES = True
RUN2 = True

print("CLASSIFICATION ", CLASSIFICATION)
print("TIGHT_CUTS ", TIGHT_CUTS)
print("RANDOM_PT ", RANDOM_PT)

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2022_postEE", "2022_preEE", "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

onnx_model_dict={
    "SPANET": "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx",
    #"SPANET": "",
    # "SPANET": "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx",
    "VBF_GGF_DNN": "",
    # "VBF_GGF_DNN":"/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx",
    #"BKG_MORPHING_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx",
    #"BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/AN_1e-2_noDropout_e20lrdrop95/state_dict/ratio/average_model_from_onnx.onnx",
    "BKG_MORPHING_DNN": "/work/tharte/datasets/ML_pytorch/out/DHH_method_20_runs/best_models/ratio/average_model_from_onnx.onnx",
    "SIG_BKG_DNN": "",
    # "SIG_BKG_DNN": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_SvsB/model_fold0.onnx",
}

print(onnx_model_dict)

## Defining the used samples
samples_list = [
            "DATA_JetMET_JMENano_C_skimmed",
            "DATA_JetMET_JMENano_D_skimmed",
            "DATA_JetMET_JMENano_E_skimmed",
            "DATA_JetMET_JMENano_F_skimmed",
            "DATA_JetMET_JMENano_G_skimmed",
            # "DATA_JetMET_JMENano_2023_Cv1_skimmed",
            # "DATA_JetMET_JMENano_2023_Cv2_skimmed",
            # "DATA_ParkingHH_2023_Cv3",
            # "DATA_ParkingHH_2023_Cv4",
            # "DATA_ParkingHH_2023_Dv1",
            # "DATA_ParkingHH_2023_Dv2",
        ]

## General workflow options
workflow_options = {
        "parton_jet_min_dR": 0.4,
        "max_num_jets": 5,
        "which_bquark": "last",
        "classification": CLASSIFICATION, 
        "tight_cuts": TIGHT_CUTS,
        "fifth_jet": "pt",
        "random_pt": RANDOM_PT,
        "rand_type": 0.3,
        "DNN_VARIABLES": DNN_VARIABLES,
        "RUN2": RUN2,
        "pad_value":-999.0,
    }
workflow_options.update(
    onnx_model_dict
)
if SAVE_CHUNK:
    workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/training_samples/GluGlutoHHto4B_spanet_loose_03_17"
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_btag_ordering"
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_no_btag"


variables_dict = get_variables_dict(True, CLASSIFICATION, RANDOM_PT, False)


## Rather complicated build of the different regions depending on what setting is activated
column_dict = DEFAULT_COLUMNS
event_cols = []
if CLASSIFICATION:
    event_cols += ["best_pairing_probability", "second_best_pairing_probability", "Delta_pairing_probabilities"]
if RANDOM_PT:
    for key in column_dict.keys():
        if not "Matched" in key:
            column_dict[key] += ["pt_orig", "mass_orig"]
    column_dict["events"] = ["random_pt_weights"]

column_list=create_DNN_columns_list(False, not SAVE_CHUNK, bkg_morphing_dnn_input_variables)
column_listRun2=create_DNN_columns_list(True, not SAVE_CHUNK, bkg_morphing_dnn_input_variables)

# Define categories based on what parameters are activated
categories_dict = {
        "4b_control_region": [hh4b_4b_region, hh4b_control_region],
        "2b_control_region_preW": [hh4b_2b_region, hh4b_control_region],
        #
        "4b_signal_region": [hh4b_4b_region, hh4b_signal_region],
        "2b_signal_region_preW": [hh4b_2b_region, hh4b_signal_region],
    }
if onnx_model_dict["BKG_MORPHING_DNN"]:
    categories_reweight = {
        "2b_control_region_postW": [hh4b_2b_region, hh4b_control_region],
        "2b_signal_region_postW": [hh4b_2b_region, hh4b_signal_region],
        }
    categories_dict |= categories_reweight
if RUN2:
    categories_run2 = {
        "4b_control_regionRun2": [hh4b_4b_region, hh4b_hh4b_control_region_run2],
        "2b_control_region_preWRun2": [hh4b_2b_region, hh4b_hh4b_control_region_run2],
        "4b_signal_regionRun2": [hh4b_4b_region, hh4b_hh4b_signal_region_run2],
        "2b_signal_region_preWRun2": [hh4b_2b_region, hh4b_hh4b_signal_region_run2],
        }
    if onnx_model_dict["BKG_MORPHING_DNN"]:
        categories_reweight = {
            "2b_control_region_postWRun2": [hh4b_2b_region, hh4b_hh4b_control_region_run2],
            "2b_signal_region_postWRun2": [hh4b_2b_region, hh4b_hh4b_signal_region_run2],
            }
        categories_dict |= categories_reweight
    categories_dict |= categories_run2

print(categories_dict.keys())

bycategory_weight_dict = {}
bycategory_column_dict = {}
for key in categories_dict.keys():
    if "postW" in key:
        if "Run2" in key:
            bycategory_weight_dict[key] = ["bkg_morphing_dnn_weight" ]
        else:
            bycategory_weight_dict[key] = ["bkg_morphing_dnn_weightRun2"]
    if "Run2" in key:
        bycategory_column_dict[key] = column_listRun2
    else:
        bycategory_column_dict[key] = column_list

### Filling a dictionary with all the samples and add the weight dictionary to them. (the inclusive way did not really work).
#weight_sample_dict = {}
#for sample in samples_list:
#    weight_sample_dict[sample] = {"inclusive" : [], "bycategory": weight_dict} 
#
#
#print(weight_sample_dict)

cfg = Configurator(
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_2022_2023_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/signal_ggF_HH4b.json",
            f"{localdir}/datasets/DATA_JetMET_skimmed.json",
            f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/SPANet_classification.json",
            f"{localdir}/datasets/signal_ggF_HH4b_local.json",
            f"{localdir}/datasets/signal_VBF_HH4b_local.json",
            f"{localdir}/datasets/DATA_ParkingHH.json",
            f"{localdir}/datasets/DATA_JetMET.json",
        ],
        "filter": {
            "samples": samples_list,
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessor,
    workflow_options=workflow_options,
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        hh4b_presel if TIGHT_CUTS is False else hh4b_presel_tight
    ],
    categories=categories_dict,
    weights_classes=common_weights
    + [bkg_morphing_dnn_weight, bkg_morphing_dnn_weightRun2],
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": bycategory_weight_dict
        },
        "bysample": {}
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables=variables_dict,
    columns={
        "common": {
            "inclusive": [],
            "bycategory": bycategory_column_dict,
        },
        "bysample": {
        },
    },
)
