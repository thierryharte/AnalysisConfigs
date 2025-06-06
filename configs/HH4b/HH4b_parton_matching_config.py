import os
from collections import defaultdict

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.weights.common.common import common_weights

from workflow import HH4bbQuarkMatchingProcessor

from configs.HH4b_common.categories_definitions_common import define_categories, define_single_category

from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
    bkg_morphing_dnn_weightRun2,
)
from configs.HH4b_common.configurator_options import (
    SPANET_TRAINING_DEFAULT_COLUMNS,
    get_variables_dict,
    get_columns_list,
    create_DNN_columns_list,
)
from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    bkg_morphing_dnn_input_variables_altOrder,
    sig_bkg_dnn_input_variables,
)
from configs.HH4b_common.config_files.__config_file__ import (
    onnx_model_dict,
    HIGGS_PARTON_MATCHING,
    VBF_PARTON_MATCHING,
    TIGHT_CUTS,
    CLASSIFICATION,
    SAVE_CHUNK,
    VBF_PRESEL,
    SEMI_TIGHT_VBF,
    DNN_VARIABLES,
    RUN2,
    VR1,
    RANDOM_PT,
    BLIND,
)

import configs.HH4b_common.custom_cuts_common as cuts


localdir = os.path.dirname(os.path.abspath(__file__))


# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2022_postEE", "2022_preEE"]#, "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration_withoutVariations.yaml",
    update=True,
)


print("onnx_model_dict", onnx_model_dict)

workflow_options = {
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "classification": CLASSIFICATION,
    "tight_cuts": TIGHT_CUTS,
    "fifth_jet": "pt",
    "donotscale_sumgenweights": True,
    "DNN_VARIABLES": DNN_VARIABLES,
    "RUN2": RUN2,
    "random_pt": RANDOM_PT,
    "rand_type": 0.3,
    "pad_value": -999.0,
}
workflow_options.update(onnx_model_dict)

if SAVE_CHUNK:
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/training_samples/GluGlutoHHto4B_spanet_loose_03_17"
    pass


## Define the variables to save
# variables_dict = get_variables_dict(
#     CLASSIFICATION=CLASSIFICATION,
#     VBF_VARIABLES=False,
#     BKG_MORPHING=True if onnx_model_dict["BKG_MORPHING_DNN"] else False,
# )
variables_dict = {}

## Define the preselection to apply
preselection = (
    [cuts.hh4b_presel if TIGHT_CUTS is False else cuts.hh4b_presel_tight]
)

## Defining the used samples
sample_list = [
            # "DATA_JetMET_JMENano_C_skimmed",
            # "DATA_JetMET_JMENano_D_skimmed",
            # "DATA_JetMET_JMENano_E_skimmed",
            # "DATA_JetMET_JMENano_F_skimmed",
            # "DATA_JetMET_JMENano_G_skimmed",
            "GluGlutoHHto4B_spanet",
            "GluGlutoHHto4B",
            # "DATA_JetMET_JMENano_2023_Cv1_skimmed",
            # "DATA_JetMET_JMENano_2023_Cv2_skimmed",
            # "DATA_ParkingHH_2023_Cv3",
            # "DATA_ParkingHH_2023_Cv4",
            # "DATA_ParkingHH_2023_Dv1",
            # "DATA_ParkingHH_2023_Dv2",
        ]

## Define the categories to save
categories_dict = define_categories(
    bkg_morphing_dnn=workflow_options["BKG_MORPHING_DNN"],
    blind=BLIND,
    spanet=workflow_options["SPANET"],
    run2=RUN2,
    vr1=VR1,
)
# AKA if no model is applied
if [model=="" for model in onnx_model_dict]:
    categories_dict = define_single_category("4b_region")

print("categories_dict", categories_dict)

## VBF SPECIFIC REGIONS
# **{f"4b_semiTight_LeadingPt_region": [hh4b_4b_region, semiTight_leadingPt]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]}
# **{"4b_VBFtight_region": [hh4b_4b_region, vbf_wrapper()]},
#
# **{
#     f"4b_VBFtight_{list(ab[0].keys())[i]}_region": [
#         hh4b_4b_region,
#         vbf_wrapper(ab[i]),
#     ]
#     for i in range(0, 6)
# },
#
# **{"4b_VBF_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region]},
# **{"4b_VBF_region": [hh4b_4b_region, VBF_region]},
# **{f"4b_VBF_0{i}qvg_region": [hh4b_4b_region, VBF_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},
# **{f"4b_VBF_0{i}qvg_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},


## Define the columns to save
assert not (RANDOM_PT and RUN2)
if DNN_VARIABLES:
    total_input_variables = (
        sig_bkg_dnn_input_variables
        | bkg_morphing_dnn_input_variables
        | {"year": ["events", "year"]}
    )
    print(total_input_variables)

    column_list = create_DNN_columns_list(
        False, not SAVE_CHUNK, total_input_variables, btag=False
    )
    column_listRun2 = create_DNN_columns_list(
        True, not SAVE_CHUNK, total_input_variables, btag=False
    )
elif [model=="" for model in onnx_model_dict]:
    column_list = get_columns_list(SPANET_TRAINING_DEFAULT_COLUMNS)
    if RANDOM_PT:
        column_list += get_columns_list({"events": ["random_pt_weights"]})
else:
    column_list = get_columns_list()
    column_listRun2 = get_columns_list()

# Add special columns
if workflow_options["SIG_BKG_DNN"] and workflow_options["SPANET"]:
    column_list += get_columns_list({"events": ["sig_bkg_dnn_score"]})
if workflow_options["SIG_BKG_DNN"] and RUN2:
    column_listRun2 += get_columns_list({"events": ["sig_bkg_dnn_scoreRun2"]})

bysample_bycategory_column_dict = {}
for sample in sample_list:
    bysample_bycategory_column_dict[sample] = {
        "inclusive": [],
        "bycategory": {},
    }
    for category in categories_dict.keys():
        if "Run2" in category:
            # if "DATA" in sample:
            #     column_listRun2 += get_columns_list({"events": ["bkg_morphing_spread_dnn_weightsRun2"]})

            bysample_bycategory_column_dict[sample]["bycategory"][category] = (
                column_listRun2
                + (
                    get_columns_list(
                        {"events": ["bkg_morphing_spread_dnn_weightsRun2"]}
                    )
                    if "DATA" in sample and workflow_options["BKG_MORPHING_SPREAD_DNN"] and "postW" in category 
                    else []
                )
            )
        else:
            # if "DATA" in sample:
            #     column_list += get_columns_list({"events": ["bkg_morphing_spread_dnn_weights"]})
            bysample_bycategory_column_dict[sample]["bycategory"][category] = (
                column_list
                + (
                    get_columns_list(
                        {"events": ["bkg_morphing_spread_dnn_weights"]}
                    )
                    if "DATA" in sample and workflow_options["BKG_MORPHING_SPREAD_DNN"] and "postW" in category
                    else []
                )
            )
print("bysample_bycategory_column_dict", bysample_bycategory_column_dict)

## Define the weights to apply
bysample_bycategory_weight_dict = {}
for sample in sample_list:
    if "DATA" in sample:
        bysample_bycategory_weight_dict[sample] = {"inclusive": [], "bycategory": {}}
        for category in categories_dict.keys():
            if "postW" in category:
                if "Run2" in category:
                    bysample_bycategory_weight_dict[sample]["bycategory"][category] = [
                        "bkg_morphing_dnn_weightRun2"
                    ]
                else:
                    bysample_bycategory_weight_dict[sample]["bycategory"][category] = [
                        "bkg_morphing_dnn_weight"
                    ]

print("bysample_bycategory_weight_dict", bysample_bycategory_weight_dict)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
            #f"{localdir}/../HH4b_common/datasets/QCD.json",
            #f"{localdir}/../HH4b_common/datasets/SPANet_classification.json",
            #f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local.json",
            #f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_local.json",
            f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET.json",
        ],
        "filter": {
            "samples": sample_list,
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
    preselections=preselection,
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
            "bycategory": {},
        },
        "bysample": bysample_bycategory_weight_dict,
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
            "bycategory": {},
        },
        "bysample": bysample_bycategory_column_dict,
    },
)
