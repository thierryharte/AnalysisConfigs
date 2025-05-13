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

from workflow import VBFHH4bProcessor
from custom_cuts import vbf_hh4b_presel, vbf_hh4b_presel_tight

from configs.HH4b_common.categories_definitions_common import define_categories

from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
    bkg_morphing_dnn_weightRun2,
)
from configs.HH4b_common.configurator_options import (
    get_variables_dict,
    get_columns_list,
    create_DNN_columns_list,
)
from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    bkg_morphing_dnn_input_variables_altOrder,
    sig_bkg_dnn_input_variables,
)
from configs.VBF_HH4b.onnx_models import onnx_model_dict

import configs.HH4b_common.custom_cuts_common as cuts


localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = "2022_postEE"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration_withoutVariations.yaml",
    update=True,
)


print("onnx_model_dict", onnx_model_dict)


VBF_PARTON_MATCHING = False
TIGHT_CUTS = False
CLASSIFICATION = False
SAVE_CHUNK = False
VBF_PRESEL = False
SEMI_TIGHT_VBF = True
DNN_VARIABLES = True
VR1 = False
BLIND = True if onnx_model_dict["SIG_BKG_DNN"] else False
RUN2 = False

workflow_options = {
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "classification": CLASSIFICATION,
    "tight_cuts": TIGHT_CUTS,
    "fifth_jet": "pt",
    "vbf_parton_matching": VBF_PARTON_MATCHING,
    "vbf_presel": VBF_PRESEL,
    "donotscale_sumgenweights": True,
    "semi_tight_vbf": SEMI_TIGHT_VBF,
    "DNN_VARIABLES": DNN_VARIABLES,
    "RUN2": RUN2,
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
    [vbf_hh4b_presel if TIGHT_CUTS is False else vbf_hh4b_presel_tight]
    if VBF_PRESEL
    else [cuts.hh4b_presel if TIGHT_CUTS is False else cuts.hh4b_presel_tight]
)

## Define the samples to process
sample_list = [
    # "DATA_JetMET_JMENano_C_skimmed",
    # "DATA_JetMET_JMENano_D_skimmed",
    "DATA_JetMET_JMENano_E_skimmed",
    "DATA_JetMET_JMENano_F_skimmed",
    "DATA_JetMET_JMENano_G_skimmed",
    "GluGlutoHHto4B_spanet_skimmed",
    # "GluGlutoHHto4B",
    # "VBF_HHto4B",
]

## Define the categories to save
categories_dict = define_categories(
    bkg_morphing_dnn=workflow_options["BKG_MORPHING_DNN"],
    blind=BLIND,
    spanet=workflow_options["SPANET"],
    vbf_ggf_dnn=workflow_options["VBF_GGF_DNN"],
    run2=RUN2,
    vr1=VR1,
)
print("categories_dict", categories_dict)

## VBF SPECIFIC REGIONS
# **{f"4b_semiTight_LeadingPt_region": [hh4b_4b_region, semiTight_leadingPt]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]}
# **{"4b_VBFtight_region": [hh4b_4b_region, VBFtight_region]},
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
else:
    column_list = get_columns_list()
    column_listRun2 = get_columns_list()

#Add special columns
if workflow_options["SIG_BKG_DNN"] and workflow_options["SPANET"]:
    column_list += get_columns_list({"events": ["sig_bkg_dnn_score"]})
if workflow_options["SIG_BKG_DNN"] and RUN2:
    column_list += get_columns_list({"events": ["sig_bkg_dnn_scoreRun2"]})
if workflow_options["BKG_MORPHING_SPREAD_DNN"] and workflow_options["SPANET"]:
    column_list += get_columns_list({"events": ["bkg_morphing_spread_dnn_weights"]})
if workflow_options["BKG_MORPHING_SPREAD_DNN"] and RUN2:
    column_list += get_columns_list({"events": ["bkg_morphing_spread_dnn_weightsRun2"]})

# Define the per category columns
bycategory_column_dict = {}
for category in categories_dict.keys():
    if "Run2" in category:
        bycategory_column_dict[category] = column_listRun2
    else:
        bycategory_column_dict[category] = column_list


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
            # f"{localdir}/../HH4b_common/datasets/QCD.json",
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local_rucio.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_SM_local_rucio_redirector.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_test.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
        ],
        "filter": {
            "samples": sample_list,
            "samples_exclude": [],
            # "year": [year],
        },
        "subsamples": {},
    },
    workflow=VBFHH4bProcessor,
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
            "bycategory": bycategory_column_dict,
        },
        "bysample": {},
    },
)
