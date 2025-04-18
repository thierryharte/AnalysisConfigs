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

from configs.HH4b_common.custom_cuts_common import (
    hh4b_presel,
    hh4b_presel_tight,
    hh4b_4b_region,
    hh4b_2b_region,
    hh4b_signal_region,
    hh4b_control_region,
    hh4b_signal_region_run2,
    hh4b_control_region_run2,
    hh4b_VR1_signal_region,
    hh4b_VR1_control_region,
    hh4b_VR1_signal_region_run2,
    hh4b_VR1_control_region_run2,
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
from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    bkg_morphing_dnn_input_variables_altOrder,
    sig_bkg_dnn_input_variables,
)
from configs.VBF_HH4b.onnx_models import onnx_model_dict


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
    f"{localdir}/params/jets_calibration.yaml",
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
RUN2 = True
VR1 = False


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


# variables_dict = get_variables_dict(
#     CLASSIFICATION=CLASSIFICATION,
#     VBF_VARIABLES=False,
#     BKG_MORPHING=True if onnx_model_dict["BKG_MORPHING_DNN"] else False,
# )
variables_dict = {}

if DNN_VARIABLES:
    total_input_variables = (
        sig_bkg_dnn_input_variables | bkg_morphing_dnn_input_variables
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


preselection = (
    [vbf_hh4b_presel if TIGHT_CUTS is False else vbf_hh4b_presel_tight]
    if VBF_PRESEL
    else [hh4b_presel if TIGHT_CUTS is False else hh4b_presel_tight]
)

sample_list = [
    # "DATA_JetMET_JMENano_C_skimmed",
    # "DATA_JetMET_JMENano_D_skimmed",
    # "DATA_JetMET_JMENano_E_skimmed",
    # "DATA_JetMET_JMENano_F_skimmed",
    # "DATA_JetMET_JMENano_G_skimmed",
    "GluGlutoHHto4B",
    # "VBF_HHto4B",
]

if not VR1:
    categories_dict = {
        "4b_control_region": [hh4b_4b_region, hh4b_control_region],
        "2b_control_region_preW": [hh4b_2b_region, hh4b_control_region],
        "2b_control_region_postW": [hh4b_2b_region, hh4b_control_region],
        #
        "4b_signal_region": [hh4b_4b_region, hh4b_signal_region],
        "2b_signal_region_preW": [hh4b_2b_region, hh4b_signal_region],
        "2b_signal_region_postW": [hh4b_2b_region, hh4b_signal_region],
        #
        # "4b_region": [hh4b_4b_region],
        # "2b_region": [hh4b_2b_region],
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
    }
    if RUN2:
        categories_dictRun2 = {
            "4b_control_regionRun2": [hh4b_4b_region, hh4b_control_region_run2],
            "2b_control_region_preWRun2": [hh4b_2b_region, hh4b_control_region_run2],
            "2b_control_region_postWRun2": [hh4b_2b_region, hh4b_control_region_run2],
            "4b_signal_regionRun2": [hh4b_4b_region, hh4b_signal_region_run2],
            "2b_signal_region_preWRun2": [hh4b_2b_region, hh4b_signal_region_run2],
            "2b_signal_region_postWRun2": [hh4b_2b_region, hh4b_signal_region_run2],
        }
        categories_dict = categories_dict | categories_dictRun2
else:
    categories_dict = {
        "4b_VR1_control_region": [hh4b_4b_region, hh4b_VR1_control_region],
        "2b_VR1_control_region_preW": [hh4b_2b_region, hh4b_VR1_control_region],
        "2b_VR1_control_region_postW": [hh4b_2b_region, hh4b_VR1_control_region],
        #
        "4b_VR1_signal_region": [hh4b_4b_region, hh4b_VR1_signal_region],
        "2b_VR1_signal_region_preW": [hh4b_2b_region, hh4b_VR1_signal_region],
        "2b_VR1_signal_region_postW": [hh4b_2b_region, hh4b_VR1_signal_region],
    }
    if RUN2:
        categories_dictRun2 = {
            "4b_VR1_control_regionRun2": [hh4b_4b_region, hh4b_VR1_control_region_run2],
            "2b_VR1_control_region_preWRun2": [
                hh4b_2b_region,
                hh4b_VR1_control_region_run2,
            ],
            "2b_VR1_control_region_postWRun2": [
                hh4b_2b_region,
                hh4b_VR1_control_region_run2,
            ],
            "4b_VR1_signal_regionRun2": [hh4b_4b_region, hh4b_VR1_signal_region_run2],
            "2b_VR1_signal_region_preWRun2": [
                hh4b_2b_region,
                hh4b_VR1_signal_region_run2,
            ],
            "2b_VR1_signal_region_postWRun2": [
                hh4b_2b_region,
                hh4b_VR1_signal_region_run2,
            ],
        }
        categories_dict = categories_dict | categories_dictRun2


bycategory_column_dict = {}
for category in categories_dict.keys():
    if "Run2" in category:
        bycategory_column_dict[category] = column_listRun2
    else:
        bycategory_column_dict[category] = column_list

bysample_bycategory_weight_dict = {}
for sample in sample_list:
    if "DATA" in sample:
        bysample_bycategory_weight_dict[sample] = {"inclusive": [], "bycategory": {}}
        for category in categories_dict.keys():
            if "postW" in category:
                if "Run2" in category:
                    bysample_bycategory_weight_dict[sample]["bycategory"][category] = [
                        "bkg_morphing_dnn_weight"
                    ]
                else:
                    bysample_bycategory_weight_dict[sample]["bycategory"][category] = [
                        "bkg_morphing_dnn_weightRun2"
                    ]

print(bysample_bycategory_weight_dict)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/signal_VBF_HH4b.json",
            # f"{localdir}/datasets/signal_ggF_HH4b_local.json",
            # f"{localdir}/datasets/signal_ggF_HH4b_local_rucio.json",
            f"{localdir}/datasets/signal_ggF_HH4b_altSite.json",
            # f"{localdir}/datasets/signal_ggF_HH4b_test.json",
            f"{localdir}/datasets/DATA_JetMET_skimmed.json",
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
    # weights_classes=common_weights
    # + [bkg_morphing_dnn_weight, bkg_morphing_dnn_weightRun2],
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {}, #bysample_bycategory_weight_dict,
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
