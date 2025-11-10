import os

from histConfigBtagEfficiency import btag_sf_hist
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters import defaults
from pocket_coffea.parameters.histograms import *
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.calibrators.legacy.legacy_calibrators import (
    JetsCalibrator,
    JetsPtRegressionCalibrator,
)
from workflow_btagSF_HH4b import HH4bCommonProcessor

import configs.HH4b_common.custom_cuts_common as cuts
from configs.HH4b_common.config_files.configurator_tools import (
    define_categories,
    define_single_category,
)
from configs.HH4b_common.config_files.spanet_ptflat_btag5WP import (
    config_options_dict,
    onnx_model_dict,
)

localdir = os.path.dirname(os.path.abspath(__file__))

for model in onnx_model_dict.keys():
    config_options_dict[model] = ""
    onnx_model_dict[model] = ""

# Loading default parameters
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir)


parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/variations.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withoutVariations_withJERC.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withVariations.yaml",
    update=True,
    )
parameters["run_period"] = "Run3"
year = ["2022_postEE", "2022_preEE"]  # , "2023_preBPix", "2023_postBPix"]
config_options_dict["num_bins"] = 20
# Defining the used samples
sample_ggF_list = [
    "GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-m2p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-m1p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-0p00_kt-0p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-3p50_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-4p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-3p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-2p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-1p50_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-0p50_kt-1p00_c2-0p00_skimmed",
]
sample_list = [
    # "DATA_JetMET_JMENano_C_skimmed",
    # "DATA_JetMET_JMENano_D_skimmed",
    # "DATA_JetMET_JMENano_E_skimmed",
    # "DATA_JetMET_JMENano_F_skimmed",
    # "DATA_JetMET_JMENano_G_skimmed",
    # "GluGlutoHHto4B_spanet_skimmed",
    # "GluGlutoHHto4B_spanet_skimmed_SM",
    # "GluGlutoHHto4B_spanet_skimmed",
    # "GluGlutoHHto4B",
    # "DATA_JetMET_JMENano_2023_Cv1_skimmed",
    # "DATA_JetMET_JMENano_2023_Cv2_skimmed",
    # "DATA_ParkingHH_2023_Cv3",
    # "DATA_ParkingHH_2023_Cv4",
    # "DATA_ParkingHH_2023_Dv1",
    # "DATA_ParkingHH_2023_Dv2",
] + sample_ggF_list

preselection = [
    (
        cuts.hh4b_presel_nobtag
        if config_options_dict["tight_cuts"] is False
        else cuts.hh4b_presel_tight
    )
]
categories_dict = define_categories(
    bkg_morphing_dnn=config_options_dict["bkg_morphing_dnn"],
    blind=config_options_dict["blind"],
    spanet=config_options_dict["spanet"],
    run2=config_options_dict["run2"],
    vr1=config_options_dict["vr1"],
)
if all([model == "" for model in onnx_model_dict.values()]):
    print("Didn't find any onnx model. Will choose region for SPANet training")
    categories_dict = define_single_category("inclusive")

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_SM.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_separateSamples.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
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

    workflow=HH4bCommonProcessor,
    workflow_options=config_options_dict,
    skim=cuts.skimming_cut_list,
    preselections=preselection,
    categories=categories_dict,
    calibrators=[JetsCalibrator, JetsPtRegressionCalibrator],

    weights_classes=common_weights,
    weights={
        "common": {
			"inclusive": ["genWeight", "lumi", "XS",
                          "pileup"
                          ],
            "bycategory": {}
        },
        "bysample": {}
    },

    variations={
        "weights": {
            "common": {
                "inclusive": ["pileup",
                              ],
                "bycategory": {}
            },
            "bysample": {
            }
        },
        "shape": {
            "common": {
                "inclusive": []
            }
        }
    },

    variables={
        **count_hist(name="nJets", coll="JetGood", bins=10, start=2, stop=12),
        **jet_hists(name="jet", coll="JetGood", pos=0),
        **jet_hists(name="jet", coll="JetGood", pos=1),
        **jet_hists(name="jet", coll="JetGood", pos=2),
        **jet_hists(name="jet", coll="JetGood", pos=3),
        **jet_hists(name="jet", coll="JetGood", pos=4),

        "bjets_deepJet_L_pt_eta_flav": btag_sf_hist("BJetGood_deepJet_L"),
        "bjets_deepJet_M_pt_eta_flav": btag_sf_hist("BJetGood_deepJet_M"),
        "bjets_deepJet_T_pt_eta_flav": btag_sf_hist("BJetGood_deepJet_T"),
        "bjets_deepJet_XT_pt_eta_flav": btag_sf_hist("BJetGood_deepJet_XT"),
        "bjets_deepJet_XXT_pt_eta_flav": btag_sf_hist("BJetGood_deepJet_XXT"),

        "bjets_particleNet_L_pt_eta_flav": btag_sf_hist("BJetGood_particleNet_L"),
        "bjets_particleNet_M_pt_eta_flav": btag_sf_hist("BJetGood_particleNet_M"),
        "bjets_particleNet_T_pt_eta_flav": btag_sf_hist("BJetGood_particleNet_T"),
        "bjets_particleNet_XT_pt_eta_flav": btag_sf_hist("BJetGood_particleNet_XT"),
        "bjets_particleNet_XXT_pt_eta_flav": btag_sf_hist("BJetGood_particleNet_XXT"),

        "bjets_robustParticleTransformer_L_pt_eta_flav": btag_sf_hist("BJetGood_robustParticleTransformer_L"),
        "bjets_robustParticleTransformer_M_pt_eta_flav": btag_sf_hist("BJetGood_robustParticleTransformer_M"),
        "bjets_robustParticleTransformer_T_pt_eta_flav": btag_sf_hist("BJetGood_robustParticleTransformer_T"),
        "bjets_robustParticleTransformer_XT_pt_eta_flav": btag_sf_hist("BJetGood_robustParticleTransformer_XT"),
        "bjets_robustParticleTransformer_XXT_pt_eta_flav": btag_sf_hist("BJetGood_robustParticleTransformer_XXT"),

        "jets_pt_eta_flav": btag_sf_hist("JetGood")
    },
)
