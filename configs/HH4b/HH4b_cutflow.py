import os

from pocket_coffea.parameters import defaults
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
    goldenJson,
    eventFlags,
    get_nPVgood,
    get_JetVetoMap,
)
from pocket_coffea.parameters.cuts import passthrough

from workflow import HH4bbQuarkMatchingProcessor
import configs.HH4b.custom_cuts as cuts
from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
    onnx_model_dict,
)
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2022_postEE", "2022_preEE", "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withoutVariations_withJERC.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_SM_local_redirector.json",
        ],
        "filter": {
            "samples": (["GluGlutoHHto4B"]),
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessor,
    workflow_options=config_options_dict,
    skim=[],
    preselections=[cuts.four_jets_cut, goldenJson, get_nPVgood(1)],
    categories={
        "four_jets_presel": [passthrough],
        "MET_filter": [eventFlags],
        "Lepton_veto": [eventFlags, cuts.lepton_veto],
        "Jet_Veto_map": [eventFlags, cuts.lepton_veto, get_JetVetoMap()],
        "HLT_selection": [
            eventFlags,
            cuts.lepton_veto,
            get_JetVetoMap(),
            get_HLTsel(primaryDatasets=["JetMET"]),
        ],
        "jet_pT_selection": [
            eventFlags,
            cuts.lepton_veto,
            get_JetVetoMap(),
            get_HLTsel(primaryDatasets=["JetMET"]),
            cuts.jet_pt_cut,
        ],
        # HLT_matching: Missing in NanoAOD,
        "2b_selection": [
            eventFlags,
            cuts.lepton_veto,
            get_JetVetoMap(),
            get_HLTsel(primaryDatasets=["JetMET"]),
            cuts.jet_pt_cut,
            cuts.two_b_cut,
        ],
        "third_btag_selection": [
            eventFlags,
            cuts.lepton_veto,
            get_JetVetoMap(),
            get_HLTsel(primaryDatasets=["JetMET"]),
            cuts.jet_pt_cut,
            cuts.two_b_cut,
            cuts.third_btag_cut,
        ],
        "fourth_btag_selection": [
            eventFlags,
            cuts.lepton_veto,
            get_JetVetoMap(),
            get_HLTsel(primaryDatasets=["JetMET"]),
            cuts.jet_pt_cut,
            cuts.two_b_cut,
            cuts.third_btag_cut,
            cuts.fourth_btag_cut,
        ],
        "signal_region_Run2_selection": [
            eventFlags,
            cuts.lepton_veto,
            get_JetVetoMap(),
            get_HLTsel(primaryDatasets=["JetMET"]),
            cuts.jet_pt_cut,
            cuts.two_b_cut,
            cuts.third_btag_cut,
            cuts.fourth_btag_cut,
            cuts.signal_region_Run2_cut,
        ],
        # "signal_region_selection": [
        #     eventFlags,
        #     cuts.lepton_veto,
        #     get_JetVetoMap(),
        #     get_HLTsel(primaryDatasets=["JetMET"]),
        #     cuts.jet_pt_cut,
        #     cuts.two_b_cut,
        #     cuts.third_btag_cut,
        #     cuts.fourth_btag_cut,
        #     cuts.signal_region_cut,
        # ],
    },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
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
    variables={
        #
    },
    columns={
        #
    },
)
