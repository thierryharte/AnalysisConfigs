import numpy as np
import awkward as ak
from pocket_coffea.lib.jets import jet_selection
import copy

def lepton_selection(events, lepton_flavour, params):
    leptons = events[lepton_flavour]
    cuts = params.object_preselection[lepton_flavour]
    # Requirements on pT and eta
    passes_eta = abs(leptons.eta) < cuts["eta"]
    passes_pt = leptons.pt > cuts["pt"]

    passes_dxy = ak.where(
        abs(leptons.eta) < 1.479,
        leptons.dxy < cuts["dxy_barrel"],
        leptons.dxy < cuts["dxy_endcap"],
    )
    passes_dz = ak.where(
        abs(leptons.eta) < 1.479,
        leptons.dz < cuts["dz_barrel"],
        leptons.dz < cuts["dz_endcap"],
    )

    if lepton_flavour == "Electron":
        passes_iso = leptons.pfRelIso03_all < cuts["iso"]
        passes_id = leptons[cuts["id"]] == True
        good_leptons = (
            passes_eta & passes_pt & passes_iso & passes_dxy & passes_dz & passes_id
        )

    elif lepton_flavour == "Muon":
        # Requirements on isolation and id
        passes_iso = leptons.pfRelIso03_all < cuts["iso"]
        passes_id = leptons[cuts["id"]] == True

        good_leptons = (
            passes_eta & passes_pt & passes_iso & passes_dxy & passes_dz & passes_id
        )

    return leptons[good_leptons]


def jet_selection_nopu(
    events, jet_type, params, tight_cuts=False, semi_tight_vbf=False
):
    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]
    # Only jets that are more distant than dr to ALL leptons are tagged as good jets
    # Mask for  jets not passing the preselection

    pt_cut = "pt"
    if tight_cuts and "pt_tight" in cuts.keys():
        pt_cut = "pt_tight"
    if semi_tight_vbf and "ptSemiTight" in cuts.keys():
        pt_cut = "ptSemiTight"

    if "eta_min" in cuts.keys() and "eta_max" in cuts.keys():
        mask_jets = (
            (jets.pt > cuts[pt_cut])
            & (np.abs(jets.eta) > cuts["eta_min"])
            & (np.abs(jets.eta) < cuts["eta_max"])
            & (jets.jetId >= cuts["jetId"])
            & (jets.btagPNetB >= cuts["btagPNetB"])
        )
    else:
        mask_jets = (
            (jets.pt > cuts[pt_cut])
            & (np.abs(jets.eta) < cuts["eta"])
            & (jets.jetId >= cuts["jetId"])
            & (jets.btagPNetB > cuts["btagPNetB"])
        )

    return jets[mask_jets]


def jet_selection_custom(
    events,
    jet_type,
    params,
    year,
    leptons_collection="",
    jet_tagger="",
    pt_type="pt",
    pt_cut_name="pt",
):
    jet_type_default = "Jet"
    params_copy = copy.copy(params)
    params_copy.object_preselection[jet_type_default]["pt"] = params.object_preselection[
        jet_type
    ][pt_cut_name]

    events_copy=copy.copy(events)
    # replace the pt with the pt_type requested to do the cut on
    events_copy[jet_type_default] = ak.with_field(
        events_copy[jet_type],
        events_copy[jet_type][pt_type],
        "pt",
    )

    _, mask = jet_selection(
        events_copy,
        jet_type_default,
        params_copy,
        year,
        leptons_collection,
        jet_tagger,
    )

    # remove copies
    del params_copy
    del events_copy


    return events[jet_type][mask]
