from collections.abc import Iterable
import awkward as ak


def hh4b(events, params, **kwargs):
    at_least_four_jets = events.nJetGood >= params["njet"]
    no_electron = events.nElectronGood == 0
    no_muon = events.nMuonGood == 0

    mask_4jet_nolep = at_least_four_jets & no_electron & no_muon

    # convert false to None
    mask_4jet_nolep_none = ak.mask(mask_4jet_nolep, mask_4jet_nolep)
    jets = events[mask_4jet_nolep_none].Jet

    mask_pt_none = (
        (jets.pt[:, 0] > params["pt_jet0"])
        & (jets.pt[:, 1] > params["pt_jet1"])
        & (jets.pt[:, 2] > params["pt_jet2"])
        & (jets.pt[:, 3] > params["pt_jet3"])
    )
    # convert none to false
    mask_pt = ak.where(ak.is_none(mask_pt_none), False, mask_pt_none)

    jets_btag_order = events[mask_4jet_nolep_none].JetGoodBtagOrdered.btagDeepFlavB # TODO: use particlenet!
    mask_btag = (
        ((jets_btag_order[:, 0] + jets_btag_order[:, 1]) / 2 > params["mean_pnet_jet"])
        & (jets_btag_order[:, 2] > params["third_pnet_jet"])
        & (jets_btag_order[:, 3] > params["fourth_pnet_jet"])
    )
    mask_btag = ak.where(ak.is_none(mask_btag), False, mask_btag)

    mask = mask_pt & mask_btag

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)