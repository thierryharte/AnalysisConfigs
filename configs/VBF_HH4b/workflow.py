import awkward as ak
from dask.distributed import get_worker
import sys
import numpy as np

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.deltaR_matching import object_matching

from custom_cut_functions import *
from custom_cuts import *

sys.path.append("../../")
from utils.parton_matching_function import get_parton_last_copy
from utils.spanet_evaluation_functions import get_pairing_information, get_best_pairings
from utils.basic_functions import add_fields
from utils.reconstruct_higgs_candidates import (
    reconstruct_higgs_from_provenance,
    reconstruct_higgs_from_idx,
)
from utils.inference_session_onnx import get_model_session
from vbf_matching import get_jets_no_higgs


class VBFHH4bbQuarkMatchingProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.max_num_jets = self.workflow_options["max_num_jets"]
        self.which_bquark = self.workflow_options["which_bquark"]
        self.SPANET_MODEL = self.workflow_options["SPANET_MODEL"]
        self.vbf_parton_matching = self.workflow_options["vbf_parton_matching"]
        self.VBF_GGF_DNN_MODEL = self.workflow_options["VBF_GGF_DNN_MODEL"]
        self.BKG_MORPHING_DNN_MODEL = self.workflow_options["BKG_MORPHING_DNN_MODEL"]
        self.fifth_jet = self.workflow_options["fifth_jet"]

    def apply_object_preselection(self, variation):
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            ak.where(
                self.events.Jet.PNetRegPtRawCorr > 0,
                self.events.Jet.pt
                * (1 - self.events.Jet.rawFactor)
                * self.events.Jet.PNetRegPtRawCorr,
                self.events.Jet.pt,
            ),
            "pt",
        )
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            ak.where(
                self.events.Jet.PNetRegPtRawCorr > 0,
                self.events.Jet.mass
                * (1 - self.events.Jet.rawFactor)
                * self.events.Jet.PNetRegPtRawCorr,
                self.events.Jet.mass,
            ),
            "mass",
        )
        self.events["Jet"] = ak.with_field(
            self.events.Jet, ak.local_index(self.events.Jet, axis=1), "index"
        )

        self.events["JetGood"] = self.events.Jet
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood,
            ak.where(
                self.events.JetGood.PNetRegPtRawCorr > 0,
                self.events.JetGood.pt * self.events.JetGood.PNetRegPtRawCorrNeutrino,
                self.events.JetGood.pt,
            ),
            "pt",
        )
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood,
            ak.where(
                self.events.JetGood.PNetRegPtRawCorr > 0,
                self.events.JetGood.mass * self.events.JetGood.PNetRegPtRawCorrNeutrino,
                self.events.JetGood.mass,
            ),
            "mass",
        )

        self.events["JetGood"] = jet_selection_nopu(self.events, "JetGood", self.params)

        self.events["JetVBF_matching"] = self.events.Jet
        self.events["JetVBF_matching"] = jet_selection_nopu(
            self.events, "JetVBF_matching", self.params
        )

        self.events["JetGoodVBF"] = self.events.Jet
        self.events["JetGoodVBF"] = jet_selection_nopu(
            self.events, "JetGoodVBF", self.params
        )

        self.events["JetVBF_generalSelection"] = self.events.Jet
        self.events["JetVBF_generalSelection"] = jet_selection_nopu(
            self.events, "JetVBF_generalSelection", self.params
        )

        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)
        # order jet by btag score and keep only the first 4
        self.events["JetGood"] = self.events.JetGood[
            ak.argsort(self.events.JetGood.btagPNetB, axis=1, ascending=False)
        ]
        # keep only the first 4 jets for the Higgs candidates reconstruction
        self.events["JetGoodHiggs"] = self.events.JetGood[:, :4]

        self.events["JetGoodHiggsPtOrder"] = self.events.JetGoodHiggs[
            ak.argsort(self.events.JetGoodHiggs.pt, axis=1, ascending=False)
        ]

        # Trying to reshuffle jets 4 and above by pt instead of b-tag score
        if self.fifth_jet == "pt":
            jets5plus = self.events["JetGood"][:, 4:]
            jets5plus_pt = jets5plus[ak.argsort(jets5plus.pt, axis=1, ascending=False)]
            self.events["JetGood"] = ak.concatenate(
                (self.events["JetGoodHiggs"], jets5plus_pt), axis=1
            )
            del jets5plus
            del jets5plus_pt

    def get_jet_higgs_provenance(self, which_bquark):  # -> ak.Array:
        # Select b-quarks at Gen level, coming from H->bb decay
        self.events["GenPart"] = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        genpart = self.events.GenPart

        isHiggs = genpart.pdgId == 25
        isB = abs(genpart.pdgId) == 5
        isLast = genpart.hasFlags(["isLastCopy"])
        isFirst = genpart.hasFlags(["isFirstCopy"])
        isHard = genpart.hasFlags(["fromHardProcess"])

        higgs = genpart[isHiggs & isLast & isHard]
        higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]
        higgs = higgs[ak.argsort(higgs.pt, ascending=False)]
        higgs = higgs[ak.argsort(higgs.pt, ascending=False)]

        if which_bquark == "last_numba":
            bquarks_first = genpart[isB & isHard & isFirst]
            mother_bquarks = genpart[bquarks_first.genPartIdxMother]
            bquarks_from_higgs = bquarks_first[mother_bquarks.pdgId == 25]
            provenance = ak.where(
                bquarks_from_higgs.genPartIdxMother == higgs.index[:, 0], 1, 2
            )

            # define variables to get the last copy
            children_idxG = ak.without_parameters(genpart.childrenIdxG, behavior={})
            children_idxG_flat = ak.flatten(children_idxG, axis=1)
            genpart_pdgId_flat = ak.flatten(
                ak.without_parameters(genpart.pdgId, behavior={}), axis=1
            )
            genpart_LastCopy_flat = ak.flatten(
                ak.without_parameters(genpart.hasFlags(["isLastCopy"]), behavior={}),
                axis=1,
            )
            genpart_pt_flat = ak.flatten(
                ak.without_parameters(genpart.pt, behavior={}), axis=1
            )
            genparts_flat = ak.flatten(genpart)
            genpart_offsets = np.concatenate(
                [
                    [0],
                    np.cumsum(ak.to_numpy(ak.num(genpart, axis=1), allow_missing=True)),
                ]
            )
            b_quark_idx = ak.to_numpy(
                bquarks_from_higgs.index + genpart_offsets[:-1], allow_missing=False
            )
            b_quarks_pdgId = ak.to_numpy(bquarks_from_higgs.pdgId, allow_missing=False)
            nevents = b_quark_idx.shape[0]
            firstgenpart_idxG = ak.firsts(genpart[:, 0].children).genPartIdxMotherG
            firstgenpart_idxG_numpy = ak.to_numpy(
                firstgenpart_idxG, allow_missing=False
            )

            b_quark_last_idx = get_parton_last_copy(
                b_quark_idx,
                b_quarks_pdgId,
                children_idxG_flat,
                genpart_pdgId_flat,
                genpart_offsets,
                genpart_LastCopy_flat,
                genpart_pt_flat,
                nevents,
                firstgenpart_idxG_numpy,
            )
            bquarks = genparts_flat[b_quark_last_idx]

        elif which_bquark == "last":
            bquarks = genpart[isB & isLast & isHard]
            bquarks_first = bquarks
            while True:
                b_mother = genpart[bquarks_first.genPartIdxMother]
                mask_mother = (abs(b_mother.pdgId) == 5) | ((b_mother.pdgId) == 25)
                bquarks = bquarks[mask_mother]
                bquarks_first = bquarks_first[mask_mother]
                b_mother = b_mother[mask_mother]
                if ak.all((b_mother.pdgId) == 25):
                    break
                bquarks_first = ak.where(
                    abs(b_mother.pdgId) == 5, b_mother, bquarks_first
                )
            provenance = ak.where(
                bquarks_first.genPartIdxMother == higgs.index[:, 0], 1, 2
            )
        elif which_bquark == "first":
            bquarks = ak.flatten(higgs.children, axis=2)
            provenance = ak.where(bquarks.genPartIdxMother == higgs.index[:, 0], 1, 2)
        else:
            raise ValueError(
                "which_bquark for the parton matching must be 'first' or 'last' or 'last_numba'"
            )

        # Adding the provenance to the quark object
        bquarks = ak.with_field(bquarks, provenance, "provenance")
        self.events["bQuark"] = bquarks

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_bquarks_higgs, matched_jets_higgs, deltaR_matched_higgs = (
            object_matching(
                bquarks,
                self.events.JetGoodHiggs,
                dr_min=self.dr_min,
            )
        )
        # matched all jetgood
        matched_bquarks, matched_jets, deltaR_matched = object_matching(
            bquarks,
            self.events.JetGood,
            dr_min=self.dr_min,
        )

        matched_jets_higgs = ak.with_field(
            matched_jets_higgs, matched_bquarks_higgs.provenance, "provenance"
        )
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs, matched_bquarks_higgs.provenance, "provenance"
        )
        matched_jets = ak.with_field(
            matched_jets, matched_bquarks.provenance, "provenance"
        )
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood, matched_bquarks.provenance, "provenance"
        )

        self.events["bQuarkHiggsMatched"] = ak.with_field(
            matched_bquarks_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            matched_jets_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["bQuarkMatched"] = ak.with_field(
            matched_bquarks, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            self.events.JetGoodHiggsMatched,
            self.events.bQuarkHiggsMatched.pdgId,
            "pdgId",
        )
        self.events["JetGoodMatched"] = ak.with_field(
            self.events.JetGoodMatched,
            self.events.bQuarkMatched.pdgId,
            "pdgId",
        )

    def do_vbf_parton_matching(self, which_bquark):  # -> ak.Array:
        # Select vbf quarks
        self.events.GenPart = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        genpart = self.events.GenPart

        isQuark = abs(genpart.pdgId) < 7
        isHard = genpart.hasFlags(["fromHardProcess"])

        quarks = genpart[isQuark & isHard]
        quarks = quarks[quarks.genPartIdxMother != -1]

        quarks_mother = genpart[quarks.genPartIdxMother]
        quarks_mother_children = quarks_mother.children
        quarks_mother_children_isH = (
            ak.sum((quarks_mother_children.pdgId == 25), axis=-1) == 2
        )
        vbf_quarks = quarks[quarks_mother_children_isH]

        children_idxG = ak.without_parameters(genpart.childrenIdxG, behavior={})
        children_idxG_flat = ak.flatten(children_idxG, axis=1)
        genpart_pdgId_flat = ak.flatten(
            ak.without_parameters(genpart.pdgId, behavior={}), axis=1
        )
        genpart_LastCopy_flat = ak.flatten(
            ak.without_parameters(genpart.hasFlags(["isLastCopy"]), behavior={}), axis=1
        )
        genpart_pt_flat = ak.flatten(
            ak.without_parameters(genpart.pt, behavior={}), axis=1
        )
        genparts_flat = ak.flatten(genpart)
        genpart_offsets = np.concatenate(
            [[0], np.cumsum(ak.to_numpy(ak.num(genpart, axis=1), allow_missing=True))]
        )
        vbf_quark_idx = ak.to_numpy(
            vbf_quarks.index + genpart_offsets[:-1], allow_missing=False
        )
        vbf_quarks_pdgId = ak.to_numpy(vbf_quarks.pdgId, allow_missing=False)
        nevents = vbf_quark_idx.shape[0]
        firstgenpart_idxG = ak.firsts(genpart[:, 0].children).genPartIdxMotherG
        firstgenpart_idxG_numpy = ak.to_numpy(firstgenpart_idxG, allow_missing=False)

        vbf_quark_last_idx = get_parton_last_copy(
            vbf_quark_idx,
            vbf_quarks_pdgId,
            children_idxG_flat,
            genpart_pdgId_flat,
            genpart_offsets,
            genpart_LastCopy_flat,
            genpart_pt_flat,
            nevents,
            firstgenpart_idxG_numpy,
        )

        vbf_quark_last = genparts_flat[vbf_quark_last_idx]

        matched_vbf_quarks, matched_vbf_jets, deltaR_matched_vbf = object_matching(
            vbf_quark_last,
            self.events.JetVBF_matching,
            dr_min=self.dr_min,
        )

        maskNotNone = ~ak.is_none(matched_vbf_jets, axis=1)
        self.events["JetVBF_matched"] = matched_vbf_jets[maskNotNone]

        self.events["JetVBF_matched"] = ak.with_field(
            self.events.JetVBF_matched,
            ak.where(
                self.events.JetVBF_matched.PNetRegPtRawCorr > 0,
                self.events.JetVBF_matched.pt
                / self.events.JetVBF_matched.PNetRegPtRawCorrNeutrino,
                self.events.JetVBF_matched.pt,
            ),
            "pt",
        )

        self.events["quarkVBF_matched"] = matched_vbf_quarks[maskNotNone]

        self.events["quarkVBF"] = vbf_quark_last

        # general Selection
        (
            matched_vbf_quarks_generalSelection,
            matched_vbf_jets_generalSelection,
            deltaR_matched_vbf,
        ) = object_matching(
            vbf_quark_last,
            self.events.JetVBF_generalSelection,
            dr_min=self.dr_min,
        )
        maskNotNone_genSel = ~ak.is_none(matched_vbf_jets_generalSelection, axis=1)

        self.events["JetVBF_generalSelection_matched"] = (
            matched_vbf_jets_generalSelection[maskNotNone_genSel]
        )

        self.events["quarkVBF_generalSelection_matched"] = (
            matched_vbf_quarks_generalSelection[maskNotNone_genSel]
        )

    def dummy_provenance(self):
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs,
            ak.ones_like(self.events.JetGoodHiggs.pt) * -1,
            "provenance",
        )
        self.events["JetGoodHiggsMatched"] = self.events.JetGoodHiggs

        self.events["JetGood"] = ak.with_field(
            self.events.JetGood, ak.ones_like(self.events.JetGood.pt) * -1, "provenance"
        )
        self.events["JetGoodMatched"] = self.events.JetGood

    def count_objects(self, variation):
        # NOTE:  ak.num counts even the None values, while ak.count counts only the non-None values
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodHiggs"] = ak.num(self.events.JetGoodHiggs, axis=1)
        self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)
        self.events["nJetVBF_generalSelection"] = ak.num(
            self.events.JetVBF_generalSelection, axis=1
        )

    def HelicityCosTheta(self, higgs, jet):
        print("jet px",jet.px)
        print("higgs px",higgs.px)
        print("jet theta",jet.theta)
        higgs_velocity = higgs.to_beta3()
        jet = jet.boost_beta3(-higgs_velocity)
        print("jet px",jet.px)
        print("jet theta",jet.theta)
        return np.cos(jet.theta)

    def Costhetastar_CS(self, higgs1_vec, hh_vec):
        print("px",higgs1_vec.px)
        print("theta",higgs1_vec.theta)
        hh_velocity = hh_vec.to_beta3()
        higgs1_vec = higgs1_vec.boost_beta3(-hh_velocity)
        print("px",higgs1_vec.px)
        print("theta",higgs1_vec.theta)
        return abs(np.cos(higgs1_vec.theta))

    def define_bkg_morphing_variables(self):
        ########################
        # ADDITIONAL VARIABLES #
        ########################

        self.events["era"] = ak.ones_like(self.events.Jet.pt)

        # HT : scalar sum of all jets with pT > 25 GeV inside | η | < 2.5
        self.events["HT"] = ak.sum(self.events.JetGood.pt, axis=1)

        # Minimum ∆R ( jj ) among all possible pairings of the leading b-tagged jets
        # Maximum ∆R( jj ) among all possible pairings of the leading b-tagged jets
        _, JetGood2 = ak.unzip(
            ak.cartesian(
                [
                    self.events.JetGood[:, :4],
                    self.events.JetGood[:, :4],
                ],
                nested=True,
            )
        )
        dR = self.events.JetGood[:, :4].delta_r(JetGood2)
        # remove dR between the same jets
        dR = ak.mask(dR, dR > 0)
        # flatten the last 2 dimension of the dR array  to get an array for each event
        dR = ak.flatten(dR, axis=2)
        self.events["dR_min"] = ak.min(dR, axis=1)
        self.events["dR_max"] = ak.max(dR, axis=1)

        # Leading-pT H candidate pT , η, φ, and mass
        # Subleading-pT H candidate pT , η, φ, and mass
        # Angular separation (∆R) between b jets for each H candidate
        self.events["HiggsLeading"] = ak.with_field(
            self.events.HiggsLeading,
            self.events["JetGoodFromHiggsOrdered"][:, 0].delta_r(
                self.events["JetGoodFromHiggsOrdered"][:, 1]
            ),
            "dR",
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events.HiggsSubLeading,
            self.events["JetGoodFromHiggsOrdered"][:, 2].delta_r(
                self.events["JetGoodFromHiggsOrdered"][:, 3]
            ),
            "dR",
        )

        # helicity | cos θ | for each H candidate
        self.events["HiggsLeading"] = ak.with_field(
            self.events.HiggsLeading,
            abs(
                self.HelicityCosTheta(
                    self.events.HiggsLeading, self.events.JetGoodFromHiggsOrdered[:, 0]
                )
            ),
            "helicityCosTheta",
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events.HiggsSubLeading,
            abs(
                self.HelicityCosTheta(
                    self.events.HiggsSubLeading,
                    self.events.JetGoodFromHiggsOrdered[:, 2],
                )
            ),
            "helicityCosTheta",
        )

        # di-Higgs system
        # pT , η, and mass of HH system
        self.events["HH"] = add_fields(
            self.events.HiggsLeading + self.events.HiggsSubLeading
        )

        # | cos θ ∗ | of HH system
        self.events["HH"] = ak.with_field(
            self.events.HH, self.Costhetastar_CS(self.events.HiggsLeading, self.events.HH),
            "cos_theta_star"
        )

        # Angular separation (∆R, ∆η, ∆φ) between H candidates
        self.events["HH"] = ak.with_field(
            self.events.HH,
            self.events.HiggsLeading.delta_r(self.events.HiggsSubLeading),
            "dR",
        )
        self.events["HH"] = ak.with_field(
            self.events.HH,
            abs(self.events.HiggsLeading.eta - self.events.HiggsSubLeading.eta),
            "dEta",
        )
        self.events["HH"] = ak.with_field(
            self.events.HH,
            self.events.HiggsLeading.delta_phi(self.events.HiggsSubLeading),
            "dPhi",
        )

        # TODO: Implement the sigma mbb calculation
        
        # jet1_up = jet1*(1+jet_pnet_ptres_b.at(j1_index));
        # jet2_up = jet2*(1+jet_pnet_ptres_b.at(j2_index));

        # jet1_dn = jet1*(1-jet_pnet_ptres_b.at(j1_index));
        # jet2_dn = jet2*(1-jet_pnet_ptres_b.at(j2_index));

        # Float_t sigma_hbbCand_A = std::max( fabs((jet1+jet2).M() - (jet1_up + jet2).M()),fabs((jet1+jet2).M() - (jet1_dn + jet2).M()));
        # Float_t sigma_hbbCand_B = std::max( fabs((jet1+jet2).M() - (jet1 + jet2_up).M()),fabs((jet1+jet2).M() - (jet1 + jet2_dn).M()));

        # Float_t sigma_mbb = (sqrt(sigma_hbbCand_A*sigma_hbbCand_A + sigma_hbbCand_B*sigma_hbbCand_B));

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self._isMC:
            if not self.SPANET_MODEL:
                # do truth matching to get b-jet from Higgs
                self.get_jet_higgs_provenance(which_bquark=self.which_bquark)
                self.events["nbQuarkHiggsMatched"] = ak.num(
                    self.events.bQuarkHiggsMatched, axis=1
                )
                self.events["nbQuarkMatched"] = ak.num(
                    self.events.bQuarkMatched, axis=1
                )

                # reconstruct the higgs candidates
                (
                    self.events["HiggsLeading"],
                    self.events["HiggsSubLeading"],
                    self.events["JetGoodFromHiggsOrdered"],
                ) = reconstruct_higgs_from_provenance(self.events.JetGoodMatched)

                matched_jet_higgs_idx_not_none = self.events.JetGoodMatched.index[
                    ~ak.is_none(self.events.JetGoodMatched.index, axis=1)
                ]
            else:
                # apply spanet model to get the pairing prediction for the b-jets from Higgs
                self.dummy_provenance()

                model_session_SPANET, input_name_SPANET, output_name_SPANET = (
                    get_model_session(self.SPANET_MODEL, "SPANET")
                )

                # try:
                #     worker = get_worker()
                # except ValueError:
                #     worker = None

                # if worker is None:
                #     import onnxruntime as ort

                #     sess_options = ort.SessionOptions()
                #     sess_options.graph_optimization_level = (
                #         ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                #     )
                #     model_session_SPANET = ort.InferenceSession(
                #         self.SPANET_MODEL,
                #         sess_options=sess_options,
                #         providers=["CPUExecutionProvider"],
                #     )

                # else:
                #     model_session_SPANET = worker.data["model_session_SPANET"]

                # input_name_SPANET = [input.name for input in model_session_SPANET.get_inputs()]
                # output_name_SPANET = [output.name for output in model_session_SPANET.get_outputs()]

                # compute the pairing information using the SPANET model
                pairing_outputs = get_pairing_information(
                    model_session_SPANET,
                    input_name_SPANET,
                    output_name_SPANET,
                    self.events,
                    self.max_num_jets,
                )

                (
                    pairing_predictions,
                    self.events["best_pairing_probability"],
                    self.events["second_best_pairing_probability"],
                ) = get_best_pairings(pairing_outputs)

                (
                    self.events["HiggsLeading"],
                    self.events["HiggsSubLeading"],
                    self.events["JetGoodFromHiggsOrdered"],
                ) = reconstruct_higgs_from_idx(self.events.JetGood, pairing_predictions)

                matched_jet_higgs_idx_not_none = (
                    self.events.JetGoodFromHiggsOrdered.index
                )

            if self.VBF_GGF_DNN_MODEL:
                (
                    model_session_VBF_GGF_DNN,
                    input_name_VBF_GGF_DNN,
                    output_name_VBF_GGF_DNN,
                ) = get_model_session(self.VBF_GGF_DNN_MODEL, "VBF_GGF_DNN")
                # try:
                #     worker = get_worker()
                # except ValueError:
                #     worker = None

                # if worker is None:
                #     import onnxruntime as ort

                #     sess_options = ort.SessionOptions()
                #     sess_options.graph_optimization_level = (
                #         ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                #     )
                #     model_session_VBF_GGF_DNN = ort.InferenceSession(
                #         self.VBF_GGF_DNN_MODEL,
                #         sess_options=sess_options,
                #         providers=["CPUExecutionProvider"],
                #     )
                # else:
                #     model_session_VBF_GGF_DNN = worker.data["model_session_VBF_GGF_DNN"]

                # input_name_VBF_GGF_DNN = [input.name for input in model_session_VBF_GGF_DNN.get_inputs()]
                # output_name_VBF_GGF_DNN = [output.name for output in model_session_VBF_GGF_DNN.get_outputs()]

            if self.BKG_MORPHING_DNN_MODEL:
                (
                    model_session_BKG_MORPHING_DNN,
                    input_name_BKG_MORPHING_DNN,
                    output_name_BKG_MORPHING_DNN,
                ) = get_model_session(self.BKG_MORPHING_DNN_MODEL, "BKG_MORPHING_DNN")

                self.define_bkg_morphing_variables()
                # try:
                #     worker = get_worker()
                # except ValueError:
                #     worker = None

                # if worker is None:
                #     import onnxruntime as ort

                #     sess_options = ort.SessionOptions()
                #     sess_options.graph_optimization_level = (
                #         ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                #     )

                #     model_session_BKG_MORPHING_DNN = ort.InferenceSession(
                #         self.BKG_MORPHING_DNN_MODEL,
                #         sess_options=sess_options,
                #         providers=["CPUExecutionProvider"],
                #     )

                # else:
                #     model_session_BKG_MORPHING_DNN = worker.data["model_session_BKG_MORPHING_DNN"]

                # input_name_BKG_MORPHING_DNN = [input.name for input in model_session_BKG_MORPHING_DNN.get_inputs()]
                # output_name_BKG_MORPHING_DNN = [output.name for output in model_session_BKG_MORPHING_DNN.get_outputs()]

            jet_offsets = np.concatenate(
                [
                    [0],
                    np.cumsum(
                        ak.to_numpy(ak.num(self.events.Jet, axis=1), allow_missing=True)
                    ),
                ]
            )
            local_index_all = ak.local_index(self.events.Jet, axis=1)
            jets_index_all = ak.to_numpy(
                ak.flatten(local_index_all + jet_offsets[:-1]), allow_missing=True
            )
            jets_from_higgs_idx = ak.to_numpy(
                ak.flatten(matched_jet_higgs_idx_not_none + jet_offsets[:-1]),
                allow_missing=False,
            )
            jets_no_higgs_idx = get_jets_no_higgs(jets_index_all, jets_from_higgs_idx)
            jets_no_higgs_idx_unflat = (
                ak.unflatten(jets_no_higgs_idx, ak.num(self.events.Jet, axis=1))
                - jet_offsets[:-1]
            )
            self.events["JetVBFNotFromHiggs"] = self.events.Jet[
                jets_no_higgs_idx_unflat >= 0
            ]
            # apply selection to the jets not from Higgs
            self.events["JetVBFNotFromHiggs"] = jet_selection_nopu(
                self.events, "JetVBFNotFromHiggs", self.params
            )

            # order in pt
            self.events["JetVBFNotFromHiggs"] = self.events.JetVBFNotFromHiggs[
                ak.argsort(self.events.JetVBFNotFromHiggs.pt, axis=1, ascending=False)
            ]

            self.events["HH"] = add_fields(
                self.events.HiggsLeading + self.events.HiggsSubLeading
            )

            if self.vbf_parton_matching:
                self.do_vbf_parton_matching(which_bquark=self.which_bquark)

                self.events["nJetVBF_matched"] = ak.num(
                    self.events.JetVBF_matched, axis=1
                )

                # Create new variable delta eta and invariant mass of the jets
                JetVBF_matched_padded = ak.pad_none(
                    self.events.JetVBF_matched, 2
                )  # Adds none jets to events that have less than 2 jets

                self.events["deltaEta_matched"] = abs(
                    JetVBF_matched_padded.eta[:, 0] - JetVBF_matched_padded.eta[:, 1]
                )

                self.events["jj_mass_matched"] = (
                    JetVBF_matched_padded[:, 0] + JetVBF_matched_padded[:, 1]
                ).mass

                # This product will give only -1 or 1 values, as it's needed to see if the two jets are in the same side or not
                self.events["etaProduct"] = (
                    JetVBF_matched_padded.eta[:, 0] * JetVBF_matched_padded.eta[:, 1]
                ) / abs(
                    JetVBF_matched_padded.eta[:, 0] * JetVBF_matched_padded.eta[:, 1]
                )

            # choose vbf jets as the two jets with the highest pt that are not from higgs decay
            self.events["JetVBFLeadingPtNotFromHiggs"] = self.events.JetVBFNotFromHiggs[
                :, :2
            ]

            # choose higgs jets as the two jets with the highest mjj that are not from higgs decay
            jet_combinations = ak.combinations(self.events.JetVBFNotFromHiggs, 2)
            jet_combinations_mass = (jet_combinations["0"] + jet_combinations["1"]).mass
            jet_combinations_mass_max_idx = ak.to_numpy(
                ak.argsort(jet_combinations_mass, axis=1, ascending=False)[:, 0]
            )

            jets_max_mass = jet_combinations[
                ak.local_index(jet_combinations, axis=0), jet_combinations_mass_max_idx
            ]
            vbf_jets_max_mass_0 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.to_numpy(jets_max_mass["0"].index),
                ],
                1,
            )
            vbf_jets_max_mass_1 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.to_numpy(jets_max_mass["1"].index),
                ],
                1,
            )

            vbf_jet_leading_mjj = ak.with_name(
                ak.concatenate([vbf_jets_max_mass_0, vbf_jets_max_mass_1], axis=1),
                name="PtEtaPhiMCandidate",
            )

            vbf_jet_leading_mjj_fields_dict = {
                field: getattr(vbf_jet_leading_mjj, field)
                for field in vbf_jet_leading_mjj.fields
                if ("muon" not in field and "electron" not in field)
            }
            self.events["JetVBFLeadingMjjNotFromHiggs"] = add_fields(
                vbf_jet_leading_mjj, vbf_jet_leading_mjj_fields_dict
            )

            self.events["JetVBFLeadingPtNotFromHiggs_deltaEta"] = abs(
                self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 0]
                - self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 1]
            )

            self.events["JetVBFLeadingMjjNotFromHiggs_deltaEta"] = abs(
                self.events.JetVBFLeadingMjjNotFromHiggs.eta[:, 0]
                - self.events.JetVBFLeadingMjjNotFromHiggs.eta[:, 1]
            )

            self.events["JetVBFLeadingPtNotFromHiggs_jjMass"] = (
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
                + self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            ).mass

            self.events["JetVBFLeadingMjjNotFromHiggs_jjMass"] = (
                self.events.JetVBFLeadingMjjNotFromHiggs[:, 0]
                + self.events.JetVBFLeadingMjjNotFromHiggs[:, 1]
            ).mass

            self.events["HH_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.HiggsSubLeading
            )

            self.events["jj_deltaR"] = self.events.JetVBFLeadingPtNotFromHiggs[
                :, 0
            ].delta_r(self.events.JetVBFLeadingPtNotFromHiggs[:, 1])

            self.events["H1j1_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
            )

            self.events["H1j2_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            )

            self.events["H2j1_deltaR"] = self.events.HiggsSubLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
            )

            self.events["H2j2_deltaR"] = self.events.HiggsSubLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            )

            JetVBFLeadingPtNotFromHiggs_etaAverage = (
                self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 0]
                + self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 1]
            ) / 2

            self.events["HH_centrality"] = np.exp(
                (
                    -(
                        (
                            self.events.HiggsLeading.eta
                            - JetVBFLeadingPtNotFromHiggs_etaAverage
                        )
                        ** 2
                    )
                    - (
                        self.events.HiggsSubLeading.eta
                        - JetVBFLeadingPtNotFromHiggs_etaAverage
                    )
                    ** 2
                )
                / (self.events.JetVBFLeadingPtNotFromHiggs_deltaEta) ** 2
            )

        else:
            self.dummy_provenance()

        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)
