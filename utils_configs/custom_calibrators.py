from pocket_coffea.lib.calibrators.common.common import JetsSoftdropMassCalibrator
from pocket_coffea.lib.jets import JET_SORTIDX_FIELD


class JetsSoftdropMassFieldCalibrator(JetsSoftdropMassCalibrator):
    """
    Same softdrop mass correction as JetsSoftdropMassCalibrator (AK4 JEC on the subjets),
    but it only replaces the FatJet.msoftdrop field.

    The original calibrator computes the correction in initialize() on the uncalibrated
    NanoAOD events and returns the full FatJet collection, which overwrites the output of
    the JetsCalibrator (JEC pt/mass, pt_raw, mass_raw, pt_gen). Replacing only the field
    keeps those. Has to be placed after the JetsCalibrator in the calibrators sequence.
    """

    name = "msoftdrop_field_calibration"

    def initialize(self, events):
        super().initialize(events)
        self.calibrated_collections = [f"{coll}.msoftdrop" for coll in self.calibrated_collections]

    def calibrate(self, events, orig_colls, variation, already_applied_calibrators=None):
        colls = super().calibrate(events, orig_colls, variation, already_applied_calibrators)
        out = {}
        for coll, jets in colls.items():
            msoftdrop = jets.msoftdrop  # NanoAOD jet order
            # follow the re-sorting of the JetsCalibrator, if it was applied (sort_by_pt)
            if JET_SORTIDX_FIELD in events[coll].fields:
                msoftdrop = msoftdrop[events[coll][JET_SORTIDX_FIELD]]
            out[f"{coll}.msoftdrop"] = msoftdrop
        return out
