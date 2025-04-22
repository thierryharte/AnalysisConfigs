import awkward as ak


def add_fields(collection, fields=None, four_vec="PtEtaPhiMLorentzVector"):
    if fields=="all":
        fields= list(collection.fields)
        for field in ["pt", "eta", "phi", "mass"]:
            if field not in fields:
                fields.append(field)
    elif fields is None:
        fields = ["pt", "eta", "phi", "mass"]
        fields_add = [ "PNetRegPtRawRes", "PNetRegPtRawCorr", "PNetRegPtRawCorrNeutrino", "btagPNetB", "index"]
        for field in fields_add:
            if field in list(collection.fields):
                fields.append(field)
                
    if four_vec=="PtEtaPhiMLorentzVector":
        # fields=["pt", "eta", "phi", "mass"]
        fields_dict = {field: getattr(collection, field) for field in fields}
        # remove fields with 2d
        fields_dict = {k: v for k, v in fields_dict.items() if v.ndim == 1}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    elif four_vec=="Momentum4D":
        # fields=["pt", "eta", "phi", "mass"]
        fields_dict = {field: getattr(collection, field) for field in fields}
        # remove fields with 2d
        fields_dict = {k: v for k, v in fields_dict.items() if v.ndim == 1}
        collection = ak.zip(
            fields_dict,
            with_name="Momentum4D",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection
