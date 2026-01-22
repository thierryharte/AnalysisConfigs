import numpy as np
import h5py
import coffea.util
import awkward as ak

MAX_JETS = 4

# collections that contain underscores and must NOT be split at first underscore
KEEP_TOGETHER_COLLECTIONS = {
    "add_jet1pt",
}

# collections that are jet-like and should be padded/clipped to (N, MAX_JETS, ...)
JET_COLLECTIONS = {
    "JetGoodFromHiggsOrdered",
}


def is_awkward(x):
    return isinstance(x, (ak.Array, ak.Record))

def unflatten_to_jagged(flat, counts):
    """
    flat: 1D array of length sum(counts)
    counts: 1D int array of length Nevents
    returns: awkward jagged array (Nevents, Nj)
    """
    flat = unwrap_accumulator(flat)
    counts = unwrap_accumulator(counts)

    flat = np.asarray(flat)
    counts = np.asarray(counts).astype(np.int64)

    if flat.ndim != 1:
        raise ValueError(f"Expected flat 1D jet array, got shape {flat.shape}")
    if counts.ndim != 1:
        raise ValueError(f"Expected 1D counts array, got shape {counts.shape}")

    if flat.shape[0] != int(counts.sum()):
        raise ValueError(f"Flat length {flat.shape[0]} != sum(counts) {int(counts.sum())}")

    return ak.unflatten(flat, counts)

def mask_from_counts(counts, max_jets=MAX_JETS):
    counts = np.asarray(unwrap_accumulator(counts)).astype(np.int64)
    idx = np.arange(max_jets)[None, :]
    return (idx < counts[:, None])

def unwrap_accumulator(x):
    """
    Unwrap common coffea accumulator wrappers (e.g. column_accumulator).
    We avoid importing private coffea classes by using duck-typing.
    """
    if hasattr(x, "value"):
        try:
            return x.value
        except Exception:
            pass

    if hasattr(x, "_value"):
        try:
            return x._value
        except Exception:
            pass

    for meth in ("to_numpy", "numpy"):
        if hasattr(x, meth):
            try:
                return getattr(x, meth)()
            except Exception:
                pass

    return x


def to_numpy_event_vector(x):
    """
    Convert event-level data to a strict 1D numeric numpy array.
    Handles column_accumulator + awkward + object-scalars.
    """
    x = unwrap_accumulator(x)

    if is_awkward(x):
        x = ak.to_numpy(x)

    arr = np.asarray(x)

    if arr.shape == () and arr.dtype == object:
        arr = np.asarray(unwrap_accumulator(arr.item()))

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D event vector, got shape {arr.shape}, dtype={arr.dtype}")
    if arr.dtype == object:
        raise TypeError("Still object dtype after unwrapping; likely not numeric.")
    return arr


def is_jet_array(x) -> bool:
    """
    True if x looks like per-event jets:
    - awkward jagged with axis=1, OR
    - dense numpy/awkward-numpy with ndim >= 2 (N, Nj, ...)
    """
    x = unwrap_accumulator(x)

    if is_awkward(x):
        # jagged?
        try:
            _ = ak.num(x, axis=1)
            return True
        except Exception:
            # could still be regular (non-jagged) awkward with ndim>=2 after to_numpy
            x_np = ak.to_numpy(x)
            return np.asarray(x_np).ndim >= 2

    return np.asarray(x).ndim >= 2

def pad_clip_jets_with_mask(jets, max_jets=5, fill_value=-9999):
    """
    jets can be:
      - awkward jagged: (N, Nj, ...)
      - dense numpy:    (N, Nj, ...)
      - numpy object:   (N,) each element is list/array of length Nj
    Returns:
      dense: (N, max_jets, ...)
      mask:  (N, max_jets) boolean (True = real jet)
    """
    jets = unwrap_accumulator(jets)

    # Case A: awkward (prefer this)
    if is_awkward(jets):
        # must be something with axis=1
        try:
            padded = ak.pad_none(jets, max_jets, axis=1, clip=True)
        except Exception as e:
            raise ValueError(f"Awkward jets cannot be padded on axis=1. awkward_type={ak.type(jets)}") from e

        mask = ~ak.is_none(padded, axis=1)
        mask = ak.to_numpy(mask)

        filled = ak.fill_none(padded, fill_value)
        dense = ak.to_numpy(filled)
        return dense, mask

    # Convert numpy object ragged -> awkward
    arr = np.asarray(jets)
    if arr.dtype == object and arr.ndim == 1 and arr.size > 0:
        # if entries look like lists/arrays, interpret as ragged jets per event
        first = arr[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            jets_ak = ak.Array(arr)
            padded = ak.pad_none(jets_ak, max_jets, axis=1, clip=True)
            mask = ~ak.is_none(padded, axis=1)
            mask = ak.to_numpy(mask)

            filled = ak.fill_none(padded, fill_value)
            dense = ak.to_numpy(filled)
            return dense, mask

    # Case B: dense numpy (N, Nj, ...)
    if arr.ndim < 2:
        raise ValueError(
            f"Jets are neither awkward jagged nor dense 2D+. Got shape={arr.shape}, dtype={arr.dtype}. "
            f"Example element type={type(arr.item()) if arr.shape == () else type(arr.flat[0])}"
        )

    N, Nj = arr.shape[0], arr.shape[1]
    use = min(Nj, max_jets)

    mask = np.zeros((N, max_jets), dtype=bool)
    mask[:, :use] = True

    out = np.full((N, max_jets) + arr.shape[2:], fill_value, dtype=arr.dtype)
    out[:, :use, ...] = arr[:, :use, ...]
    return out, mask



def infer_collection_and_var(name: str):
    """
    Rules:
    - if name starts with "events_", treat as Event-level collection "Event"
    - else split into (collection, var) by first underscore, EXCEPT keep-together collections
      e.g. "HH_pt" -> ("HH", "pt")
           "add_jet1pt_pt" -> ("add_jet1pt", "pt")
    - if no underscore, put it under Event-level "Event"
    """
    if name.startswith("events_"):
        return "Event", name[len("events_"):]

    for c in KEEP_TOGETHER_COLLECTIONS:
        prefix = c + "_"
        if name.startswith(prefix):
            return c, name[len(prefix):]

    if "_" in name:
        c, v = name.split("_", 1)
        return c, v

    return "Event", name


def is_signal_dataset(dataset_key: str):
    return not dataset_key.startswith("DATA")


def pick_region(is_signal: bool):
    return "4b_signal_region" if is_signal else "2b_signal_region_postW"


def ensure_resizable_dataset(group, path_parts, data, compression="gzip"):
    """
    Create or append to a resizable dataset located at group/<path_parts...>.
    data: numpy array, first dimension is N (events)
    """
    g = group
    for p in path_parts[:-1]:
        if p not in g:
            g = g.create_group(p)
        else:
            g = g[p]
    name = path_parts[-1]

    data = np.asarray(data)

    if data.dtype == object:
        raise TypeError(
            f"Refusing to write object dtype to HDF5 at {group.name}/{'/'.join(path_parts)}. "
            f"Example element type: {type(data.flat[0]) if data.size else None}, shape={data.shape}"
        )

    if name not in g:
        maxshape = (None,) + data.shape[1:]
        dset = g.create_dataset(
            name,
            data=data,
            maxshape=maxshape,
            chunks=True,
            compression=compression,
            shuffle=True,
        )
        return dset

    dset = g[name]
    if dset.shape[1:] != data.shape[1:]:
        raise ValueError(f"Shape mismatch for {g.name}/{name}: existing {dset.shape}, new {data.shape}")

    old_n = dset.shape[0]
    new_n = old_n + data.shape[0]
    dset.resize((new_n,) + dset.shape[1:])
    dset[old_n:new_n, ...] = data
    return dset


def convert_nested_coffea_to_h5(
    coffea_path,
    h5_path,
    columns_key="columns",
    weight_name="weight",
):
    myfile = coffea.util.load(coffea_path)
    cols = myfile[columns_key]

    with h5py.File(h5_path, "w") as f:
        g_inputs = f.create_group("INPUTS")
        g_weights = f.create_group("WEIGHTS")
        g_class = f.create_group("CLASSIFICATIONS")
        g_targets = f.create_group("TARGETS")

        for dataset_key in cols.keys():
            sig = is_signal_dataset(dataset_key)
            region_key = pick_region(sig)

            for era_key in cols[dataset_key].keys():
                if region_key not in cols[dataset_key][era_key]:
                    continue

                payload = cols[dataset_key][era_key][region_key]

                # --- weights ---
                if weight_name not in payload:
                    raise KeyError(f'Missing "{weight_name}" in {dataset_key}/{era_key}/{region_key}')
                w = to_numpy_event_vector(payload[weight_name])
                # TODO make better. Currently just a placeholder basically
                # --- TARGETS: fixed pairing indices (event-level) ---
                N = w.shape[0]
                ensure_resizable_dataset(g_targets, ["h1", "b1"], np.zeros(N, dtype=np.int16))
                ensure_resizable_dataset(g_targets, ["h1", "b2"], np.ones(N, dtype=np.int16))
                ensure_resizable_dataset(g_targets, ["h2", "b1"], np.full(N, 2, dtype=np.int16))
                ensure_resizable_dataset(g_targets, ["h2", "b2"], np.full(N, 3, dtype=np.int16))


                signal_arr = np.full_like(w, 1 if sig else 0, dtype=np.int8)

                ensure_resizable_dataset(g_weights, ["weights"], w)
                ensure_resizable_dataset(g_class, ["Event", "signal"], signal_arr)

                jet_counts = None
                jetN_key = "JetGoodFromHiggsOrdered_N"
                if jetN_key in payload:
                    jet_counts = to_numpy_event_vector(payload[jetN_key]).astype(np.int64)

                # --- variables ---
                jet_mask_written = False

                for name, arr in payload.items():
                    if name == weight_name:
                        continue

                    collection, var = infer_collection_and_var(name)

                    # Always unwrap once here
                    arr_u = unwrap_accumulator(arr)

                    # FORCE JetGoodFromHiggsOrdered_* (except N) into Jet
                    is_candidate_jet = (collection in JET_COLLECTIONS and var != "N")

                    if is_candidate_jet:
                        arr_u = unwrap_accumulator(arr)

                        # Case 1: flattened jets (1D of length sum(N))
                        if np.asarray(arr_u).ndim == 1 and jet_counts is not None and var != "N":
                            jagged = unflatten_to_jagged(arr_u, jet_counts)
                            dense, mask = pad_clip_jets_with_mask(jagged, max_jets=MAX_JETS, fill_value=-9999)

                            ensure_resizable_dataset(g_inputs, ["Jet", var], dense)

                            if not jet_mask_written:
                                # mask based on counts is equivalent; but using padded jagged is fine too
                                ensure_resizable_dataset(g_inputs, ["Jet", "MASK"], mask.astype(np.bool_))
                                jet_mask_written = True

                        # Case 2: already jagged/dense jets (your existing pad function can handle it)
                        else:
                            dense, mask = pad_clip_jets_with_mask(arr_u, max_jets=MAX_JETS, fill_value=-9999)

                            ensure_resizable_dataset(g_inputs, ["Jet", var], dense)

                            if not jet_mask_written:
                                ensure_resizable_dataset(g_inputs, ["Jet", "MASK"], mask.astype(np.bool_))
                                jet_mask_written = True

                    else:
                        # event-level
                        arr_np = to_numpy_event_vector(arr_u)
                        ensure_resizable_dataset(g_inputs, [collection, var], arr_np)



    print(f"Wrote merged HDF5: {h5_path}")


if __name__ == "__main__":
    convert_nested_coffea_to_h5("output_all.coffea", "columns_for_classifier.h5")

