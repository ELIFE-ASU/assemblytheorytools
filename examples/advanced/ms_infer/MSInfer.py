import json
from bisect import bisect_right
from typing import List, Dict, Any

import numpy as np
from pyopenms import MSExperiment, MzMLFile


# === Utilities ===
def ppm_to_da(mz: float, ppm: float) -> float:
    return mz * ppm / 1e6


# === Load spectra once ===
def load_spectra(fname: str) -> MSExperiment:
    exp = MSExperiment()
    MzMLFile().load(fname, exp)
    exp.updateRanges()
    return exp


# === Build node records, store NumPy arrays for peaks; MS1 centroid computed here ===
def extract_node_fast(spec, index: int, min_intensity=0) -> Dict[str, Any]:
    ms_level = int(spec.getMSLevel())
    mz_array, int_array = spec.get_peaks()
    if len(mz_array) == 0:
        mz_arr = np.empty(0, dtype=float)
        int_arr = np.empty(0, dtype=float)
    else:
        mz_arr = np.array(mz_array, dtype=float)
        int_arr = np.array(int_array, dtype=float)

    if min_intensity > 0:
        mask = int_arr >= min_intensity
        mz_arr = mz_arr[mask]
        int_arr = int_arr[mask]

    # Determine precursor_mz:
    precursor_mz = None
    precursors = spec.getPrecursors()
    if precursors:
        pc = precursors[0]
        if pc.getMZ() > 0:
            precursor_mz = float(pc.getMZ())

    # If this is an MS1 spectrum (no precursor), compute intensity-weighted centroid as parent m/z
    if precursor_mz is None and ms_level == 1 and mz_arr.size > 0 and int_arr.size > 0:
        # Avoid division by zero if all intensities are zero
        total_int = float(np.sum(int_arr))
        if total_int > 0:
            centroid = float(np.sum(mz_arr * int_arr) / total_int)
            precursor_mz = centroid
        else:
            # fallback to base peak if intensities all zero
            idx = int(np.argmax(int_arr)) if int_arr.size > 0 else None
            if idx is not None and idx >= 0:
                precursor_mz = float(mz_arr[idx])

    # Isolation window if present
    isolation_win = None
    if precursors:
        try:
            lower = float(precursors[0].getIsolationWindowLowerOffset())
            upper = float(precursors[0].getIsolationWindowUpperOffset())
            if precursor_mz is not None:
                isolation_win = (precursor_mz, lower, upper)
        except Exception:
            isolation_win = None

    return {
        "id": spec.getNativeID() or f"scan={index}",
        "index": index,
        "rt": float(spec.getRT()),  # seconds (PyOpenMS uses seconds)
        "ms_level": ms_level,
        "peaks_mz": mz_arr,  # numpy array
        "peaks_int": int_arr,  # numpy array
        "precursor_mz": precursor_mz,  # now set for MS1 via centroid if missing
        "isolation": isolation_win,
        "children": [],
        "parent_node": None
    }


def build_nodes(exp: MSExperiment, min_intensity=0) -> List[Dict[str, Any]]:
    nodes = []
    for i, spec in enumerate(exp.getSpectra()):
        nodes.append(extract_node_fast(spec, i, min_intensity))
    return nodes


# === Create fast indexes by MS level (sorted by RT) ===
def build_level_index(nodes: List[Dict[str, Any]]):
    ms_by_level = {}
    for n in nodes:
        ms_by_level.setdefault(n["ms_level"], []).append(n)
    # sort each level list by RT and build parallel RT lists for bisecting
    rt_index = {}
    for level, lst in ms_by_level.items():
        lst.sort(key=lambda x: x["rt"])
        rt_index[level] = [n["rt"] for n in lst]
    return ms_by_level, rt_index


# === Attach children using RT-limited search + vectorized peak matching ===
def attach_children_fast(nodes: List[Dict[str, Any]],
                         ppm_tol=10,
                         rt_window=60.0,
                         max_candidates=200,
                         MIN_FRAGMENT_LOSS=0.003):
    ms_by_level, rt_index = build_level_index(nodes)

    def candidate_parents_for(child):
        target_level = child["ms_level"] - 1
        if target_level not in ms_by_level:
            return []
        rts = rt_index[target_level]
        lst = ms_by_level[target_level]
        pos = bisect_right(rts, child["rt"])
        left_pos = bisect_right(rts, child["rt"] - rt_window)
        start = max(left_pos, 0)
        end = pos  # exclusive
        if end - start > max_candidates:
            start = end - max_candidates
        return lst[start:end]

    for child in nodes:
        if child["ms_level"] <= 1:
            continue
        prec_mz = child["precursor_mz"]
        if prec_mz is None:
            continue

        best_parent = None
        best_score = None  # (scan_dist, mz_diff)

        candidates = candidate_parents_for(child)
        for parent in reversed(candidates):
            if parent["index"] >= child["index"]:
                continue
            if parent["ms_level"] != child["ms_level"] - 1:
                continue

            # If parent has its own precursor (e.g., MS1 centroid or MS2 precursor), enforce fragment loss
            parent_prec = parent.get("precursor_mz")
            if parent_prec is not None:
                mass_loss = parent_prec - prec_mz
                if mass_loss < MIN_FRAGMENT_LOSS:
                    continue

            parent_mz_arr = parent["peaks_mz"]
            if parent_mz_arr.size == 0:
                continue

            # Compute tolerance in Da relative to max(precursor, parent_prec if available)
            ref_mz_for_tol = max(prec_mz, parent_prec if parent_prec is not None else prec_mz)
            tol_da = ppm_to_da(ref_mz_for_tol, ppm_tol)

            diffs = np.abs(parent_mz_arr - prec_mz)
            mask = diffs <= tol_da
            if not np.any(mask):
                continue

            scan_dist = child["index"] - parent["index"]
            min_mz_diff = float(np.min(diffs[mask])) if np.any(mask) else float("inf")
            score = (scan_dist, min_mz_diff)

            if best_score is None or score < best_score:
                best_parent = parent
                best_score = score

            if score[0] == 1 and score[1] == 0.0:
                break

        if best_parent is not None:
            best_parent["children"].append(child)
            child["parent_node"] = best_parent


# === Virtual MS1 creation (fallback) ===
def create_virtual_ms1_nodes(nodes, ppm_tol=10):
    all_children = {c["id"] for p in nodes for c in p["children"]}
    ms2_roots = [n for n in nodes if n["id"] not in all_children and n["ms_level"] == 2]

    if not ms2_roots:
        return nodes

    ms1_groups = []
    for ms2_node in ms2_roots:
        prec_mz = ms2_node["precursor_mz"]
        if prec_mz is None:
            continue

        found_group = False
        for group in ms1_groups:
            group_mz = group["precursor_mz"]
            tol = ppm_to_da(max(prec_mz, group_mz), ppm_tol)
            if abs(prec_mz - group_mz) <= tol:
                group["children"].append(ms2_node)
                ms2_node["parent_node"] = group
                found_group = True
                break

        if not found_group:
            virtual_ms1 = {
                "id": f"virtual_ms1_mz_{prec_mz:.4f}",
                "index": -1,
                "rt": ms2_node["rt"],
                "ms_level": 1,
                "peaks_mz": np.array([prec_mz], dtype=float),
                "peaks_int": np.array([1e6], dtype=float),
                "precursor_mz": prec_mz,
                "isolation": None,
                "children": [ms2_node],
                "parent_node": None
            }
            ms2_node["parent_node"] = virtual_ms1
            ms1_groups.append(virtual_ms1)

    return nodes + ms1_groups


# === Tree functions (same logic but using precursor_mz as node mz) ===
def shrink_node(node, parent_mz_value=None):
    mz = node["precursor_mz"]
    return {
        "id": node["id"],
        "rt": node["rt"],
        "ms_level": node["ms_level"],
        "mz": mz,
        "parent_mz": parent_mz_value,
        "children": [shrink_node(c, mz) for c in node["children"]]
    }


def collapse_matching_fragments(node, ppm_tol=10):
    node_mz = node["mz"]
    new_children = []
    for child in node["children"]:
        child_mz = child["mz"]
        if child_mz is not None and node_mz is not None:
            tol = ppm_to_da(max(node_mz, child_mz), ppm_tol)
            if abs(child_mz - node_mz) <= tol:
                for grandchild in child["children"]:
                    processed = collapse_matching_fragments(grandchild, ppm_tol)
                    new_children.extend(processed)
                continue
        processed = collapse_matching_fragments(child, ppm_tol)
        new_children.extend(processed)
    node["children"] = new_children
    return [node]


def convert_to_mz_tree(node):
    mz = node["mz"]
    children_dict = {}
    for child in node["children"]:
        child_tree = convert_to_mz_tree(child)
        children_dict[child["mz"]] = child_tree
    return children_dict


def merge_duplicate_fragments(mz_dict: Dict[float, Dict], ppm_tol=10):
    if not mz_dict:
        return {}

    items = sorted(mz_dict.items(), key=lambda x: x[0])
    merged = {}
    i = 0
    while i < len(items):
        mz_i, subtree_i = items[i]
        group_subtrees = [subtree_i]
        j = i + 1
        while j < len(items):
            mz_j, subtree_j = items[j]
            tol = ppm_to_da(max(mz_i, mz_j), ppm_tol)
            if abs(mz_j - mz_i) <= tol:
                group_subtrees.append(subtree_j)
                j += 1
            else:
                break
        combined = {}
        for st in group_subtrees:
            for child_mz, child_tree in st.items():
                combined[child_mz] = merge_two_trees(combined.get(child_mz, {}), child_tree, ppm_tol)
        merged[mz_i] = merge_duplicate_fragments(combined, ppm_tol)
        i = j

    return merged


def merge_two_trees(tree1, tree2, ppm_tol=10):
    if not tree1:
        return tree2.copy()
    combined = tree1.copy()
    for mz2, subtree2 in tree2.items():
        matched = False
        for mz1 in list(combined.keys()):
            if abs(mz1 - mz2) <= ppm_to_da(max(mz1, mz2), ppm_tol):
                combined[mz1] = merge_two_trees(combined[mz1], subtree2, ppm_tol)
                matched = True
                break
        if not matched:
            combined[mz2] = subtree2
    return combined


def find_roots(nodes):
    all_children = {c["id"] for p in nodes for c in p["children"]}
    roots = [n for n in nodes if n["id"] not in all_children and n["ms_level"] == 1]
    if not roots:
        roots = [n for n in nodes if n["id"] not in all_children]
    return roots


# testing:
if __name__ == "__main__":
    # === User params ===
    FNAME = ""  # add path to mzml file
    MZ_TOL_PPM = 10.0
    MIN_INTENSITY = 0.0

    # Performance tuning
    RT_WINDOW = 60.0  # seconds to search backwards for parent candidates
    MAX_PARENT_CANDIDATES = 200  # limit candidate parents checked per child (to bound worst-case work)
    MIN_FRAGMENT_LOSS = 0.003  # minimum Da loss to consider real fragmentation

    print("Loading spectra...")
    exp = load_spectra(FNAME)
    print(f"Spectra count: {len(exp.getSpectra())}")

    print("Building nodes...")
    nodes = build_nodes(exp, MIN_INTENSITY)

    print("Attaching children (fast)...")
    attach_children_fast(nodes, ppm_tol=MZ_TOL_PPM, rt_window=RT_WINDOW, max_candidates=MAX_PARENT_CANDIDATES)

    any_ms1 = any(n["ms_level"] == 1 for n in nodes)
    if not any_ms1:
        print("No MS1 found — creating virtual MS1 nodes for MS2 roots...")
        nodes = create_virtual_ms1_nodes(nodes, ppm_tol=MZ_TOL_PPM)

    roots = find_roots(nodes)
    print(f"Found {len(roots)} root(s). Shrinking tree...")
    tree = [shrink_node(r) for r in roots]

    OUT = "ms_tree_pyopenms_fast_centroid.json"
    with open(OUT, "w") as f:
        json.dump(tree, f, indent=2)
    print(f"Tree built! Output: {OUT}")

    print("Collapsing matching fragments...")
    collapsed_trees = []
    for root in tree:
        collapsed = collapse_matching_fragments(root)
        collapsed_trees.extend(collapsed)

    print("Converting to m/z-only trees...")
    mz_trees = []
    for root in collapsed_trees:
        root_mz = root["mz"]
        children_tree = convert_to_mz_tree(root)
        mz_trees.append({root_mz: children_tree})

    print("Merging duplicate fragments (fast)...")
    mz_trees_merged = []
    for tree_dict in mz_trees:
        merged_tree = {}
        for root_mz, root_tree in tree_dict.items():
            merged_tree[root_mz] = merge_duplicate_fragments(root_tree, ppm_tol=MZ_TOL_PPM)
        mz_trees_merged.append(merged_tree)

    OUT_MZ = "ms_tree_mz_only_fast_centroid.json"
    with open(OUT_MZ, "w") as f:
        json.dump(mz_trees_merged, f, indent=2)
    print(f"M/Z tree saved to: {OUT_MZ} ({len(mz_trees_merged)} molecules)")

    print("Creating depth-pruned versions...")


    def get_max_depth(mz_dict, current_depth=1):
        if not mz_dict:
            return current_depth
        max_child_depth = current_depth
        for child_tree in mz_dict.values():
            child_depth = get_max_depth(child_tree, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth


    def prune_to_depth(mz_dict, max_depth, current_depth=1):
        if current_depth >= max_depth:
            return {}
        pruned = {}
        for mz, child_tree in mz_dict.items():
            pruned[mz] = prune_to_depth(child_tree, max_depth, current_depth + 1)
        return pruned


    max_depth = 0
    for tree_dict in mz_trees_merged:
        for root_mz, root_tree in tree_dict.items():
            depth = get_max_depth(root_tree)
            if depth > max_depth:
                max_depth = depth

    depth_pruned = {}
    for level in range(1, max_depth + 1):
        pruned_trees = []
        for tree_dict in mz_trees_merged:
            pruned_tree = {}
            for root_mz, root_tree in tree_dict.items():
                pruned_tree[root_mz] = prune_to_depth(root_tree, level)
            pruned_trees.append(pruned_tree)
        depth_pruned[level] = pruned_trees

    OUT_PRUNED = "ms_tree_depth_pruned_fast_centroid.json"
    with open(OUT_PRUNED, "w") as f:
        json.dump(depth_pruned, f, indent=2)

    print(f"Depth-pruned trees saved to: {OUT_PRUNED}")
    print(f"Maximum tree depth: {max_depth}")
    print(f"Pruned levels: {list(depth_pruned.keys())}")
