import json
from bisect import bisect_right
from typing import List, Dict, Any

import numpy as np
from pyopenms import MSExperiment, MzMLFile


def ppm_to_da(mz: float, ppm: float) -> float:
    """
    Convert a mass-to-charge ratio (m/z) value from parts-per-million (ppm) to Daltons (Da).

    This utility function calculates the equivalent mass difference in Daltons
    for a given m/z value and a specified tolerance in ppm.

    Parameters
    ----------
    mz : float
        The mass-to-charge ratio (m/z) value.
    ppm : float
        The tolerance in parts-per-million (ppm).

    Returns
    -------
    float
        The equivalent mass difference in Daltons (Da).
    """
    return mz * ppm / 1e6


def load_spectra(fname: str) -> MSExperiment:
    """
    Load an mzML file and return an MSExperiment object.

    This function reads an mzML file, loads its contents into an MSExperiment
    object, and updates the experiment's ranges.

    Parameters
    ----------
    fname : str
        The file path to the mzML file to be loaded.

    Returns
    -------
    MSExperiment
        An MSExperiment object containing the loaded spectra data.
    """
    exp = MSExperiment()
    MzMLFile().load(fname, exp)
    exp.updateRanges()
    return exp


def extract_node_fast(spec, index: int, min_intensity=0.0) -> Dict[str, Any]:
    """
    Extract information from a spectrum and create a node representation.

    This function processes a given spectrum to extract its metadata, peaks, and
    precursor information. It also computes the intensity-weighted centroid for
    MS1 spectra if no precursor is available.

    Parameters
    ----------
    spec : pyopenms.MSSpectrum
        The spectrum object to extract data from.
    index : int
        The index of the spectrum in the experiment.
    min_intensity : float, optional
        The minimum intensity threshold for filtering peaks. Defaults to 0.

    Returns
    -------
    dict
        A dictionary containing the extracted node information, including:
        - id: The native ID or a generated scan ID.
        - index: The spectrum index.
        - rt: The retention time in seconds.
        - ms_level: The MS level of the spectrum.
        - peaks_mz: A NumPy array of m/z values for the peaks.
        - peaks_int: A NumPy array of intensity values for the peaks.
        - precursor_mz: The precursor m/z value, if available.
        - isolation: The isolation window as a tuple (precursor_mz, lower, upper), if available.
        - children: An empty list to store child nodes.
        - parent_node: None, to be set later during tree construction.
    """
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


def build_nodes(exp: MSExperiment, min_intensity=0.0) -> List[Dict[str, Any]]:
    """
    Build a list of nodes from an MSExperiment object.

    This function iterates through the spectra in the given MSExperiment object,
    processes each spectrum using the `extract_node_fast` function, and returns
    a list of nodes representing the spectra.

    Parameters
    ----------
    exp : MSExperiment
        The MSExperiment object containing the spectra to process.
    min_intensity : float, optional
        The minimum intensity threshold for filtering peaks. Defaults to 0.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents a node
        extracted from a spectrum.
    """
    nodes = []
    for i, spec in enumerate(exp.getSpectra()):
        nodes.append(extract_node_fast(spec, i, min_intensity))
    return nodes


def build_level_index(nodes: List[Dict[str, Any]]):
    """
    Build an index of nodes grouped by MS level and sorted by retention time (RT).

    This function organizes the given nodes into a dictionary where the keys are
    MS levels and the values are lists of nodes belonging to that level. Each list
    is sorted by the retention time (RT). Additionally, it creates a parallel index
    of RT values for each MS level to enable efficient bisecting.

    Parameters
    ----------
    nodes : List[Dict[str, Any]]
        A list of nodes, where each node is a dictionary containing metadata
        such as MS level and retention time.

    Returns
    -------
    Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[float]]]
        - A dictionary mapping MS levels to lists of nodes sorted by RT.
        - A dictionary mapping MS levels to lists of RT values for bisecting.
    """
    ms_by_level = {}
    for n in nodes:
        ms_by_level.setdefault(n["ms_level"], []).append(n)
    # sort each level list by RT and build parallel RT lists for bisecting
    rt_index = {}
    for level, lst in ms_by_level.items():
        lst.sort(key=lambda x: x["rt"])
        rt_index[level] = [n["rt"] for n in lst]
    return ms_by_level, rt_index


def attach_children_fast(nodes: List[Dict[str, Any]],
                         ppm_tol=10.0,
                         rt_window=60.0,
                         max_candidates=200,
                         min_fragment_loss=0.003):
    """
    Attach child nodes to their parent nodes based on precursor m/z and retention time.

    This function iterates through a list of nodes, identifies potential parent nodes
    for each child node, and attaches the child to the best-matching parent. The matching
    is based on precursor m/z, retention time, and other criteria.

    Parameters
    ----------
    nodes : List[Dict[str, Any]]
        A list of nodes, where each node is a dictionary containing metadata such as
        precursor m/z, retention time, and MS level.
    ppm_tol : float, optional
        The mass tolerance in parts-per-million (ppm) for matching precursor m/z values.
        Defaults to 10.0.
    rt_window : float, optional
        The retention time window (in seconds) to search for parent candidates.
        Defaults to 60.0.
    max_candidates : int, optional
        The maximum number of parent candidates to consider for each child node.
        Defaults to 200.
    min_fragment_loss : float, optional
        The minimum mass difference (in Daltons) between a parent precursor and a child
        precursor to consider the match valid. Defaults to 0.003.

    Returns
    -------
    None
        The function modifies the input `nodes` list in place, attaching child nodes
        to their respective parent nodes.
    """
    ms_by_level, rt_index = build_level_index(nodes)
    for child in nodes:
        if child["ms_level"] <= 1:
            continue
        prec_mz = child["precursor_mz"]
        if prec_mz is None:
            continue

        best_parent = None
        best_score = None  # (scan_dist, mz_diff)

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
        candidates = lst[start:end]

        for parent in reversed(candidates):
            if parent["index"] >= child["index"]:
                continue
            if parent["ms_level"] != child["ms_level"] - 1:
                continue

            # If parent has its own precursor (e.g., MS1 centroid or MS2 precursor), enforce fragment loss
            parent_prec = parent.get("precursor_mz")
            if parent_prec is not None:
                mass_loss = parent_prec - prec_mz
                if mass_loss < min_fragment_loss:
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
    return None


def create_virtual_ms1_nodes(nodes, ppm_tol=10.0):
    """
    Create virtual MS1 nodes for orphaned MS2 nodes.

    This function identifies MS2 nodes that do not have a parent node and creates
    virtual MS1 nodes to act as their parents. The virtual MS1 nodes are grouped
    based on their precursor m/z values within a specified ppm tolerance.

    Parameters
    ----------
    nodes : List[Dict[str, Any]]
        A list of nodes, where each node is a dictionary containing metadata such as
        precursor m/z, retention time, and MS level.
    ppm_tol : float, optional
        The mass tolerance in parts-per-million (ppm) for grouping MS2 nodes under
        virtual MS1 nodes. Defaults to 10.0.

    Returns
    -------
    List[Dict[str, Any]]
        The updated list of nodes, including the newly created virtual MS1 nodes.
    """
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


def shrink_node(node, parent_mz_value=None):
    """
    Recursively shrink a node to a simplified representation.

    This function reduces the information in a node to a minimal structure
    containing its ID, retention time (RT), MS level, m/z value, and parent m/z value.
    It also processes the node's children recursively, applying the same transformation.

    Parameters
    ----------
    node : dict
        The node to be simplified. It is expected to be a dictionary containing
        metadata such as precursor m/z, retention time, MS level, and children.
    parent_mz_value : float, optional
        The m/z value of the parent node. Defaults to None.

    Returns
    -------
    dict
        A simplified representation of the node, including:
        - id: The node's ID.
        - rt: The retention time of the node.
        - ms_level: The MS level of the node.
        - mz: The precursor m/z value of the node.
        - parent_mz: The m/z value of the parent node.
        - children: A list of simplified child nodes.
    """
    mz = node["precursor_mz"]
    return {
        "id": node["id"],
        "rt": node["rt"],
        "ms_level": node["ms_level"],
        "mz": mz,
        "parent_mz": parent_mz_value,
        "children": [shrink_node(c, mz) for c in node["children"]]
    }


def collapse_matching_fragments(node, ppm_tol=10.0):
    """
    Collapse child nodes that match the parent node's m/z value within a specified tolerance.

    This function recursively processes a node's children, merging any child nodes whose
    m/z values are within the given ppm tolerance of the parent node's m/z value. Matching
    child nodes are replaced by their own children, effectively collapsing the hierarchy.

    Parameters
    ----------
    node : dict
        The node to process. It is expected to be a dictionary containing metadata such as
        m/z value, children, and other attributes.
    ppm_tol : float, optional
        The mass tolerance in parts-per-million (ppm) for determining whether a child's
        m/z value matches the parent's m/z value. Defaults to 10.0.

    Returns
    -------
    list
        A list containing the processed node with its children updated.
    """
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
    """
    Convert a node and its children into an m/z-based tree structure.

    This function recursively processes a node and its children, creating a dictionary
    where the keys are the m/z values of the children and the values are their respective
    subtrees. The resulting structure represents the hierarchy of nodes based on their
    m/z values.

    Parameters
    ----------
    node : dict
        The node to convert. It is expected to be a dictionary containing metadata such as
        m/z value and children.

    Returns
    -------
    dict
        A dictionary representing the m/z-based tree structure, where:
        - Keys are the m/z values of the children.
        - Values are the subtrees of the corresponding children.
    """
    children_dict = {}
    for child in node["children"]:
        child_tree = convert_to_mz_tree(child)
        children_dict[child["mz"]] = child_tree
    return children_dict


def merge_duplicate_fragments(mz_dict: Dict[float, Dict], ppm_tol=10.0):
    """
    Merge duplicate fragments in an m/z-based tree structure.

    This function processes a dictionary representing an m/z-based tree structure,
    merging subtrees whose m/z values are within a specified ppm tolerance. The
    merging is performed recursively, combining matching subtrees into a single
    structure.

    Parameters
    ----------
    mz_dict : Dict[float, Dict]
        A dictionary where the keys are m/z values and the values are subtrees
        representing the hierarchy of fragments.
    ppm_tol : float, optional
        The mass tolerance in parts-per-million (ppm) for determining whether
        two m/z values are considered duplicates. Defaults to 10.0.

    Returns
    -------
    Dict[float, Dict]
        A new dictionary with duplicate fragments merged based on the specified
        ppm tolerance.
    """
    if not mz_dict:
        return {}

    # Sort the items by m/z value for sequential processing
    items = sorted(mz_dict.items(), key=lambda x: x[0])
    merged = {}
    i = 0

    # Iterate through the sorted items
    while i < len(items):
        mz_i, subtree_i = items[i]
        group_subtrees = [subtree_i]
        j = i + 1

        # Group subtrees with m/z values within the tolerance
        while j < len(items):
            mz_j, subtree_j = items[j]
            tol = ppm_to_da(max(mz_i, mz_j), ppm_tol)
            if abs(mz_j - mz_i) <= tol:
                group_subtrees.append(subtree_j)
                j += 1
            else:
                break

        # Combine the grouped subtrees
        combined = {}
        for st in group_subtrees:
            for child_mz, child_tree in st.items():
                combined[child_mz] = merge_two_trees(combined.get(child_mz, {}), child_tree, ppm_tol=ppm_tol)

        # Add the merged subtree to the result
        merged[mz_i] = merge_duplicate_fragments(combined, ppm_tol=ppm_tol)
        i = j

    return merged


def merge_two_trees(tree1, tree2, ppm_tol=10.0):
    """
    Merge two m/z-based tree structures.

    This function combines two m/z-based tree structures into a single tree. Nodes
    with m/z values within the specified ppm tolerance are merged recursively. If
    no match is found for a node, it is added to the combined tree as is.

    Parameters
    ----------
    tree1 : dict
        The first tree to merge, represented as a dictionary where keys are m/z
        values and values are subtrees.
    tree2 : dict
        The second tree to merge, represented in the same format as `tree1`.
    ppm_tol : float, optional
        The mass tolerance in parts-per-million (ppm) for determining whether two
        m/z values are considered duplicates. Defaults to 10.0.

    Returns
    -------
    dict
        A new dictionary representing the merged tree structure.
    """
    if not tree1:
        return tree2.copy()
    combined = tree1.copy()
    for mz2, subtree2 in tree2.items():
        matched = False
        for mz1 in list(combined.keys()):
            if abs(mz1 - mz2) <= ppm_to_da(max(mz1, mz2), ppm_tol):
                combined[mz1] = merge_two_trees(combined[mz1], subtree2, ppm_tol=ppm_tol)
                matched = True
                break
        if not matched:
            combined[mz2] = subtree2
    return combined


def find_roots(nodes):
    """
    Identify root nodes in a list of nodes.

    This function determines the root nodes from a given list of nodes. A root node
    is defined as a node that is not a child of any other node. If no MS1-level root
    nodes are found, it defaults to returning all nodes that are not children.

    Parameters
    ----------
    nodes : List[Dict[str, Any]]
        A list of nodes, where each node is a dictionary containing metadata such as
        'id', 'ms_level', and 'children'.

    Returns
    -------
    List[Dict[str, Any]]
        A list of root nodes.
    """
    all_children = {c["id"] for p in nodes for c in p["children"]}
    roots = [n for n in nodes if n["id"] not in all_children and n["ms_level"] == 1]
    if not roots:
        roots = [n for n in nodes if n["id"] not in all_children]
    return roots


def get_max_depth(mz_dict, current_depth=1):
    """
    Calculate the maximum depth of an m/z-based tree structure.

    This function recursively traverses an m/z-based tree structure to determine
    the maximum depth of the tree. The depth is calculated by exploring all child
    nodes and finding the deepest branch.

    Parameters
    ----------
    mz_dict : dict
        A dictionary representing the m/z-based tree structure, where keys are
        m/z values and values are subtrees.
    current_depth : int, optional
        The current depth of the tree during traversal. Defaults to 1.

    Returns
    -------
    int
        The maximum depth of the tree.
    """
    if not mz_dict:
        return current_depth
    max_child_depth = current_depth
    for child_tree in mz_dict.values():
        child_depth = get_max_depth(child_tree, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)
    return max_child_depth


def prune_to_depth(mz_dict, max_depth, current_depth=1):
    """
    Prune an m/z-based tree structure to a specified maximum depth.

    This function recursively traverses an m/z-based tree structure and removes
    all nodes beyond the specified maximum depth. Nodes at the maximum depth
    are replaced with empty dictionaries.

    Parameters
    ----------
    mz_dict : dict
        A dictionary representing the m/z-based tree structure, where keys are
        m/z values and values are subtrees.
    max_depth : int
        The maximum depth to retain in the tree. Nodes deeper than this level
        will be pruned.
    current_depth : int, optional
        The current depth of the tree during traversal. Defaults to 1.

    Returns
    -------
    dict
        A pruned version of the m/z-based tree structure, with nodes beyond
        the specified depth removed.
    """
    if current_depth >= max_depth:
        return {}
    pruned = {}
    for mz, child_tree in mz_dict.items():
        pruned[mz] = prune_to_depth(child_tree, max_depth, current_depth + 1)
    return pruned


if __name__ == "__main__":
    fname = ""  # add path to mzml file
    mz_tol_ppm = 10.0
    min_intensity = 0.0

    # Performance tuning
    rt_window = 60.0  # seconds to search backwards for parent candidates
    max_parent_candidates = 200  # limit candidate parents checked per child (to bound worst-case work)
    min_fragment_loss = 0.003  # minimum Da loss to consider real fragmentation

    print("Loading spectra...")
    exp = load_spectra(fname)
    print(f"Spectra count: {len(exp.getSpectra())}")

    print("Building nodes...")
    nodes = build_nodes(exp, min_intensity)

    print("Attaching children (fast)...")
    attach_children_fast(nodes, ppm_tol=mz_tol_ppm, rt_window=rt_window, max_candidates=max_parent_candidates)

    any_ms1 = any(n["ms_level"] == 1 for n in nodes)
    if not any_ms1:
        print("No MS1 found — creating virtual MS1 nodes for MS2 roots...")
        nodes = create_virtual_ms1_nodes(nodes, ppm_tol=mz_tol_ppm)

    roots = find_roots(nodes)
    print(f"Found {len(roots)} root(s). Shrinking tree...")
    tree = [shrink_node(r) for r in roots]

    out = "ms_tree_pyopenms_fast_centroid.json"
    with open(out, "w") as f:
        json.dump(tree, f, indent=2)
    print(f"Tree built! Output: {out}")

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
            merged_tree[root_mz] = merge_duplicate_fragments(root_tree, ppm_tol=mz_tol_ppm)
        mz_trees_merged.append(merged_tree)

    out_mz = "ms_tree_mz_only_fast_centroid.json"
    with open(out_mz, "w") as f:
        json.dump(mz_trees_merged, f, indent=2)
    print(f"M/Z tree saved to: {out_mz} ({len(mz_trees_merged)} molecules)")

    print("Creating depth-pruned versions...")

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

    out_pruned = "ms_tree_depth_pruned_fast_centroid.json"
    with open(out_pruned, "w") as f:
        json.dump(depth_pruned, f, indent=2)

    print(f"Depth-pruned trees saved to: {out_pruned}")
    print(f"Maximum tree depth: {max_depth}")
    print(f"Pruned levels: {list(depth_pruned.keys())}")
