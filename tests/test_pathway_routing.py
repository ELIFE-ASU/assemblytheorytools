"""Rendered pathway geometry stays readable across backends and figure sizes."""

from io import BytesIO
from itertools import combinations

import networkx as nx
import numpy as np
import pytest
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from rdkit import Chem

import assemblytheorytools as att
import assemblytheorytools.tools_plotting as plotting


def _node_boxes(ax, renderer):
    boxes = [
        artist.patch.get_window_extent(renderer)
        for artist in ax.artists
        if isinstance(artist, AnnotationBbox)
    ]
    boxes.extend(
        text.get_bbox_patch().get_window_extent(renderer)
        for text in ax.texts
        if text.get_bbox_patch() is not None
    )
    return boxes


def _arrow_paths(ax):
    return [
        patch.get_transform().transform_path(patch.get_path())
        for patch in ax.patches
        if isinstance(patch, FancyArrowPatch)
    ]


def _assert_clear_pathway(fig, ax, renderer, node_count, edge_count):
    boxes = _node_boxes(ax, renderer)
    paths = _arrow_paths(ax)
    assert len(boxes) == node_count
    assert len(paths) >= edge_count
    for box in boxes:
        assert fig.bbox.x0 - 1 <= box.x0 < box.x1 <= fig.bbox.x1 + 1
        assert fig.bbox.y0 - 1 <= box.y0 < box.y1 <= fig.bbox.y1 + 1
        # Avoid backend rounding at the border while still rejecting an arrow
        # which enters an image or label, including either endpoint's card.
        assert not any(
            path.intersects_bbox(box.padded(-0.5), filled=False) for path in paths
        )
    assert not any(
        a.padded(-0.5).overlaps(b.padded(-0.5)) for a, b in combinations(boxes, 2)
    )


def _skip_layer_pathway(plot_type):
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 4),
            (2, 5),
            (3, 5),
            (3, 6),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 7),
            (1, 7),
        ]
    )
    labels = ["A", "B", "AB", "ABA", "BAB", "ABABA", "BABAB", "ABABABAB"]
    for node in graph:
        graph.nodes[node]["vo"] = (
            "C" * (node + 1) if plot_type == "mol" else labels[node]
        )
    return graph


@pytest.mark.parametrize("plot_type", ["mol", "string"])
@pytest.mark.parametrize("arrow_style", ["1", "2"])
def test_pathway_arrows_avoid_cards_including_long_edges(plot_type, arrow_style):
    graph = _skip_layer_pathway(plot_type)
    fig, ax = att.plot_pathway(
        graph, plot_type=plot_type, arrow_style=arrow_style, fig_size=(10, 6)
    )
    fig.canvas.draw()

    _assert_clear_pathway(
        fig, ax, fig.canvas.get_renderer(), len(graph), graph.number_of_edges()
    )
    assert len(_arrow_paths(ax)) == graph.number_of_edges()


@pytest.mark.parametrize("plot", [att.plot_pathway, att.plot_pathway_mid_arrow])
def test_real_molecular_pathway_stays_clear_when_resized_and_exported(
    plot, data_dir, monkeypatch
):
    molecule = att.molfile_to_mol(
        str(data_dir / "mol_files" / "anthracene.mol"), add_hydrogens=False
    )
    graph = att.parse_pathway_dot(
        (data_dir / "pathway" / "anthracene_pathway.dot").read_text(),
        mol=molecule,
        vo_type="smiles",
    )
    fig, ax = plot(graph, fig_size=(10, 6))
    original_draw = fig.draw
    draws = []

    def checked_draw(renderer):
        original_draw(renderer)
        _assert_clear_pathway(fig, ax, renderer, len(graph), graph.number_of_edges())
        draws.append(type(renderer).__name__)

    monkeypatch.setattr(fig, "draw", checked_draw)
    fig.canvas.draw()
    fig.set_size_inches(8, 4.5)
    fig.canvas.draw()
    for fmt, dpi in (("png", 72), ("png", 200), ("svg", 120)):
        with BytesIO() as output:
            fig.savefig(output, format=fmt, dpi=dpi, bbox_inches="tight")
            assert len(output.getvalue()) > 1000
    fig.canvas.draw()
    assert len(draws) >= 6


def test_parallel_edges_are_distinct_and_clear_of_their_endpoint_boxes():
    graph = nx.MultiDiGraph()
    graph.add_node("source", vo="CC")
    graph.add_node("target", vo="CCCC")
    graph.add_edges_from([("source", "target"), ("source", "target")])
    fig, ax = att.plot_pathway(graph, fig_size=(6, 3))
    fig.canvas.draw()

    _assert_clear_pathway(fig, ax, fig.canvas.get_renderer(), 2, 2)
    paths = _arrow_paths(ax)
    assert len(paths) == 2
    assert paths[0].vertices.shape != paths[1].vertices.shape or not np.allclose(
        paths[0].vertices, paths[1].vertices
    )


@pytest.mark.parametrize("edge_count", [10, 16])
def test_parallel_edge_lanes_fit_inside_a_short_figure(edge_count):
    graph = nx.MultiDiGraph()
    graph.add_node(0, vo="CC")
    graph.add_node(1, vo="CCCC")
    graph.add_edges_from([(0, 1)] * edge_count)
    fig, ax = att.plot_pathway(graph, fig_size=(6, 2))

    _assert_clear_pathway(fig, ax, fig.canvas.get_renderer(), 2, edge_count)
    for path in _arrow_paths(ax):
        assert all(ax.bbox.contains(*point) for point in path.vertices)


@pytest.mark.parametrize("nodes,characters,size", [(4, 48, (6, 4)), (8, 100, (3, 2))])
def test_long_string_labels_use_measured_font_sizes_when_fitted(
    nodes, characters, size
):
    graph = nx.path_graph(nodes, create_using=nx.DiGraph)
    label = ("ABCD" * characters)[:characters]
    nx.set_node_attributes(graph, label, "vo")
    fig, ax = att.plot_pathway(graph, plot_type="string", fig_size=size)

    _assert_clear_pathway(fig, ax, fig.canvas.get_renderer(), nodes, nodes - 1)
    assert all(text.get_text().replace("\n", "") == label for text in ax.texts)
    # Repeated draws and a larger canvas must use the original label, not wrap
    # an already wrapped value or mutate the pathway itself.
    fig.set_size_inches(16, 6)
    fig.canvas.draw()
    _assert_clear_pathway(fig, ax, fig.canvas.get_renderer(), nodes, nodes - 1)
    assert all(text.get_text().replace("\n", "") == label for text in ax.texts)
    assert all(data["vo"] == label for _, data in graph.nodes(data=True))


@pytest.mark.parametrize("style", ["simple", "fancy", "wedge", ArrowStyle.Simple()])
@pytest.mark.parametrize("position", [0, 0.5, 1])
def test_filled_arrow_styles_support_routed_edges(style, position):
    graph = _skip_layer_pathway("string")
    fig, ax = att.plot_pathway(
        graph, plot_type="string", plt_arrow_style=style, arrow_pos=position
    )
    fig.canvas.draw()

    _assert_clear_pathway(
        fig, ax, fig.canvas.get_renderer(), len(graph), graph.number_of_edges()
    )


@pytest.mark.parametrize("layout", ["crossmin_long", "plain"])
@pytest.mark.parametrize("size,position", [(80, 0.25), (160, 0.75)])
def test_large_midpoint_heads_avoid_intermediate_boxes(layout, size, position):
    graph = _skip_layer_pathway("mol")
    fig, ax = att.plot_pathway_mid_arrow(
        graph,
        layout_style=layout,
        arrow_size=size,
        arrow_pos=position,
        fig_size=(10, 6),
    )

    _assert_clear_pathway(
        fig, ax, fig.canvas.get_renderer(), len(graph), graph.number_of_edges()
    )


@pytest.mark.parametrize("arrow_pos", [0, 0.05])
def test_arrowheads_near_source_do_not_extend_back_into_its_card(arrow_pos):
    graph = nx.DiGraph()
    graph.add_node(0, vo="CC")
    graph.add_node(1, vo="CCCC")
    graph.add_edge(0, 1)
    fig, ax = att.plot_pathway(
        graph, arrow_pos=arrow_pos, arrow_size=40, fig_size=(6, 3)
    )
    fig.canvas.draw()

    _assert_clear_pathway(fig, ax, fig.canvas.get_renderer(), 2, 1)


@pytest.mark.parametrize("representation", ["smiles", "rdkit"])
def test_molecular_pathway_preserves_isotopes_charges_and_input(
    representation, monkeypatch
):
    smiles = "[13CH3][NH3+]"
    virtual_object = Chem.MolFromSmiles(smiles) if representation == "rdkit" else smiles
    graph = nx.DiGraph()
    graph.add_node(0, vo=virtual_object)
    rendered_atoms = []
    draw_molecule = plotting.Draw.MolToImage

    def record_molecule(molecule, *args, **kwargs):
        rendered_atoms.append(
            [
                (atom.GetIsotope(), atom.GetFormalCharge())
                for atom in molecule.GetAtoms()
            ]
        )
        return draw_molecule(molecule, *args, **kwargs)

    monkeypatch.setattr(plotting.Draw, "MolToImage", record_molecule)
    fig, ax = att.plot_pathway(graph)
    fig.canvas.draw()

    assert rendered_atoms == [[(13, 0), (0, 1)]]
    assert graph.nodes[0] == {"vo": virtual_object}
    if representation == "rdkit":
        assert Chem.MolToSmiles(virtual_object) == smiles
        assert virtual_object.GetNumConformers() == 0
    _assert_clear_pathway(fig, ax, fig.canvas.get_renderer(), 1, 0)


@pytest.mark.parametrize("fraction", [0.25, 0.75])
def test_mid_edge_heads_track_visible_arc_length_after_resize(fraction):
    graph = _skip_layer_pathway("string")
    fig, ax = att.plot_pathway_mid_arrow(
        graph, plot_type="string", arrow_pos=fraction, fig_size=(10, 5)
    )
    for size, dpi in [((10, 5), 100), ((6, 5), 180)]:
        fig.set_size_inches(*size)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        bodies, heads = [], []
        for patch in ax.patches:
            if isinstance(patch, FancyArrowPatch):
                path = patch.get_transform().transform_path(patch.get_path())
                if isinstance(patch.get_arrowstyle(), ArrowStyle.Curve):
                    bodies.append(path)
                else:
                    heads.append(path)
        assert len(bodies) == len(heads) == graph.number_of_edges()
        matched = set()
        for body in bodies:
            vertices = np.asarray(
                [point for point, _ in body.iter_segments(curves=False)]
            )
            distance = np.r_[
                0, np.linalg.norm(np.diff(vertices, axis=0), axis=1).cumsum()
            ]
            expected = np.array(
                [
                    np.interp(fraction * distance[-1], distance, vertices[:, axis])
                    for axis in (0, 1)
                ]
            )
            errors = [
                np.linalg.norm(head.vertices - expected, axis=1).min() for head in heads
            ]
            closest = int(np.argmin(errors))
            assert errors[closest] < 2
            matched.add(closest)
        assert len(matched) == graph.number_of_edges()


@pytest.mark.parametrize("node_count", [0, 1])
@pytest.mark.parametrize("plot_type", ["mol", "string"])
def test_empty_and_single_node_pathways_have_no_arrows(node_count, plot_type):
    graph = nx.DiGraph()
    if node_count:
        graph.add_node(0, vo="C")
    fig, ax = att.plot_pathway(graph, plot_type=plot_type, fig_size=(4, 3))
    fig.canvas.draw()

    _assert_clear_pathway(fig, ax, fig.canvas.get_renderer(), node_count, 0)
    assert not _arrow_paths(ax)
    assert not ax.axison
