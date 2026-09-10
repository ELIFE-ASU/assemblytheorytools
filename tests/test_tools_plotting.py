"""Behavioral contracts for plotting styles, data, and layout metadata."""

import builtins
import copy
import importlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image
from rdkit import Chem
from scipy.stats import gaussian_kde

import assemblytheorytools as att
import assemblytheorytools.tools_plotting as plotting


def _cairo_is_available():
    """Report whether cairosvg can load the Cairo system library it binds to.

    cairosvg is a declared dependency, but the Cairo library it wraps through
    cffi is a system package that pip cannot supply; where it is absent the
    import raises OSError. The Conda environment files install it, and the
    Linux CI images ship it, so this only skips on a pip-only macOS or bare
    Linux environment.
    """
    try:
        importlib.import_module("cairosvg")
    except (ImportError, OSError):
        return False
    return True


CAIRO_AVAILABLE = _cairo_is_available()


@pytest.mark.parametrize("plot", [plotting.plot_graph, plotting.plot_mol_graph])
def test_graph_spring_layout_preserves_input_and_rendering_options(plot):
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, dict(enumerate(["C", "O", "Xe"])), "color")
    nx.set_edge_attributes(graph, {(0, 1): 1, (1, 2): 2}, "color")
    before = copy.deepcopy(graph)
    options = {
        "layout": "spring",
        "seed": 17,
        "fig_size": (6, 4),
        "f_labs": True,
        "node_size": 123,
        "width": 3,
        "linewidths": 4,
    }

    fig, ax = plot(graph, **options)
    _, repeated = plot(graph, **options)

    assert nx.utils.graphs_equal(graph, before)
    np.testing.assert_array_equal(fig.get_size_inches(), (6, 4))
    assert fig.axes == [ax]
    assert not ax.axison
    assert [text.get_text() for text in ax.texts] == ["0", "1", "2"]
    nodes, edges = ax.collections
    np.testing.assert_allclose(
        nodes.get_offsets(), repeated.collections[0].get_offsets()
    )
    np.testing.assert_array_equal(nodes.get_sizes(), [123])
    np.testing.assert_array_equal(nodes.get_linewidths(), [4])
    np.testing.assert_array_equal(edges.get_linewidths(), [3])
    if plot is plotting.plot_mol_graph:
        np.testing.assert_allclose(
            nodes.get_facecolors(), colors.to_rgba_array(["darkgray", "red", "black"])
        )
        np.testing.assert_allclose(
            edges.get_colors(), colors.to_rgba_array(["black", "green"])
        )


def test_topological_graph_layer_assignment_does_not_change_input():
    graph = nx.DiGraph([(0, 1), (0, 2), (2, 3)])
    before = copy.deepcopy(graph)

    _, ax = plotting.plot_graph(graph, layout="topological")

    assert nx.utils.graphs_equal(graph, before)
    positions = dict(zip(graph, ax.collections[0].get_offsets()))
    assert positions[0][0] < positions[1][0] == positions[2][0] < positions[3][0]


def test_interactive_graph_colors_copy_and_forwards_output_name(monkeypatch):
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "original", "color")
    before = copy.deepcopy(graph)
    saved = []
    monkeypatch.setattr(plotting.Network, "show", lambda self, name: saved.append(name))

    network = plotting.plot_interactive_graph(graph, filename="degree.html")

    assert nx.utils.graphs_equal(graph, before)
    assert saved == ["degree.html"]
    node_colors = {node["id"]: node["color"] for node in network.nodes}
    assert node_colors[0] == node_colors[2] != node_colors[1]
    assert network.width == "900px" and network.height == "900px"


def test_interactive_graph_writes_html_in_requested_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    graph = att.smi_to_nx("c1ccccc1")

    network = plotting.plot_interactive_graph(graph, filename="molecule.html")

    assert "vis.Network" in (tmp_path / "molecule.html").read_text()
    assert len(network.nodes) == graph.number_of_nodes()
    assert len(network.edges) == graph.number_of_edges()


@pytest.mark.skipif(
    not CAIRO_AVAILABLE,
    reason="cairosvg cannot load the Cairo system library",
)
@pytest.mark.parametrize("representation", ["string", "mol", "graph"])
def test_metro_pathway_writes_svg_and_png_without_mutating_graph(
    representation, tmp_path
):
    graph = nx.path_graph(3, create_using=nx.DiGraph)
    for node, smiles in enumerate(["C", "CC", "CCC"]):
        if representation == "mol":
            virtual_object = Chem.MolFromSmiles(smiles)
        elif representation == "graph":
            virtual_object = att.smi_to_nx(smiles, add_hydrogens=False)
        else:
            virtual_object = smiles
        graph.nodes[node]["vo"] = virtual_object
    before = graph.copy()
    filename = tmp_path / "metro"

    plotting.plot_digraph_metro(graph, filename=filename)

    assert nx.utils.graphs_equal(graph, before)
    assert "<svg" in filename.with_suffix(".svg").read_text()
    with Image.open(filename.with_suffix(".png")) as image:
        assert image.format == "PNG"
        assert min(image.size) > 0
        image.verify()


def test_metro_pathway_reports_a_missing_cairo_system_library(monkeypatch, tmp_path):
    """A pip-only environment fails the cairosvg import with OSError, not ImportError."""
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "cairosvg":
            raise OSError('no library called "cairo" was found')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    graph = nx.DiGraph()
    graph.add_node(0, vo="C")

    with pytest.raises(ImportError, match="Cairo system library"):
        plotting.plot_digraph_metro(graph, filename=tmp_path / "metro")

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("plot_kind", ["graph", "pathway", "circle"])
def test_automatic_figure_size_grows_with_node_count(plot_kind):
    def plot(node_count, **options):
        graph = nx.path_graph(node_count, create_using=nx.DiGraph)
        nx.set_node_attributes(graph, {node: str(node) for node in graph}, "vo")
        if plot_kind == "graph":
            return plotting.plot_graph(graph, **options)[0]
        if plot_kind == "pathway":
            return plotting.plot_pathway(graph, plot_type="string", **options)[0]
        return plotting.plot_assembly_circle(
            list(graph), nx.to_numpy_array(graph), list(graph), **options
        )[0]

    small = plot(4, auto_fig_size=True)
    large = plot(40, auto_fig_size=True)
    fixed = plot(40, fig_size=9 if plot_kind == "circle" else (9, 4))

    assert np.all(large.get_size_inches() > small.get_size_inches())
    np.testing.assert_array_equal(
        fixed.get_size_inches(), [9, 9] if plot_kind == "circle" else [9, 4]
    )


@pytest.mark.parametrize("plot_type", ["mol", "graph", "atoms"])
def test_pathway_renders_one_icon_per_virtual_object(plot_type):
    pathway = nx.path_graph(3, create_using=nx.DiGraph)
    for node, smiles in enumerate(["CC", "CCC", "CCCCC"]):
        pathway.nodes[node]["vo"] = (
            smiles
            if plot_type == "atoms"
            else att.smi_to_nx(smiles, add_hydrogens=False)
        )

    fig, ax = plotting.plot_pathway(pathway, plot_type=plot_type)

    assert fig.axes == [ax]
    icons = [artist for artist in ax.artists if isinstance(artist, AnnotationBbox)]
    assert len(icons) == pathway.number_of_nodes()
    assert all(icon.offsetbox.get_data().size > 0 for icon in icons)
    assert not ax.axison


@pytest.mark.parametrize("show_icons", [False, True])
def test_string_pathway_labels_colors_mid_arrows_and_nonmutation(show_icons):
    graph = nx.DiGraph([(0, 1), (1, 2)])
    nx.set_node_attributes(graph, {0: "a", 1: "aa", 2: "aaaa"}, "vo")
    before = copy.deepcopy(graph)

    fig, ax = plotting.plot_pathway_mid_arrow(
        graph,
        plot_type="string",
        show_icons=show_icons,
        node_color="orange",
        arrow_color="purple",
        font_size=13,
        layout_style="crossmin",
        arrow_size=27,
    )

    assert fig.axes == [ax]
    assert nx.utils.graphs_equal(graph, before)
    assert not ax.axison
    expected_labels = ["a", "aa", "aaaa"] if show_icons else []
    assert [text.get_text() for text in ax.texts] == expected_labels
    assert all(text.get_fontsize() == 13 for text in ax.texts)
    np.testing.assert_allclose(
        ax.collections[0].get_facecolors(),
        [colors.to_rgba("white" if show_icons else "orange")],
    )
    arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    assert len(arrows) == graph.number_of_edges() * 3
    heads = arrows[-graph.number_of_edges() :]
    assert all(head.get_mutation_scale() == 27 for head in heads)
    assert all(head.get_edgecolor() == colors.to_rgba("purple") for head in heads)


@pytest.mark.parametrize(
    "plot,plot_type",
    [
        (plotting.plot_pathway, "mol"),
        (plotting.plot_pathway_mid_arrow, "mol"),
        (plotting.plot_pathway, "graph"),
    ],
)
def test_rust_dot_pathway_renders_icons_and_arrowheads(data_dir, plot, plot_type):
    molecule = att.molfile_to_mol(
        str(data_dir / "mol_files" / "anthracene.mol"), add_hydrogens=False
    )
    dot = (data_dir / "pathway" / "anthracene_pathway.dot").read_text()
    pathway = att.parse_pathway_dot(
        dot, mol=molecule, vo_type="graph" if plot_type == "graph" else "smiles"
    )

    fig, ax = plot(pathway, plot_type=plot_type)

    assert fig.axes == [ax]
    assert len(ax.artists) == pathway.number_of_nodes()
    arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    arrows_per_edge = 3 if plot is plotting.plot_pathway_mid_arrow else 2
    assert len(arrows) == arrows_per_edge * pathway.number_of_edges()


def test_circle_reuses_axes_preserves_data_and_hyperbolic_radii():
    fig, ax = plt.subplots(figsize=(5, 4))
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    before = adjacency.copy()
    indices = [0, 1, 2]

    result = plotting.plot_assembly_circle(
        ["a", "aa", "aaaa"],
        adjacency,
        indices,
        labels=["A", "", "C"],
        fig=fig,
        ax=ax,
        spacing_mode="hyperbolic",
        spacing_hyperbolic_factor=0.2,
        cmap="viridis",
        norm=colors.Normalize(0, 2),
        colorbar_label="Assembly index",
    )

    assert result == (fig, ax)
    np.testing.assert_array_equal(adjacency, before)
    assert indices == [0, 1, 2]
    expected_radii = np.arange(1, 4) + 0.2 * np.sinh(np.arange(1, 4))
    np.testing.assert_allclose(
        np.linalg.norm(ax.collections[0].get_offsets(), axis=1), expected_radii
    )
    np.testing.assert_allclose(
        [p.radius for p in ax.patches if isinstance(p, Circle)], expected_radii
    )
    assert [text.get_text() for text in ax.texts] == ["A", "C"]
    assert fig.axes[1].get_ylabel() == "Assembly index"


def test_assembly_circle_saves_png_and_renders_labels(tmp_path):
    nodes = ["a", "aa", "b", "aab"]
    adjacency = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    filename = tmp_path / "circle.png"

    fig, ax = plotting.plot_assembly_circle(
        nodes,
        adjacency,
        [0, 1, 0, 2],
        labels=nodes,
        node_color="skyblue",
        filename=filename,
        dpi=60,
    )

    assert [text.get_text() for text in ax.texts] == nodes
    assert fig.axes == [ax]
    with Image.open(filename) as image:
        assert image.format == "PNG"
        assert min(image.size) > 0
        image.verify()


def test_scatter_data_labels_sizes_and_alpha():
    x, y = [3, 1, 2], [6, 5, 4]

    fig, ax = plotting.scatter_plot(
        x, y, "Mass", "Index", figsize=(6, 4), fontsize=18, alpha=0.3
    )

    np.testing.assert_array_equal(
        ax.collections[0].get_offsets(), np.column_stack([x, y])
    )
    np.testing.assert_array_equal(ax.collections[0].get_sizes(), [50])
    assert ax.collections[0].get_alpha() == 0.3
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Mass", "Index")
    assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 18
    np.testing.assert_array_equal(fig.get_size_inches(), (6, 4))


def test_density_scatter_orders_points_by_density_without_mutating_input():
    x = np.array([0.0, 0.1, 0.2, 1.0, 2.0])
    y = np.array([0.0, 0.2, 0.1, 1.5, 0.5])
    original = np.column_stack([x, y])
    density = gaussian_kde(original.T)(original.T)

    fig, ax = plotting.scatter_plot_with_colorbar(x, y, cmap="plasma")

    points = ax.collections[0]
    np.testing.assert_allclose(points.get_offsets(), original[np.argsort(density)])
    np.testing.assert_allclose(points.get_array(), np.sort(density))
    np.testing.assert_array_equal(np.column_stack([x, y]), original)
    assert points.get_cmap().name == "plasma"
    assert fig.axes == [ax]


def test_heatmap_counts_orientation_and_colorbar():
    fig, ax = plotting.plot_heatmap(
        [50, 50, 150],
        [1, 4, 1],
        "x",
        "y",
        nbins=([0, 100, 200, 300], [0, 2, 6]),
        c_map="plasma",
    )

    image = ax.images[0]
    np.testing.assert_array_equal(image.get_array(), [[1, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(image.get_extent(), [0, 300, 0, 6])
    np.testing.assert_array_equal(ax.get_xlim(), [0, 300])
    np.testing.assert_array_equal(ax.get_ylim(), [0, 6])
    assert image.origin == "lower"
    assert image.get_cmap().name == "plasma"
    assert fig.axes[1].get_ylabel() == "Count"


def test_contour_uses_common_axis_range_and_requested_colormap():
    x = np.array([0.0, 0.1, 0.2, 1.0, 2.0])
    y = np.array([0.0, 0.2, 0.1, 1.5, 0.5])

    _, ax = plotting.plot_contourf_full(x, y, "Mass", "Index", c_map="plasma")

    np.testing.assert_array_equal(ax.get_xlim(), [0, 2])
    np.testing.assert_array_equal(ax.get_ylim(), [0, 2])
    assert ax.collections[0].get_cmap().name == "plasma"
    assert ax.collections[0].get_alpha() == 0.9
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Mass", "Index")


def test_3d_explicit_colors_preserve_point_order_and_options():
    fig, ax = plotting.scatter_plot_3d_with_colorbar(
        [3, 1, 2],
        [6, 5, 4],
        [2, 4, 1],
        c=[9, 3, 6],
        s=[20, 40, 60],
        xlab="A",
        ylab="B",
        zlab="C",
        cmap="plasma",
        alpha=0.4,
        labelpad=11,
    )

    scatter = ax.collections[0]
    np.testing.assert_array_equal(scatter._offsets3d, [[3, 1, 2], [6, 5, 4], [2, 4, 1]])
    np.testing.assert_array_equal(scatter.get_array(), [9, 3, 6])
    np.testing.assert_array_equal(scatter.get_sizes(), [20, 40, 60])
    assert scatter.get_alpha() == 0.4
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("A", "B", "C")
    assert ax.zaxis.labelpad == 11
    assert fig.axes[1].get_ylabel() == "Point Density"


@pytest.mark.parametrize("guide_line", [False, True])
def test_hexbin_counts_guide_and_axis_ranges(guide_line):
    fig, ax = plotting.plot_hexbin_scatter(
        [1, 2, 2, 4], [3, 5, 5, 7], guide_line=guide_line
    )

    assert ax.collections[0].get_array().sum() == 4
    np.testing.assert_array_equal(ax.get_xlim(), [1, 4])
    np.testing.assert_array_equal(ax.get_ylim(), [3, 7])
    assert len(ax.lines) == int(guide_line)
    if guide_line:
        np.testing.assert_array_equal(ax.lines[0].get_xydata(), [[1, 1], [4, 4]])
        assert ax.lines[0].get_linestyle() == "--"
    assert fig.axes[1].get_ylabel() == "counts"


def test_histogram_integer_bins_and_comparison_labels():
    _, ax = plotting.plot_histogram_all_x([1, 1, 3])
    assert [bar.get_x() for bar in ax.patches] == [1, 2, 3]
    assert [bar.get_height() for bar in ax.patches] == [2, 0, 1]

    _, compare = plotting.plot_histogram_compare(
        [0, 0, 2], [1, 2, 2], ["First", "Second"], bins=[0, 1, 2, 3]
    )
    assert [bar.get_height() for bar in compare.patches] == [2, 0, 1, 0, 1, 2]
    assert [text.get_text() for text in compare.get_legend().get_texts()] == [
        "First",
        "Second",
    ]
    assert compare.get_yscale() == "log"


def test_kde_reuses_axes_and_scales_density_to_grid_counts():
    values = np.array([0.0, 0.2, 1.0, 2.0, 3.0])
    fig, ax = plt.subplots()
    result = plotting.plot_kde(
        values, bandwidth=0.4, grid_size=31, y_scale=None, fig=fig, ax=ax
    )

    assert result == (fig, ax)
    x, counts = ax.lines[0].get_data()
    assert len(x) == 31
    np.testing.assert_allclose(
        counts, gaussian_kde(values, bw_method=0.4)(x) * len(values) * 0.1
    )
    np.testing.assert_array_equal(ax.get_xlim(), [0, 3])
    assert ax.get_yscale() == "linear"
    assert ax.lines[0].get_color() == "red"


@pytest.mark.parametrize("flip_x", [False, True])
def test_ir_spectrum_peak_data_shading_and_direction(flip_x):
    spectrum = np.array([[400, 1], [1000, 3], [2000, 2], [3000, 4]])
    before = spectrum.copy()

    _, ax = plotting.plot_ir_spectrum(
        spectrum,
        peaks=np.array([1, 3]),
        flip_x=flip_x,
        highlight_range=(500, 1400),
        xlab="Frequency",
        fontsize=18,
    )

    np.testing.assert_array_equal(ax.lines[0].get_xydata(), spectrum)
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), spectrum[[1, 3]])
    np.testing.assert_array_equal(spectrum, before)
    assert bool(ax.xaxis_inverted()) == flip_x
    assert ax.get_xlabel() == "Frequency"
    assert len(ax.patches) == 1
    assert ax.patches[0].get_alpha() == 0.5


def test_ms2_filters_parent_tolerance_and_sorts_peak_data(capsys):
    data = pd.DataFrame(
        {
            "parent": [100.0, 100.005, 100.02],
            "mz": [70, 30, 90],
            "intensity": [4, 8, 12],
        }
    )
    before = data.copy(deep=True)

    result = plotting.plot_ms2_spectrum(data, 100.0, {})

    assert result is None
    ax = plt.gca()
    np.testing.assert_array_equal(
        ax.collections[0].get_segments(), [[[30, 0], [30, 8]], [[70, 0], [70, 4]]]
    )
    np.testing.assert_array_equal(ax.get_xlim(), [0, 90])
    pd.testing.assert_frame_equal(data, before)
    assert "Plotting all 2 processed MS2 fragments" in capsys.readouterr().out


@pytest.mark.parametrize(
    "layout",
    [
        plotting.multipartite_layout_crossmin,
        plotting.multipartite_layout_crossmin_long,
        plotting.multipartite_layout_sa,
    ],
)
def test_layered_layout_removes_simple_crossing_and_preserves_graph(layout):
    graph = nx.Graph()
    graph.add_nodes_from(
        (node, {"rank": rank})
        for node, rank in [("a", 0), ("b", 0), ("c", 1), ("d", 1)]
    )
    graph.add_edges_from([("a", "d"), ("b", "c")])
    before = copy.deepcopy(graph)
    options = {
        "subset_key": "rank",
        "seed": 13,
        "align": "horizontal",
        "layer_spacing": 3,
        "node_spacing": 2,
        "scale": 2,
        "return_order": True,
    }

    positions, order = layout(graph, **options)

    assert (positions, order) == layout(graph, **options)
    assert nx.utils.graphs_equal(graph, before)
    assert set(order) == {0, 1}
    assert (positions["a"][0] - positions["b"][0]) * (
        positions["d"][0] - positions["c"][0]
    ) > 0
    assert positions["a"][1] == positions["b"][1] == 0
    assert positions["c"][1] == positions["d"][1] == 6
    assert abs(positions["a"][0] - positions["b"][0]) == 4


@pytest.mark.parametrize(
    "layout",
    [plotting.multipartite_layout_crossmin_long, plotting.multipartite_layout_sa],
)
@pytest.mark.parametrize(
    "return_order,return_routes",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_long_layout_metadata_and_reverse_edge_routes(
    layout, return_order, return_routes
):
    graph = nx.DiGraph()
    graph.add_nodes_from((node, {"subset": rank}) for rank, node in enumerate("abcd"))
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    before = copy.deepcopy(graph)
    options = {
        "seed": 7,
        "return_dummies": True,
        "dummy_prefix": "via_",
        "return_order": return_order,
        "return_routes": return_routes,
    }

    result = layout(graph, **options)

    if return_order or return_routes:
        assert (
            isinstance(result, tuple)
            and len(result) == 1 + return_order + return_routes
        )
        positions = result[0]
    else:
        assert isinstance(result, dict)
        positions = result
    assert len(positions) == 6
    assert nx.utils.graphs_equal(graph, before)
    if return_order:
        order = result[1]
        assert set(order) == {0, 1, 2, 3}
        assert {node for layer in order.values() for node in layer} == set(positions)
    if return_routes:
        routes = result[-1]
        long_route = next(route for route in routes if route["endpoints"] == ("d", "a"))
        assert long_route["nodes"] == ["d", "via_2", "via_1", "a"]
        assert long_route["points"] == [positions[node] for node in long_route["nodes"]]
    without_dummies = layout(graph, seed=7, dummy_prefix="via_")
    assert without_dummies == {node: positions[node] for node in graph}


def test_weighted_crossings_exclude_shared_endpoints():
    edges = [("a", "d", 2), ("b", "c", 3), ("a", "c", 5)]
    assert plotting._pair_crossings_weighted(["a", "b"], ["c", "d"], edges) == 6


def test_molecule_grid_converts_invalid_smiles_and_limits_before_legends(monkeypatch):
    captured = {}
    sentinel = object()

    def draw(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(plotting.Draw, "MolsToGridImage", draw)
    molecule = Chem.MolFromSmiles("CO")
    result = plotting.draw_mol_grid(
        [molecule, "invalid", "CC"],
        legends=["A", "B"],
        max_mols=2,
        n_cols=2,
        sub_img_size=(80, 60),
        use_svg=True,
    )

    assert result is sentinel
    assert captured["mols"][0] is molecule
    assert captured["mols"][1].GetNumAtoms() == 0
    assert captured["legends"] == ["A", "B"]
    assert captured["molsPerRow"] == 2 and captured["subImgSize"] == (80, 60)
    assert captured["useSVG"] is True


def test_box_grid_sorts_legends_preserves_inputs_and_tile_spacing(monkeypatch):
    rendered = []

    def draw(mol, size, legend):
        rendered.append((Chem.MolToSmiles(mol), size, legend))
        return Image.new("RGB", size, "red")

    monkeypatch.setattr(plotting.Draw, "MolToImage", draw)
    molecules = ["CC", "CO", "CN"]
    legends = ["ethane", "methanol", "methylamine"]
    image = plotting.draw_mol_grid_box(
        molecules,
        legends,
        sort_by=[3, 1, 2],
        n_cols=2,
        sub_img_size=(30, 20),
        gap=3,
        outer_margin=4,
        inner_pad=2,
        box_bg="blue",
    )

    assert molecules == ["CC", "CO", "CN"]
    assert legends == ["ethane", "methanol", "methylamine"]
    assert rendered == [
        ("CO", (26, 16), "methanol"),
        ("CN", (26, 16), "methylamine"),
        ("CC", (26, 16), "ethane"),
    ]
    assert image.size == (71, 51)
    assert image.getpixel((4, 4)) == (0, 0, 255)
    assert image.getpixel((6, 6)) == (255, 0, 0)
    assert image.getpixel((35, 4)) == image.getpixel((40, 30)) == (255, 255, 255)


@pytest.mark.parametrize("grid", [plotting.draw_mol_grid, plotting.draw_mol_grid_box])
def test_molecule_grid_validation(grid):
    with pytest.raises(ValueError, match="n_cols"):
        grid(["CO"], n_cols=0)
    with pytest.raises(TypeError, match="Item 1"):
        grid(["CO", 123])
    with pytest.raises(ValueError, match="legends"):
        grid(["CO"], legends=["A", "B"])


def test_empty_box_grid_returns_blank_margin_sized_image():
    image = plotting.draw_mol_grid_box([], outer_margin=3)
    assert image.size == (7, 7)
    assert image.getextrema() == ((255, 255),) * 3


@pytest.mark.parametrize(
    "smiles,overlap", [(("CCO", "CCCO"), True), (("C", "O"), False)]
)
def test_common_bond_highlights_and_no_overlap_image(smiles, overlap, monkeypatch):
    captured = {}
    sentinel = object()

    def draw(mols, **kwargs):
        captured.update(kwargs)
        captured["mols"] = mols
        return sentinel

    monkeypatch.setattr(plotting.Draw, "MolsToGridImage", draw)
    result = plotting.show_common_bonds(
        *smiles,
        legends=["First", "Second"],
        size=(400, 180),
        common_atom_color=(1.0, 0.0, 0.0),
        common_bond_color=(0.0, 0.0, 1.0),
    )

    assert result is sentinel
    assert captured["molsPerRow"] == 2
    assert captured["subImgSize"] == (200, 180)
    assert captured["legends"] == ["First", "Second"]
    assert [Chem.MolToSmiles(mol) for mol in captured["mols"]] == list(smiles)
    if overlap:
        assert [len(indices) for indices in captured["highlightAtomLists"]] == [3, 3]
        assert [len(indices) for indices in captured["highlightBondLists"]] == [2, 2]
        for atom_colors, bond_colors in zip(
            captured["highlightAtomColors"], captured["highlightBondColors"]
        ):
            assert set(atom_colors.values()) == {(1.0, 0.0, 0.0)}
            assert set(bond_colors.values()) == {(0.0, 0.0, 1.0)}
    else:
        assert "highlightAtomLists" not in captured
        assert "highlightBondLists" not in captured


@pytest.mark.parametrize(
    "renderer", [plotting.draw_mol_grid, plotting.show_common_bonds]
)
def test_molecular_images_render_real_structures(renderer):
    smiles = ["CCO", "CCCO"]
    image = (
        renderer(smiles, legends=smiles)
        if renderer is plotting.draw_mol_grid
        else renderer(*smiles, legends=smiles)
    )

    assert isinstance(image, Image.Image)
    assert min(image.size) > 0
    assert image.convert("L").getextrema()[0] < 255


def test_ase_plot_forwards_view_and_save_options(monkeypatch, tmp_path):
    atoms = object()
    viewed = []
    saved = []
    monkeypatch.setattr(
        plotting, "plot_atoms", lambda *args, **kwargs: viewed.append((args, kwargs))
    )
    monkeypatch.setattr(
        Figure, "savefig", lambda *args, **kwargs: saved.append((args, kwargs))
    )
    outfile = tmp_path / "atoms.png"

    fig, ax = plotting.plot_ase_atoms(
        atoms,
        outfile,
        rotation="90x,20y,0z",
        show_unit_cell=2,
        fig_size=(7, 4),
        dpi=120,
        transparent=True,
    )

    assert viewed == [
        ((atoms,), {"ax": ax, "rotation": "90x,20y,0z", "show_unit_cell": 2})
    ]
    assert saved == [
        (
            (fig, outfile),
            {"dpi": 120, "transparent": True, "bbox_inches": "tight", "pad_inches": 0},
        )
    ]
    np.testing.assert_array_equal(fig.get_size_inches(), [7, 4])
    assert not ax.axison


def test_ase_plot_renders_all_atoms():
    atoms = att.smiles_to_atoms("c1ccccc1")

    fig, ax = plotting.plot_ase_atoms(atoms)

    assert fig.axes == [ax]
    assert len(ax.patches) == len(atoms)
    assert not ax.axison
