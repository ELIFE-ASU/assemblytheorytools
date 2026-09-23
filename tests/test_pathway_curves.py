"""Long routed arrows retain broad bends while clearing nearby node cards."""

import numpy as np
from matplotlib.transforms import Bbox

from assemblytheorytools._pathway_plotting import _rounded_path


def _sample_path(path):
    return np.concatenate(
        [
            curve(np.linspace(0, 1, 201))
            for curve, _ in path.iter_bezier()
            if curve.degree > 0
        ]
    )


def _distance_from_corner(path, corner):
    return np.linalg.norm(_sample_path(path) - corner, axis=1).min()


def test_open_bends_are_broad_and_join_without_sharp_changes_in_direction():
    points = np.array([[0, 0], [100, 0], [100, 100], [200, 100]], dtype=float)
    path = _rounded_path(points, radius=72)

    np.testing.assert_allclose(path.vertices[[0, -1]], points[[0, -1]])
    # Tiny corner fillets still approach the original right-angle vertices.
    # Broad curves must turn well before either corner.
    for corner in points[1:-1]:
        assert _distance_from_corner(path, corner) > 15

    segments = [
        curve.control_points
        for curve, _ in path.iter_bezier()
        if curve.degree > 0
        and np.linalg.norm(curve.control_points[-1] - curve.control_points[0]) > 1e-9
    ]
    for before, after in zip(segments, segments[1:]):
        np.testing.assert_allclose(before[-1], after[0])
        incoming = before[-1] - before[-2]
        outgoing = after[1] - after[0]
        cosine = np.dot(incoming, outgoing) / (
            np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        )
        assert cosine > 1 - 1e-10


def test_obstacle_tightens_only_the_affected_bend():
    points = np.array([[0, 0], [100, 0], [100, 100], [200, 100]], dtype=float)
    card = Bbox.from_extents(75, 8, 92, 22)
    open_path = _rounded_path(points, radius=72)
    protected_path = _rounded_path(points, radius=72, boxes=[card])

    assert open_path.intersects_bbox(card, filled=False)
    assert not protected_path.intersects_bbox(card, filled=False)
    assert _distance_from_corner(protected_path, points[1]) < 10
    assert _distance_from_corner(protected_path, points[2]) > 15
    np.testing.assert_allclose(protected_path.vertices[[0, -1]], points[[0, -1]])


def test_collinear_layer_waypoints_do_not_limit_the_curve_size():
    points = np.array(
        [
            [0, 0],
            [25, 0],
            [50, 0],
            [75, 0],
            [100, 0],
            [100, 25],
            [100, 50],
            [100, 75],
            [100, 100],
        ],
        dtype=float,
    )
    path = _rounded_path(points, radius=72)

    np.testing.assert_allclose(path.vertices[[0, -1]], points[[0, -1]])
    assert _distance_from_corner(path, np.array([100, 0])) > 15
