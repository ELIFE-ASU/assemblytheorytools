"""Display-space layout and obstacle-aware edges for pathway figures.

The layout artist runs before the edges on every draw. Measuring in the active
renderer keeps image boxes, text, arrowheads and their clearance in agreement
when a figure is resized or saved at a different resolution.
"""

from collections import Counter, defaultdict
from heapq import heappop, heappush

import numpy as np
from matplotlib.artist import Artist
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import Bbox, IdentityTransform


def _hits_box(start, end, box):
    """Whether an open segment enters the interior of a rectangle."""
    low, high = 0.0, 1.0
    for a, b, lower, upper in zip(start, end, box.p0, box.p1):
        delta = b - a
        if abs(delta) < 1e-9:
            if a <= lower + 1e-7 or a >= upper - 1e-7:
                return False
        else:
            t0, t1 = sorted(((lower - a) / delta, (upper - a) / delta))
            low, high = max(low, t0), min(high, t1)
    return high - low > 1e-7 and high > 1e-7 and low < 1 - 1e-7


def _visible_route(start, end, boxes):
    """Shortest polyline around rectangles; direct segments take the fast path."""
    if not any(_hits_box(start, end, box) for box in boxes):
        return [start, end]
    vertices = [start, end]
    for box in boxes:
        vertices.extend(
            [
                box.p0,
                np.array([box.x0, box.y1]),
                box.p1,
                np.array([box.x1, box.y0]),
            ]
        )
    distances = {0: 0.0}
    previous = {}
    queue = [(0.0, 0)]
    while queue:
        distance, i = heappop(queue)
        if distance != distances[i]:
            continue
        if i == 1:
            chain = [1]
            while chain[-1] != 0:
                chain.append(previous[chain[-1]])
            return [vertices[j] for j in reversed(chain)]
        for j, point in enumerate(vertices):
            if i == j:
                continue
            cost = distance + np.linalg.norm(point - vertices[i])
            if cost >= distances.get(j, float("inf")):
                continue
            if any(_hits_box(vertices[i], point, box) for box in boxes):
                continue
            distances[j] = cost
            previous[j] = i
            heappush(queue, (cost, j))
    # Non-overlapping, fitted node boxes always admit a visibility route.
    raise RuntimeError("Unable to route a pathway edge around its node boxes")


def _box_port(center, toward, box):
    """Intersect a ray from a node center with its padded rectangle."""
    delta = np.asarray(toward) - center
    factors = []
    for axis, size in enumerate((box.width, box.height)):
        if abs(delta[axis]) > 1e-9:
            factors.append(size / (2 * abs(delta[axis])))
    return center + min(factors, default=0) * delta


def _rounded_path(points, radius, boxes=()):
    """Use broad, tangent-continuous bends, tightening only near node boxes."""
    # A long straight lane may contain a waypoint at every intermediate layer.
    # Removing these lets the adjacent bend use the whole available run.
    simplified = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if simplified and np.linalg.norm(point - simplified[-1]) < 1e-9:
            continue
        while len(simplified) >= 2:
            incoming = simplified[-1] - simplified[-2]
            outgoing = point - simplified[-1]
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            if (
                abs(cross) > 1e-9 * np.linalg.norm(incoming) * np.linalg.norm(outgoing)
                or np.dot(incoming, outgoing) <= 0
            ):
                break
            simplified.pop()
        simplified.append(point)
    points = simplified
    vertices = [points[0]]
    codes = [Path.MOVETO]
    for before, corner, after in zip(points, points[1:], points[2:]):
        incoming, outgoing = corner - before, after - corner
        length_in, length_out = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(length_in, length_out) < 1e-9:
            continue
        trim = min(radius, length_in / 2, length_out / 2)
        # Wide fillets make open lanes flow naturally. At a crowded corner,
        # use the largest local fillet which retains the route's clearance.
        for _ in range(16):
            curve = Path(
                [
                    corner - incoming * trim / length_in,
                    corner,
                    corner + outgoing * trim / length_out,
                ],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )
            if not any(curve.intersects_bbox(box, filled=False) for box in boxes):
                break
            trim *= 0.5
        vertices.extend(
            [
                corner - incoming * trim / length_in,
                corner,
                corner + outgoing * trim / length_out,
            ]
        )
        codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])
    vertices.append(points[-1])
    codes.append(Path.LINETO)
    return Path(vertices, codes)


def _point_on_path(path, fraction):
    """Return a point and tangent at a fraction of the visible arc length."""
    points = np.asarray([p for p, _ in path.iter_segments(curves=False)])
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    keep = lengths > 1e-9
    starts, steps, lengths = points[:-1][keep], steps[keep], lengths[keep]
    arc = np.r_[0, np.cumsum(lengths)]
    target = np.clip(fraction, 0, 1) * arc[-1]
    i = min(np.searchsorted(arc, target, side="right") - 1, len(lengths) - 1)
    tangent = steps[i] / lengths[i]
    return starts[i] + tangent * (target - arc[i]), tangent


class _Route:
    """A connection style whose path is refreshed by the layout artist."""

    def __init__(self):
        self.path = Path([(0, 0), (1, 1)])

    def __call__(self, posA, posB, **kwargs):
        return self.path


class _PathwayArtist(Artist):
    """Fit measured nodes and route their edges before Matplotlib draws them."""

    def __init__(
        self,
        ax,
        positions,
        routes,
        nodes,
        collection,
        *,
        font_size,
        arrow_color,
        arrow_style,
        arrow_size,
        arrow_pos,
    ):
        super().__init__()
        self.set_zorder(0)
        self.set_in_layout(False)
        self.positions = positions
        self.routes = routes
        self.nodes = nodes
        self.collection = collection
        self.font_size = font_size
        self.arrow_size = arrow_size
        self.arrow_pos = arrow_pos
        self._measure_only = False
        style = ArrowStyle(arrow_style) if isinstance(arrow_style, str) else arrow_style
        # Filled arrow styles only accept a single quadratic Bezier. Give
        # them a short local head, leaving the routed shaft independent.
        self.filled_head = isinstance(
            style, (ArrowStyle.Simple, ArrowStyle.Fancy, ArrowStyle.Wedge)
        )
        separate_head = arrow_pos < 1 or self.filled_head
        self.head_length = max(0.6, getattr(style, "head_length", 0.6)) * arrow_size
        self.head_width = (
            max(
                0.2,
                getattr(style, "head_width", 0.2),
                getattr(style, "tail_width", 0.2),
            )
            * arrow_size
        )
        self.zooms = {
            node: artist.offsetbox.get_zoom()
            for node, artist in nodes.items()
            if isinstance(artist, AnnotationBbox)
        }
        self.wrapped_labels = {}
        self.connections, self.edges, self.heads = [], [], []
        for _ in routes:
            connection = _Route()
            edge = FancyArrowPatch(
                (0, 0),
                (1, 1),
                connectionstyle=connection,
                transform=IdentityTransform(),
                arrowstyle="-" if separate_head else arrow_style,
                mutation_scale=arrow_size,
                color=arrow_color,
                linewidth=1.5,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
                capstyle="round",
                joinstyle="round",
            )
            ax.add_patch(edge)
            self.connections.append(connection)
            self.edges.append(edge)
        if separate_head:
            for _ in routes:
                head = FancyArrowPatch(
                    (0, 0),
                    (1, 1),
                    transform=IdentityTransform(),
                    arrowstyle=arrow_style,
                    mutation_scale=arrow_size,
                    color=arrow_color,
                    linewidth=1.5,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=1,
                    capstyle="round",
                    joinstyle="round",
                )
                ax.add_patch(head)
                self.heads.append(head)

    def _measure(self, renderer, scale):
        sizes = {}
        for node, artist in self.nodes.items():
            if isinstance(artist, AnnotationBbox):
                artist.offsetbox.set_zoom(self.zooms[node] * scale)
                artist.set_fontsize(10 * scale)
                artist.update_positions(renderer)
                box = artist.get_window_extent(renderer)
            elif artist is not None:
                artist.set_fontsize(self.font_size * scale)
                artist.update_bbox_position_size(renderer)
                box = artist.get_bbox_patch().get_window_extent(renderer)
            else:
                diameter = renderer.points_to_pixels(np.sqrt(1000) * scale)
                box = Bbox.from_bounds(0, 0, diameter, diameter)
            sizes[node] = np.array([box.width, box.height])
        return sizes

    def _layout_size(
        self, sizes, layers, span, parallel_count, route_padding, gap, pixel, scale
    ):
        """Measure the canvas demand, including dummy lanes and arrow clearance."""
        widths = {
            layer: max(
                (sizes[n][0] for n in self.nodes if self.positions[n][0] == layer),
                default=0,
            )
            for layer in layers
        }
        tallest = max(size[1] for size in sizes.values())
        pitch = tallest + 2 * route_padding * scale
        width = sum(widths.values()) + gap * scale * (len(layers) - 1)
        height = span * pitch + tallest + (parallel_count - 1) * 18 * pixel * scale
        return widths, pitch, width, height

    def autosize_figure(self):
        """Grow once at native node size; subsequent resizes remain user-controlled."""
        if not self.positions:
            return
        fig = self.axes.figure
        # Use the active backend's renderer without allocating a large raster or
        # routing edges twice. Never resize within draw: savefig may temporarily
        # change the DPI and canvas bounds, especially with bbox_inches='tight'.
        self._measure_only = True
        try:
            fig.draw_without_rendering()
        finally:
            self._measure_only = False
        fig.set_size_inches(np.maximum(fig.get_size_inches(), self._required_inches))

    def _fit_nodes(self, renderer, layers, span, parallel_count, route_padding, gap):
        """Fit actual font metrics, which do not scale linearly at small sizes."""
        for node, (original, wrapped) in self.wrapped_labels.items():
            if self.nodes[node].get_text() == wrapped:
                self.nodes[node].set_text(original)
        self.wrapped_labels.clear()
        pixel = renderer.points_to_pixels(1)
        available = np.maximum(1, self.axes.bbox.size - 36 * pixel)
        scale = 1.0
        for attempt in range(16):
            sizes = self._measure(renderer, scale)
            widths, pitch, width, height = self._layout_size(
                sizes, layers, span, parallel_count, route_padding, gap, pixel, scale
            )
            shrink = min(1, *(available / np.maximum(1, [width, height])))
            if shrink >= 0.999 or attempt == 15:
                return sizes, widths, pitch, width, scale
            scale *= shrink * 0.98
            if self.font_size * scale < 1:
                # Matplotlib clamps text at 1pt. Wrap unusually long labels
                # rather than continuing to assume they can become narrower.
                target = max(pixel * 3, available[0] / len(layers) - gap * scale)
                for node, artist in self.nodes.items():
                    if artist is None or isinstance(artist, AnnotationBbox):
                        continue
                    if sizes[node][0] <= target:
                        continue
                    original = self.wrapped_labels.get(node, (artist.get_text(),))[0]
                    lines = artist.get_text().split("\n")
                    chunk = max(
                        1, int(max(map(len, lines)) * target / sizes[node][0] * 0.9)
                    )
                    wrapped = "\n".join(
                        line[i : i + chunk]
                        for line in original.split("\n")
                        for i in range(0, max(1, len(line)), chunk)
                    )
                    self.wrapped_labels[node] = (original, wrapped)
                    artist.set_text(wrapped)

    def draw(self, renderer):
        if not self.positions:
            return
        ax = self.axes
        pixel = renderer.points_to_pixels(1)
        layers = sorted({xy[0] for xy in self.positions.values()})
        head_extent = self.head_length + self.head_width
        # A separate head needs its full extent at the source. Reserve the
        # same space between rows so a steep source port cannot enter the
        # padded obstacle belonging to the next node in its layer.
        route_padding = (
            max(12, head_extent if self.arrow_pos < 1 else head_extent - 4) * pixel
        )
        gap = max(48, head_extent * 2.5) * pixel
        free = np.array([xy[1] for xy in self.positions.values()])
        # Normalize the free-axis spacing, including NetworkX's rescaled layouts.
        separations = []
        for layer in layers:
            ys = sorted({xy[1] for xy in self.positions.values() if xy[0] == layer})
            separations.extend(np.diff(ys))
        unit = min(separations, default=1)
        span = np.ptp(free) / unit
        multiplicity = Counter(route["endpoints"] for route in self.routes)
        parallel_count = max(
            (
                multiplicity[route["endpoints"]]
                for route in self.routes
                if len(route["nodes"]) == 2
            ),
            default=1,
        )
        if self._measure_only:
            _, _, width, height = self._layout_size(
                self._measure(renderer, 1),
                layers,
                span,
                parallel_count,
                route_padding,
                gap,
                pixel,
                1,
            )
            # Keep the same margin as _fit_nodes and a little extra room for
            # font metric differences between interactive and vector backends.
            required = np.array([width, height]) * 1.02 + 36 * pixel
            self._required_inches = required / (72 * pixel * ax.get_position().size)
            return
        sizes, widths, pitch, width, scale = self._fit_nodes(
            renderer, layers, span, parallel_count, route_padding, gap
        )
        x = ax.bbox.x0 + (ax.bbox.width - width) / 2
        columns = {}
        for layer in layers:
            columns[layer] = x + widths[layer] / 2
            x += widths[layer] + gap * scale
        center_y = (ax.bbox.y0 + ax.bbox.y1) / 2
        mid = (free.min() + free.max()) / 2
        centers = {
            node: np.array([columns[xy[0]], center_y + (xy[1] - mid) / unit * pitch])
            for node, xy in self.positions.items()
        }
        inverse = ax.transData.inverted()
        boxes = {}
        for node, artist in self.nodes.items():
            center = centers[node]
            position = inverse.transform(center)
            if isinstance(artist, AnnotationBbox):
                artist.xy = artist.xybox = position
                artist.update_positions(renderer)
                boxes[node] = artist.get_window_extent(renderer)
                artist.patch.set_linewidth(0.75 * scale)
            elif artist is not None:
                artist.set_position(position)
                artist.update_bbox_position_size(renderer)
                boxes[node] = artist.get_bbox_patch().get_window_extent(renderer)
                artist.get_bbox_patch().set_linewidth(0.75 * scale)
            else:
                boxes[node] = Bbox.from_bounds(
                    *(center - sizes[node] / 2), *sizes[node]
                )
        if self.collection is not None:
            self.collection.set_offsets(
                [inverse.transform(centers[n]) for n in self.nodes]
            )
            self.collection.set_sizes([1000 * scale**2])

        seen = defaultdict(int)
        clearance = max(5, self.arrow_size * 0.3) * pixel * scale
        for index, (route, connection, edge) in enumerate(
            zip(self.routes, self.connections, self.edges)
        ):
            source, target = route["endpoints"]
            points = [centers[node] for node in route["nodes"]]
            pair = (source, target)
            if len(points) == 2 and multiplicity[pair] > 1:
                lane = seen[pair] - (multiplicity[pair] - 1) / 2
                middle = (points[0] + points[-1]) / 2
                middle = middle + np.array([0, lane * 18 * pixel * scale])
                points.insert(1, middle)
            seen[pair] += 1
            # A midpoint head extends backwards from its tip. Reserve its full
            # length at the source even when arrow_pos is zero or close to it.
            source_clearance = (
                max(clearance, head_extent * pixel * scale)
                if self.arrow_pos < 1
                else clearance
            )
            points[0] = _box_port(
                centers[source], points[1], boxes[source].padded(source_clearance)
            )
            points[-1] = _box_port(
                centers[target], points[-2], boxes[target].padded(clearance)
            )
            obstacles = [
                box.padded(route_padding * scale)
                for node, box in boxes.items()
                if node not in pair
            ]
            routed = [points[0]]
            for start, end in zip(points, points[1:]):
                routed.extend(_visible_route(start, end, obstacles)[1:])
            smoothing_boxes = [
                box.padded(
                    (1.5 * pixel if node in pair else route_padding * 0.65) * scale
                )
                for node, box in boxes.items()
            ]
            connection.path = _rounded_path(routed, 72 * pixel * scale, smoothing_boxes)
            edge.set_mutation_scale(self.arrow_size * scale)
            edge.set_linewidth(1.5 * scale)
            if self.heads:
                point, tangent = _point_on_path(connection.path, self.arrow_pos)
                head = self.heads[index]
                length = self.head_length * pixel * scale if self.filled_head else 0.01
                head.set_positions(point - tangent * length, point)
                head.set_mutation_scale(self.arrow_size * scale)
                head.set_linewidth(1.5 * scale)
        self.stale = False
