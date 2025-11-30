import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as _plt
from matplotlib.patches import Circle as _MplCircle
from matplotlib.lines import Line2D as _MplLine


class Line:
    """Represents a detected line segment and stores matched edge points.

    Attributes
    - ref: tuple (x0, y0, x1, y1)
    - pts: (M,2) numpy array of matched (x,y) points
    - pts_idx: (M,) integer indices into the provided edge_points array
    - params: dict of tunable thresholds (e.g. max_perp_dist, require_on_segment)
    """

    def __init__(self, ref: Tuple[float, float, float, float], **params):
        self.ref = tuple(map(float, ref))
        self.pts: np.ndarray = np.zeros((0, 2), dtype=float)
        self.pts_idx: np.ndarray = np.zeros((0,), dtype=int)
        # TODO: TUNE THESE PARAMETERS
        # max_perp_dist: max distance from line segment to edge point to be considered a match
        # require_on_segment: if True, only consider points that project onto the segment
        self.params = {
            "max_perp_dist": 3.0,
            "require_on_segment": True,
        }
        self.params.update(params)

        x0, y0, x1, y1 = self.ref
        self._p0 = np.array([x0, y0], dtype=float)
        self._p1 = np.array([x1, y1], dtype=float)
        self._v = self._p1 - self._p0
        self._len2 = float(np.dot(self._v, self._v)) if np.any(self._v) else 1.0

    def match_pts(self, edge_points: np.ndarray, **override_params) -> Tuple[np.ndarray, np.ndarray]:
        """Match `edge_points` (Nx2) to this line segment.

        Parameters are tunable via class params or override_params.
        Returns (matched_pts (M,2), matched_idx (M,)).
        """
        params = dict(self.params)
        params.update(override_params)

        pts = np.asarray(edge_points)
        if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
            pts = pts.T
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("edge_points must be shape (N,2) or (2,N)")

        w = pts - self._p0  # (N,2)
        v = self._v
        t = np.dot(w, v) / self._len2  # (N,)
        proj = self._p0 + np.outer(t, v)
        perp_vec = pts - proj
        perp_dist = np.linalg.norm(perp_vec, axis=1)

        mask = perp_dist <= float(params["max_perp_dist"])
        if bool(params["require_on_segment"]):
            mask = mask & (t >= 0.0) & (t <= 1.0)

        matched_idx = np.nonzero(mask)[0]
        matched_pts = pts[matched_idx] if matched_idx.size else np.zeros((0, 2), dtype=float)

        self.pts = matched_pts
        self.pts_idx = matched_idx
        return matched_pts, matched_idx

    def calc_dims(self) -> None:
        """Compute basic geometric properties: length and direction."""
        v = self._v
        self.length = float(np.linalg.norm(v))
        if self.length > 0:
            self.direction = (v / self.length).tolist()
        else:
            self.direction = [0.0, 0.0]

    def calc_tols(self) -> None:
        """Compute simple straightness: RMS perpendicular distance of matched points."""
        if self.pts.size == 0:
            self.straightness = None
            return
        # recompute perp distance for current pts
        w = self.pts - self._p0
        v = self._v
        t = np.dot(w, v) / self._len2
        proj = self._p0 + np.outer(t, v)
        perp = np.linalg.norm(self.pts - proj, axis=1)
        self.straightness = float(np.sqrt(np.mean(perp ** 2)))


class Circle:
    """Represents a detected circle and stores matched edge points.

    Attributes
    - ref: tuple (xc, yc, r)
    - pts, pts_idx: matched points and indices
    - params: tunable thresholds (e.g. max_rad_diff)
    """

    # TODO: TUNE THESE PARAMETERS
    # max_rad_diff: max difference in radius from circle to edge point to be considered a match
    # (others can be added later)
    def __init__(self, ref: Tuple[float, float, float], **params):
        self.ref = tuple(map(float, ref))
        self.pts: np.ndarray = np.zeros((0, 2), dtype=float)
        self.pts_idx: np.ndarray = np.zeros((0,), dtype=int)
        self.params = {"max_rad_diff": 3.0}
        self.params.update(params)

        xc, yc, r = self.ref
        self.center = np.array([xc, yc], dtype=float)
        self.radius = float(r)

    def match_pts(self, edge_points: np.ndarray, **override_params) -> Tuple[np.ndarray, np.ndarray]:
        """Match `edge_points` (Nx2) to this circle.

        Returns (matched_pts, matched_idx).
        """
        params = dict(self.params)
        params.update(override_params)

        pts = np.asarray(edge_points)
        if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
            pts = pts.T
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("edge_points must be shape (N,2) or (2,N)")

        dists = np.linalg.norm(pts - self.center, axis=1)
        diff = np.abs(dists - self.radius)
        mask = diff <= float(params["max_rad_diff"])

        matched_idx = np.nonzero(mask)[0]
        matched_pts = pts[matched_idx] if matched_idx.size else np.zeros((0, 2), dtype=float)

        self.pts = matched_pts
        self.pts_idx = matched_idx
        return matched_pts, matched_idx

    def calc_dims(self) -> None:
        """Compute mean radius from matched points and store radial residuals."""
        if self.pts.size == 0:
            self.mean_radius = None
            self.radial_std = None
            return
        dists = np.linalg.norm(self.pts - self.center, axis=1)
        self.mean_radius = float(np.mean(dists))
        self.radial_std = float(np.std(dists))

    def calc_tols(self) -> None:
        """Alias for now to compute radial spread"""
        self.calc_dims()


def match_points_to_lines(detected_lines: List[Tuple[float, float, float, float]],
                          edge_points: np.ndarray,
                          **match_params) -> List[Line]:
    """Create Line objects from detected_lines and match edge_points to each.

    `match_params` are passed to each Line.match_pts call (can include thresholds).
    """
    # allow caller to require a minimum line length (pixels)
    min_length = float(match_params.pop("min_length", 20.0))
    features: List[Line] = []
    print(f"[match_points_to_lines] detected_lines: {len(detected_lines)}, min_length: {min_length}")
    for ln in detected_lines:
        x0, y0, x1, y1 = map(float, ln)
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < min_length:
            # skip small lines
            continue
        L = Line((x0, y0, x1, y1))
        matched_pts, matched_idx = L.match_pts(edge_points, **match_params)
        print(f"  line ref=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) len={length:.1f} matched={matched_idx.size}")
        features.append(L)
    print(f"[match_points_to_lines] returning features: {len(features)}")
    return features


def match_points_to_circles(detected_circles: List[Tuple[float, float, float]],
                            edge_points: np.ndarray,
                            **match_params) -> List[Circle]:
    # allow caller to require a minimum radius (pixels)
    min_radius = float(match_params.pop("min_radius", 20.0))
    features: List[Circle] = []
    print(f"[match_points_to_circles] detected_circles: {len(detected_circles)}, min_radius: {min_radius}")
    for c in detected_circles:
        xc, yc, r = map(float, c)
        if r < min_radius:
            continue
        C = Circle((xc, yc, r))
        matched_pts, matched_idx = C.match_pts(edge_points, **match_params)
        print(f"  circle ref=({xc:.1f},{yc:.1f},r={r:.1f}) matched={matched_idx.size}")
        features.append(C)
    print(f"[match_points_to_circles] returning features: {len(features)}")
    return features



def visualize_matches(image: np.ndarray,
                      features: List[object],
                      figsize: Tuple[int, int] = (8, 8),
                      title: Optional[str] = None,
                      show: bool = True,
                      show_circle_points: bool = False,
                      show_line_points: bool = True):
    """Visualize matched points on an image using matplotlib.
    - `image` is an HxWxC image (RGB or grayscale). It will be shown as-is.
    - `features` is a list of `Line` and/or `Circle` objects (instances defined above).
    The function plots the image, draws each primitive (line segment or circle), and
    scatters matched edge points for each feature.
    Returns the `(fig, ax)` tuple so callers can further customize the plot.
    """
    fig, ax = _plt.subplots(figsize=figsize)
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    if title:
        ax.set_title(title)

    # color cycle for features
    colors = ["lime", "cyan", "magenta", "yellow", "orange", "red"]
    ci = 0

    for f in features:
        # debug-print each feature's ref
        try:
            ref = getattr(f, "ref", None)
            print(f"[visualize_matches] drawing feature ref={ref}")
        except Exception:
            pass
        col = colors[ci % len(colors)]
        ci += 1
        if isinstance(f, Line):
            x0, y0, x1, y1 = f.ref
            # draw the segment
            ln = _MplLine([x0, x1], [y0, y1], color=col, linewidth=2)
            ax.add_line(ln)
            # draw matched points
            if show_line_points and getattr(f, "pts", None) is not None and f.pts.size:
                pts = np.asarray(f.pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=20, c=col, marker="o", edgecolors="k")
        elif isinstance(f, Circle):
            xc, yc = f.center
            r = f.radius
            circ = _MplCircle((xc, yc), r, edgecolor=col, facecolor="none", linewidth=2)
            ax.add_patch(circ)
            if show_circle_points and getattr(f, "pts", None) is not None and f.pts.size:
                pts = np.asarray(f.pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=20, c=col, marker="x", edgecolors="k")
        else:
            # unknown feature type: try to draw pts if present
            if hasattr(f, "pts") and getattr(f, "pts") is not None and f.pts.size:
                pts = np.asarray(f.pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="white")

    ax.set_axis_off()
    _plt.tight_layout()
    if show:
        _plt.show()
    return fig, ax


__all__ = ["Line", "Circle", "match_points_to_lines", "match_points_to_circles", "visualize_matches"]

# Helper function to extract edge points from an image
def edge_points_from_image(image: np.ndarray):
    """Return `edge_points` (Nx2) from an image using preprocessing.

    Tries `preprocessing.get_edges(image)` first . If unavailable,
    falls back to `preprocessing.getEdgesMasked(image)[0]`, then to
    `preprocessing.gradient(image)[0]`.
    """
    try:
        import preprocessing as pp
    except Exception:
        raise ImportError("preprocessing module not available")

    edges = None
    # Prefer if present
    if hasattr(pp, "get_edges"):
        try:
            edges,_,_ = pp.get_edges(image)
        except Exception:
            edges = None
    # Fallback to masked edges
    if edges is None and hasattr(pp, "getEdgesMasked"):
        try:
            edges = pp.getEdgesMasked(image)[0]
        except Exception:
            edges = None
    # Final fallback to gradient magnitude
    if edges is None and hasattr(pp, "gradient"):
        try:
            edges = pp.gradient(image)[0]
        except Exception:
            edges = None

    if edges is None:
        raise RuntimeError("Could not derive edges from image via preprocessing APIs")

    ys, xs = np.nonzero(edges)
    return np.column_stack((xs, ys))

__all__.append("edge_points_from_image")
