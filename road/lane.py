"""Turn a road mask into a stable lane and a steering target.

The road is treated as flat ground seen by a forward-facing camera in the middle of the
vehicle. Once the horizon row is known (estimated automatically from the road width),
each image row maps to a relative distance ahead and each pixel to a sideways position.
In those ground coordinates the road is fitted as a straight strip of constant width
(a rectangle on the ground, a trapezoid in the image), which stays stable when one edge
is off-screen or the mask is ragged, taken as the median of the last few frames and then
smoothed with an adaptive (1-euro) filter, as is the steering target. Final offset
and steering outputs pass through a time-based low-pass filter and a hard rate limit.
Where the road splits (see junction.py), each way to go gets its own rectangle.
No camera calibration is needed.
"""
from collections import deque
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .filters import OneEuroFilter, RateLimitedLowPass
from .junction import JunctionTracker, build_ways, find_junction

MIN_ROWS = 5
NEAR_HORIZON = 0.12   # ignore rows this close to the horizon (fraction of height): too far to trust
CHECK_Z = np.array([1.5, 3.0, 5.0])  # distances (1 / fraction of height below the horizon) used to
                                    # compare a new fit with the recent road: near, middle, far
WARMUP = 30           # frames of road width to collect before looking for junctions


@dataclass
class LaneResult:
    found: bool
    held: bool = False                   # absent/rejected observation; showing the last good lane
    mask: np.ndarray | None = None       # cleaned road mask
    horizon: float | None = None         # image row of the horizon
    rows: np.ndarray | None = None       # y of every sample, bottom to top
    left: np.ndarray | None = None       # x of the lane (road) edges at each row
    right: np.ndarray | None = None
    path: np.ndarray | None = None       # x of the lane centre: the line to follow
    target: tuple | None = None          # (x, y) lookahead point to steer towards
    steer_deg: float = 0.0               # direction of the target from straight ahead, + = right
    offset: float = 0.0                  # lane centre vs vehicle, in lane widths, + = centre is to the right
    raw_steer_deg: float = 0.0           # upstream steering target before the final output limiter
    raw_offset: float = 0.0              # upstream lane offset before the final output limiter
    obstacle: float | None = None        # image row of the nearest obstacle in the lane
    ways: list = field(default_factory=list)  # at a junction: every way to go (junction.Way)
    way: str = ""                        # the way being followed at a junction
    heading_deg: float | None = None     # road direction relative to vehicle, independent of offset
    straight_clearance: float | None = None  # minimum forward-ray clearance in lane widths


class LaneEstimator:
    def __init__(self, lookahead=0.4, horizon=None, hfov=70.0, average=5, hold_frames=15, row_step=4,
                 prefer="straight", fps=30.0, stabilize=True, smoothing_tau=0.25,
                 max_offset_rate=0.35, max_steer_rate=20.0, max_dt=0.1):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        if not np.isfinite(lookahead) or not 0 < lookahead <= 1:
            raise ValueError("lookahead must be in (0, 1]")
        if not np.isfinite(hfov) or not 0 < hfov < 180:
            raise ValueError("hfov must be in (0, 180)")
        if horizon is not None and (not np.isfinite(horizon) or not 0 <= horizon < 1 - NEAR_HORIZON):
            raise ValueError("horizon must leave room for the visible road")
        for name, value, minimum in (("average", average, 1), ("hold_frames", hold_frames, 0),
                                     ("row_step", row_step, 1)):
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if prefer not in {"straight", "left", "right"}:
            raise ValueError("prefer must be straight, left or right")
        self.prefer = prefer            # way to take at a junction: straight, left or right
        self.dt = 1.0 / fps
        self.stabilize = stabilize      # adaptive smoothing of the lane and the steering target
        # Per lane value [heading, offset, width, top row]: how much to smooth when steady (min
        # cutoff, Hz) and how fast a real change opens the filter up (beta). The width barely
        # changes, so it is smoothed hardest.
        self.lane_filter = OneEuroFilter(min_cutoff=[0.8, 0.8, 0.3, 1.0], beta=[5.0, 1.0, 0.2, 0.02])
        self.target_filter = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        self.output_filter = RateLimitedLowPass(
            tau=smoothing_tau, max_rate=[max_offset_rate, max_steer_rate],
            initial=[0.0, 0.0], max_dt=max_dt)
        self.max_dt = self.output_filter.max_dt
        self.rng = np.random.default_rng(0)
        self._last_result = None
        self._shape = None
        self.junctions = JunctionTracker()
        self.fit_below = None           # where the road before a junction ends, if the road ends there
        self.lookahead = lookahead      # target row: 0 = horizon, 1 = bottom of the frame
        self.fixed_horizon = horizon    # horizon row as a fraction of height; None = estimate it
        self.hfov = hfov                # camera's horizontal field of view in degrees
        self.hold_frames = hold_frames  # keep the last road this many frames when it is lost
        self.row_step = row_step
        self.horizons = deque(maxlen=150)    # the horizon and the road width barely change, so
        self.widths = deque(maxlen=150)      # use the median over the last ~5 s for both
        self.recent = deque(maxlen=average)    # last good fits, each [heading, offset, width, top row]
        self.rejected = deque(maxlen=max(3, average))  # consecutive, mutually agreeing new fits
        self.missed = 0

    def horizon(self, h):
        if self.fixed_horizon is not None:
            return self.fixed_horizon * h
        if len(self.horizons) >= 5:
            return float(np.median(self.horizons))
        return 0.33 * h

    def road_width(self):
        return float(np.median(self.widths)) if len(self.widths) >= 5 else None

    def __call__(self, mask, dt=None):
        """Estimate a lane using elapsed frame time in seconds (nominal FPS if omitted)."""
        dt = self.dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite, positive number of seconds")
        dt = min(dt, self.max_dt)
        mask = np.asarray(mask)
        if mask.ndim != 2 or min(mask.shape) < 8:
            raise ValueError("mask must be a two-dimensional image at least 8 pixels in each dimension")
        # Both 0/1 and 0/255 masks are supported. Nonfinite pixels cannot count as road.
        mask = (np.isfinite(mask) & (mask > 0)).astype(np.uint8)
        if self._shape is not None and self._shape != mask.shape:
            self._clear_tracking()
            self.horizons.clear()
            self.widths.clear()
            self._last_result = None
        self._shape = mask.shape
        h, w = mask.shape
        rows = np.arange(h - 1, -1, -self.row_step)
        road = clean_mask(mask)
        edges = row_extents(road, rows) if road is not None else None
        top = None
        if edges is not None:
            top = rows[~np.isnan(edges[0])].min()
            edges = self._before_junction(rows, edges)
        if edges is not None and self.fixed_horizon is None:
            y_v = horizon_row(rows, *edges, h)
            if y_v is not None:
                self.horizons.append(y_v)
        y_h = self.horizon(h)
        fit = None
        if edges is not None:
            width = measure_width(rows, *edges, y_h, w, h)
            if width is not None:
                self.widths.append(width)
            fit = fit_road(rows, *edges, y_h, w, h, self.road_width(), rng=self.rng)
            if fit is not None:
                fit[3] = max(top, y_h + NEAR_HORIZON * h)  # still show the road beyond a junction
                if not np.all(np.isfinite(fit)) or fit[2] <= 0:
                    fit = None

        if fit is None:
            self.rejected.clear()  # confirmation requires consecutive observed fits
            self.missed += 1
            if not self.recent or self.missed > self.hold_frames:
                self._clear_tracking()
                return self._lost(road, y_h)
            return self._held(road)
        elif not self._remember(fit):
            self.missed += 1
            # Retain the trusted reference while rejecting inconsistent masks. Otherwise
            # a long run of noise would erase the reference and accept its next sample.
            if self.missed > self.hold_frames:
                return self._lost(road, y_h)
            return self._held(road)
        else:
            self.missed = 0
        result = self._build(mask, road, y_h, held=False, dt=dt)
        if result.found:
            self._last_result = result
        return result

    def _clear_tracking(self):
        self.recent.clear()
        self.rejected.clear()
        self.junctions.clear()
        self.lane_filter.reset()
        self.target_filter.reset()
        self.fit_below = None
        # Keep output_filter: reacquisition must obey the same bounds as tracking.

    def _lost(self, road, y_h):
        offset, steering = self.output_filter.x
        last = self._last_result
        if not self.stabilize and last is not None:
            offset, steering = last.offset, last.steer_deg
        return LaneResult(False, mask=road, horizon=y_h, offset=float(offset),
                          steer_deg=float(steering),
                          raw_offset=last.raw_offset if last is not None else 0.0,
                          raw_steer_deg=last.raw_steer_deg if last is not None else 0.0)

    def _held(self, road):
        # Do not integrate filters or update junctions from an absent/rejected observation.
        if self._last_result is None:
            return self._lost(road, self.horizon(self._shape[0]))
        return replace(self._last_result, held=True, mask=road)

    def _before_junction(self, rows, edges):
        """Where the road ends at a junction ahead (a T-junction, say), fit the lane only to the
        road before it, so the crossing road doesn't bend or widen it - as long as enough of
        that road is in view. Not done at side roads, where the road carries on: cutting the
        lane short there makes it switch back and forth as the junction comes and goes."""
        if self.fit_below is None:
            return edges
        below = rows >= self.fit_below
        left, right, left_seen, right_seen = edges
        if (below & ~np.isnan(left)).sum() < 8:
            return edges
        return (np.where(below, left, np.nan), np.where(below, right, np.nan),
                left_seen & below, right_seen & below)

    def _remember(self, fit):
        """Add a fit to the recent frames, leaving out one-off glitches. The road is the median of
        the recent frames, so a stray fit that does get in can't drag it either."""
        if not np.all(np.isfinite(fit)) or fit[2] <= 0:
            self.rejected.clear()
            return False
        if self.recent:
            road = np.median(self.recent, axis=0)
            jump = np.abs(np.polyval(fit[:2], CHECK_Z) - np.polyval(road[:2], CHECK_Z)).max() / road[2]
            width_change = abs(fit[2] - road[2]) / road[2]
            if jump > 0.25 or width_change > 0.35:
                # A frame that disagrees with the recent road is more likely a bad mask than a real
                # change - unless a whole window of frames in a row agrees on it: then that's the road.
                # Merely collecting N rejected fits accepts alternating left/right noise.
                # A new candidate must agree with every sample in its confirmation window.
                for candidate in self.rejected:
                    width = min(candidate[2], fit[2])
                    distance = np.max(np.abs(np.polyval(fit[:2] - candidate[:2], CHECK_Z))) / width
                    if distance > 0.15 or abs(fit[2] - candidate[2]) / width > 0.2:
                        self.rejected.clear()
                        break
                self.rejected.append(fit.copy())
                if len(self.rejected) == self.rejected.maxlen:
                    self.recent.clear()
                    self.recent.extend(self.rejected)
                    self.rejected.clear()
                    return True
                return False
        self.rejected.clear()
        self.recent.append(fit.copy())
        return True

    def _build(self, raw, road, y_h, held, dt):
        h, w = raw.shape
        cx = (w - 1) / 2
        params = np.median(self.recent, axis=0)
        if self.stabilize:
            params = self.lane_filter(params, dt)
        heading, offset, width, top = params
        y_top = max(top, y_h + NEAR_HORIZON * h)
        rows = np.arange(h - 1, y_top, -self.row_step, dtype=float)
        if len(rows) < 2:
            return self._lost(road, y_h)
        d = rows - y_h
        centre = heading * h / d + offset  # in ground units; image x = cx + ground x * d
        left, right, path = cx + (centre - width / 2) * d, cx + (centre + width / 2) * d, cx + centre * d

        i = np.abs(rows - max(y_h + self.lookahead * (h - y_h), rows[-1])).argmin()
        tx, ty = float(path[i]), float(rows[i])
        focal = (w / 2) / np.tan(np.radians(self.hfov) / 2)
        res = LaneResult(
            found=True, held=held, mask=road, horizon=y_h, rows=rows,
            left=left, right=right, path=path, target=(tx, ty),
            steer_deg=float(np.degrees(np.arctan2(tx - cx, focal))),
            offset=float(centre[0] / width),
            # X = heading * h / (y - horizon) + offset. The vanishing
            # point's x displacement is heading*h; lateral offset alone
            # must not be mistaken for a change of road direction.
            heading_deg=float(np.degrees(np.arctan2(heading * h, focal))),
            straight_clearance=float(np.minimum((cx - left) / (right - left),
                                                 (right - cx) / (right - left)).min()),
        )
        self.fit_below = None
        # Junctions are found by comparing the road with its usual width, so wait until that
        # has settled; right after the start the lane is still moving about.
        if road is not None and len(self.widths) >= WARMUP:
            self._junction(res, road, width, focal)
        if self.stabilize:
            tx, ty = self.target_filter(res.target, dt)
            res.target = (float(tx), float(ty))
            res.steer_deg = float(np.degrees(np.arctan2(tx - cx, focal)))
        res.raw_offset, res.raw_steer_deg = res.offset, res.steer_deg
        if self.stabilize:
            offset, steering = self.output_filter([res.offset, res.steer_deg], dt)
            res.offset, res.steer_deg = float(offset), float(steering)
            # The drawn target must agree with the angle actually sent downstream.
            res.target = (float(cx + focal * np.tan(np.radians(res.steer_deg))), res.target[1])
        res.obstacle = nearest_obstacle(raw, res.rows, res.left, res.right)
        return res

    def _junction(self, res, road, width, focal):
        """Find the ways to go where the road splits, and follow the preferred one."""
        h, w = road.shape
        y_h = res.horizon

        def depth(y0, y1):
            """Ground distance between image rows, in lane widths."""
            return np.abs(focal / (np.asarray(y0) - y_h) - focal / (np.asarray(y1) - y_h)) / width

        # The road stops short of the far limit: it ends there (or turns).
        stops = res.rows[-1] > y_h + NEAR_HORIZON * h + 2 * self.row_step
        junction = self.junctions.update(find_junction(road, res.rows, res.left, res.right, depth, y_h,
                                                       res.rows[-1] if stops else None,
                                                       shown=self.junctions.shown))
        target_row = y_h + self.lookahead * (h - y_h)
        ways, trunk_end = build_ways(junction, res.rows, res.left, res.right, target_row, y_h)
        if not ways:
            return
        if junction.end is not None:
            self.fit_below = trunk_end
        # The lane itself is only the shared stretch of road before the junction.
        if trunk_end is not None:
            keep = res.rows >= trunk_end
            res.rows, res.left, res.right, res.path = (a[keep] for a in (res.rows, res.left, res.right, res.path))
        order = {"straight": ("straight", "left", "right"), "left": ("left", "straight", "right"),
                 "right": ("right", "straight", "left")}[self.prefer]
        chosen = min(ways, key=lambda way: order.index(way.name))
        res.ways, res.way, res.target = ways, chosen.name, chosen.target
        res.steer_deg = float(np.degrees(np.arctan2(chosen.target[0] - (w - 1) / 2, focal)))


def clean_mask(mask):
    """Keep only the road region in front of the vehicle, with small gaps and specks removed."""
    h, w = mask.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w // 40 | 1,) * 2)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, labels = cv2.connectedComponents(m)
    if n < 2:
        return None
    # Prefer the region right in front of the vehicle (bottom fifth, middle half of the frame),
    # otherwise whatever region touches the bottom of the frame.
    for region in (labels[int(h * 0.8):, w // 4:3 * w // 4], labels[int(h * 0.8):]):
        counts = np.bincount(region.ravel(), minlength=n)
        counts[0] = 0
        if counts.max():
            return (labels == counts.argmax()).astype(np.uint8)
    return None


def row_extents(road, rows, min_width=0.03):
    """Leftmost and rightmost road pixel per row (NaN where there is no road),
    plus whether each edge is really seen or cut off by the image border."""
    _, w = road.shape
    sub = road[rows].astype(bool)
    left = sub.argmax(1).astype(float)
    right = (w - 1 - sub[:, ::-1].argmax(1)).astype(float)
    has = sub.any(1) & (right - left >= min_width * w)
    if has.sum() < MIN_ROWS:
        return None
    left[~has] = right[~has] = np.nan
    return left, right, has & (left > 1), has & (right < w - 2)


def horizon_row(rows, left, right, left_seen, right_seen, h):
    """Row where the road's width in pixels shrinks to zero. On flat ground a road of constant
    width looks narrower in proportion to its distance, so this is the horizon, even on curves."""
    both = left_seen & right_seen
    if both.sum() < 8 or np.ptp(rows[both]) < 0.1 * h:
        return None
    slope, icept = robust_polyfit(rows[both].astype(float), (right - left)[both], 1, floor=2.0)
    if slope < 0.2:
        return None
    y = -icept / slope
    return float(y) if 0.05 * h <= y <= min(0.65 * h, rows[both].min()) else None


def to_ground(rows, left, right, y_h, w, h):
    """Road edges in ground coordinates: z = h / (row - horizon) is distance ahead and
    x = (col - centre col) / (row - horizon) is sideways. Rows near the horizon are dropped."""
    d = rows - y_h
    has = ~np.isnan(left) & (d > NEAR_HORIZON * h)
    d = np.where(has, d, np.nan)
    cx = (w - 1) / 2
    return has, h / d, (left - cx) / d, (right - cx) / d


def measure_width(rows, left, right, left_seen, right_seen, y_h, w, h):
    """Road width in ground units from the rows where both edges are in view, or None.
    Never less than the green region itself, so the lane always covers the visible road -
    measured on the nearer half of the road only, so a crossroad ahead doesn't widen it."""
    has, z, xl, xr = to_ground(rows, left, right, y_h, w, h)
    both = has & left_seen & right_seen
    if both.sum() < MIN_ROWS:
        return None
    near = has & (z <= np.median(z[has]))
    return float(max(np.median(xr[both] - xl[both]), np.percentile(xr[near] - xl[near], 80)))


def fit_road(rows, left, right, left_seen, right_seen, y_h, w, h, width, rng=None):
    """Fit the road as a straight strip of the given width - a rectangle on the ground - to the
    edges of the road mask, with the centre line x = heading * z + offset in ground coordinates.
    Returns [heading, offset, width, farthest road row]."""
    has, z, xl, xr = to_ground(rows, left, right, y_h, w, h)
    if has.sum() < MIN_ROWS:
        return None
    left_seen, right_seen = left_seen & has, right_seen & has
    if width is None:  # no reliable width yet: the road is at least as wide as what we see
        width = np.median(xr[has] - xl[has])
    if not np.isfinite(width) or width <= 0:
        return None

    if left_seen.sum() + right_seen.sum() >= MIN_ROWS:
        zs = np.concatenate([z[left_seen], z[right_seen]])
        centres = np.concatenate([xl[left_seen] + width / 2, xr[right_seen] - width / 2])
    else:  # road wider than the view on every row (e.g. an open lot): aim at the middle of what we see
        zs, centres = z[has], (xl[has] + xr[has]) / 2

    # Errors are compared in pixels (ground error x row distance to the horizon), so a ragged far
    # edge counts no more than a clean near one. RANSAC ignores stray edge points such as a
    # bright patch the model missed, as long as most of the edge agrees.
    heading, offset = ransac_line(zs, centres, h / zs, tol=max(4.0, 0.015 * w), rng=rng)
    return np.array([heading, offset, width, rows[has].min()])


def ransac_line(x, y, scale, tol, iters=100, rng=None):
    """Fit y = a x + b to the largest group of points that agree within tol (error x scale),
    then refit on that group by least squares."""
    best = None
    rng = np.random.default_rng(0) if rng is None else rng
    for _ in range(iters):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 0.05:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        agree = np.abs(a * (x - x[i]) + y[i] - y) * scale < tol
        if best is None or agree.sum() > best.sum():
            best = agree
    if best is None or best.sum() < 3:
        best = np.ones(len(x), bool)
    return np.polyfit(x[best], y[best], 1, w=scale[best])


def robust_polyfit(x, y, deg, weights=None, floor=0.02):
    """np.polyfit that drops outliers (beyond 3 robust standard deviations) and refits."""
    keep = np.ones(len(x), bool)
    for _ in range(3):
        coef = np.polyfit(x[keep], y[keep], deg, w=None if weights is None else weights[keep])
        err = np.abs(np.polyval(coef, x) - y)
        new = err <= max(3 * 1.4826 * np.median(err[keep]), floor)
        if new.sum() < deg + 3 or (new == keep).all():
            break
        keep = new
    return coef


def nearest_obstacle(raw, rows, lane_l, lane_r, share=0.3, run=3):
    """Nearest row where something that isn't road covers `share` of the middle half of the lane
    for `run` rows in a row, scanning away from the vehicle. None if the lane is clear."""
    h, w = raw.shape
    hits = 0
    for i, (y, l, r) in enumerate(zip(rows, lane_l, lane_r)):
        if y > 0.9 * h:  # the model often misses washed-out road right in front of the camera;
            continue      # a real obstacle is seen in the rows above before it gets that close
        quarter = (r - l) / 4
        a, b = int(max(0, l + quarter)), int(min(w, r - quarter))
        if b - a < 2:
            hits = 0
            continue
        hits = hits + 1 if 1 - raw[int(y), a:b].mean() >= share else 0
        if hits == run:
            return float(rows[i - run + 1])
    return None
