"""Find where the road splits - side roads, T-junctions, crossroads - and the ways you can go.

Works row by row on the image. Within one image row, sideways distances on the ground are
in proportion to pixels, so at each row it compares how far the road reaches sideways from
the lane's centre with the lane's width there:
  * a band of rows where the road reaches well past one lane edge - further than it usually
    does along this road - is the mouth of a side road;
  * rows where the whole lane stops being road mean the road ends ahead (e.g. the island of a
    T-junction), and road running out of view sideways just before that end is a way to go.
Side roads are taken to leave at right angles, and every way is drawn as its own rectangle.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Way:
    name: str             # straight, left or right
    outline: np.ndarray   # image points of its rectangle
    path: np.ndarray      # image points of the centre line to follow, from the vehicle
    target: tuple         # (x, y) lookahead point on the path


@dataclass
class Junction:
    end: float | None     # image row where the road ends ahead, None if it goes on
    sides: dict           # "left"/"right" -> (far row, near row) of the side road's mouth


def find_junction(road, rows, left, right, depth, horizon, road_top_end=None, shown=()):
    """Look for side roads and a road end in the cleaned road mask, given the lane edges
    (image x at each of `rows`, bottom to top). `depth(y0, y1)` is the ground distance
    between two rows in lane widths, used to ignore openings too narrow to be a road.
    `road_top_end` is the row where the road visibly stops, if it stops short of the far limit.
    Whatever is already `shown` ("left", "right", "end") only has to pass looser thresholds to
    stay, so a junction near a threshold doesn't come and go."""
    h, w = road.shape
    width = right - left
    centre = np.clip((left + right) / 2, 0, w - 1).round().astype(int)
    usable = rows < 0.9 * h  # the model is unreliable on the last few rows

    # Road end: going up the lane, the first two rows where nearly all of its middle is not road.
    end = None
    covered = np.array([_share(road[int(y)], l + 0.2 * (r - l), r - 0.2 * (r - l))
                        for y, l, r in zip(rows, left, right)])
    ended = usable & (covered < (0.4 if "end" in shown else 0.3))
    for i in np.flatnonzero(ended[:-1] & ended[1:]):
        # A real end has (almost) no road beyond it; a washed-out patch or a speed breaker that
        # the model missed has road again past it.
        if ended[i:].mean() >= 0.6:
            end = float(rows[i])
            break
    if end is None and road_top_end is not None:
        end = float(road_top_end)
    elif end is None and not _continues(road, rows, left, right, horizon, 0.6 if "end" in shown else 0.5):
        end = float(rows[-1])

    sides = {}
    for name, side in (("left", -1), ("right", 1)):
        reach, at_border = np.zeros(len(rows)), np.zeros(len(rows), bool)
        for i, (y, c) in enumerate(zip(rows, centre)):
            reach[i], at_border[i] = _reach(road[int(y)], c, side)
        # How far past the lane edge the road goes, in lane widths, compared with how far it
        # usually goes along this road (the lane is never a perfect fit).
        beyond = (reach - width / 2) / width
        usual = max(np.median(beyond[usable & ~at_border]), 0.0) if (usable & ~at_border).sum() >= 5 else 0.0
        extra = beyond - usual
        loose = name in shown
        opening = usable & ((extra >= (0.3 if loose else 0.5)) | (at_border & (extra >= (0.15 if loose else 0.3))))
        if end is not None:
            # Just before the road ends, road running out of view sideways is a way to go,
            # even when the lane already fills the view.
            band = (rows >= end) & (depth(end, rows) <= 1.0)
            opening |= usable & band & at_border & (beyond >= -0.25)
            opening &= rows >= end
        run = _longest_run(opening)
        if run is None:
            continue
        mouth = rows[run[0]:run[1]]
        near = mouth[0]
        # A side road is about as wide as the main one; when its far edge is out of sight the
        # opening runs on to the far limit, so cap it at 1.2 lane widths.
        far = mouth[depth(mouth, near) <= 1.2].min()
        if depth(far, near) >= (0.2 if name in shown else 0.35):
            sides[name] = (float(far), float(near))
    return Junction(end, sides)


def _continues(road, rows, left, right, horizon, share=0.5):
    """Whether the road carries on beyond the far limit of the lane, towards the horizon:
    the middle of the lane, extended in a straight line, is mostly (`share`) road up there."""
    far = np.arange(rows[-1] - 1, horizon + 0.03 * road.shape[0], -2.0)
    if len(far) < 2 or len(rows) < 2:
        return True
    l_far, r_far = (np.polyval(np.polyfit(rows, edge, 1), far) for edge in (left, right))
    covered = [_share(road[int(y)], l + 0.4 * (r - l), r - 0.4 * (r - l)) for y, l, r in zip(far, l_far, r_far)]
    return np.mean(covered) >= share


def _share(row, l, r):
    """Share of the lane between x = l and x = r (clipped to the image) that is road."""
    a, b = int(max(0, l)), int(min(len(row), r))
    return row[a:b].mean() if b - a >= 3 else 1.0


def _reach(row, c, side):
    """How far road runs from column c towards side -1 (left) or +1 (right), and whether it
    reaches the image border."""
    if not row[c]:
        return 0.0, False
    seg = row[c::-1] if side < 0 else row[c:]
    gaps = np.flatnonzero(seg == 0)
    if len(gaps):
        return float(gaps[0]), False
    return float(len(seg)), True


def _longest_run(flags):
    """(start, end) of the longest run of True values, or None."""
    best, start = None, None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


class JunctionTracker:
    """Shows a side road or a road end once it turns up in most of the recent frames, and only
    drops it again when it has (almost) stopped turning up - hysteresis, so junctions don't
    blink on and off at the threshold. Positions are the median of the recent detections."""

    def __init__(self, window=7):
        self.window = window
        self.show_at = window // 2 + 1  # seen in this many of the recent frames: show it
        self.hide_at = window // 3      # seen in this many or fewer: hide it
        self.ends = deque(maxlen=window)
        self.sides = {"left": deque(maxlen=window), "right": deque(maxlen=window)}
        self.shown = set()

    def update(self, junction):
        self.ends.append(junction.end)
        for name, recent in self.sides.items():
            recent.append(junction.sides.get(name))
        sides = {}
        for name, recent in self.sides.items():
            found = [s for s in recent if s is not None]
            if self._keep(name, len(found)):
                sides[name] = tuple(np.median(found, axis=0))
        ends = [e for e in self.ends if e is not None]
        end = float(np.median(ends)) if self._keep("end", len(ends)) else None
        return Junction(end, sides)

    def _keep(self, name, count):
        if count >= self.show_at or (name in self.shown and count > self.hide_at):
            self.shown.add(name)
            return True
        self.shown.discard(name)
        return False

    def clear(self):
        self.ends.clear()
        self.shown.clear()
        for recent in self.sides.values():
            recent.clear()


def build_ways(junction, rows, left, right, target_row, horizon):
    """Rectangles, centre lines and steering targets for every way to go, and the image row
    where the shared stretch of road before the junction ends. ([], None) if the road just
    goes on. `target_row` is where the lane's lookahead target would be."""
    if not junction.sides:
        return [], None
    width = right - left
    centre = (left + right) / 2
    trunk_end = max(near for _, near in junction.sides.values())  # nearest mouth
    mouth_top = min(far for far, _ in junction.sides.values())
    trunk = rows >= trunk_end

    def along_trunk(y_stop):
        keep = rows >= y_stop
        return np.stack([centre[keep], rows[keep]], 1)

    ways = []
    if junction.end is None:
        # Straight on: the lane carried on past the junction, towards the horizon.
        ys = np.linspace(mouth_top, horizon + 0.04 * (rows[0] + 1), 16)
        ls, rs = (np.polyval(np.polyfit(rows, edge, 1), ys) for edge in (left, right))
        outline = np.vstack([np.stack([ls, ys], 1), np.stack([rs, ys], 1)[::-1]])
        path = np.vstack([along_trunk(mouth_top), np.stack([(ls + rs) / 2, ys], 1)])
        ways.append(Way("straight", outline, path, _target(path, target_row)))

    for name, (far, near) in junction.sides.items():
        side = -1 if name == "left" else 1
        band = (rows <= near) & (rows >= far)
        if band.sum() < 2:
            continue
        ys, w_band = rows[band], width[band]
        edge = (left if side < 0 else right)[band]
        outer = edge + side * 1.5 * w_band  # the side road drawn 1.5 lane widths out
        outline = np.vstack([np.stack([edge, ys], 1), np.stack([outer, ys], 1)[::-1]])
        # Path: up the lane to the middle of the mouth, then out along the side road.
        mid = np.abs(rows - (far + near) / 2).argmin()
        out = np.linspace(centre[mid], edge.mean() + side * 0.75 * w_band.mean(), 12)
        path = np.vstack([along_trunk(rows[mid]), np.stack([out, np.full(12, rows[mid])], 1)])
        ways.append(Way(name, outline, path, _target(path, target_row)))
    return ways, float(trunk_end) if trunk.sum() >= 2 else None


def _target(path, target_row):
    """Lookahead point on a path: where it crosses the target row, or its far end if the
    path turns off before reaching that row."""
    ahead = np.flatnonzero(path[:, 1] <= target_row)
    return tuple(map(float, path[ahead[0]] if len(ahead) else path[-1]))
