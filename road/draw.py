"""Overlay the road, the lane, the ways to go at a junction and the steering info on a frame."""
import cv2
import numpy as np

ROAD = (0, 170, 0)
LANE = (255, 120, 0)
EDGE = (0, 220, 255)
PATH = (255, 0, 255)
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
RED = (40, 40, 255)
ORANGE = (0, 160, 255)
WAY_COLORS = {"straight": (255, 120, 0), "left": (0, 140, 255), "right": (200, 60, 200)}


def draw(frame, res, fps=None):
    out = frame.copy()
    h, w = out.shape[:2]
    if res.mask is not None:
        tint(out, res.mask.astype(bool), ROAD, 0.3)
    if res.horizon is not None:
        y = int(round(res.horizon))
        cv2.line(out, (0, y), (w - 1, y), GREY, 1, cv2.LINE_AA)

    if res.found:
        def pts(xs):
            return np.stack([xs, res.rows], 1).round().astype(np.int32)

        lane = np.zeros((h, w), np.uint8)
        cv2.fillPoly(lane, [np.vstack([pts(res.left), pts(res.right)[::-1]])], 1)
        tint(out, lane.astype(bool), LANE, 0.35)
        for xs in (res.left, res.right):
            cv2.polylines(out, [pts(xs)], False, EDGE, 2, cv2.LINE_AA)
        if res.ways:
            draw_ways(out, res)
        else:
            cv2.polylines(out, [pts(res.path)], False, PATH, 2, cv2.LINE_AA)
        if res.obstacle is not None:
            i = np.abs(res.rows - res.obstacle).argmin()
            y = int(round(res.obstacle))
            cv2.line(out, (int(res.left[i]), y), (int(res.right[i]), y), RED, 3, cv2.LINE_AA)
        tx, ty = (int(round(v)) for v in res.target)
        cv2.line(out, (w // 2, h - 1), (tx, ty), WHITE, 1, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 6, PATH, -1, cv2.LINE_AA)

    lines = []
    if not res.found:
        lines.append(("NO ROAD", RED))
    else:
        lines.append((f"steer {res.steer_deg:+5.1f} deg", WHITE))
        lines.append((f"offset {res.offset:+.2f} lane", WHITE))
        if res.ways:
            names = [way.name for way in res.ways]
            label = "TURN" if len(names) == 1 else "JUNCTION"
            lines.append((f"{label}: {' / '.join(names)} -> {res.way}", ORANGE))
        if res.held:
            lines.append(("road lost - holding last lane", ORANGE))
        if res.obstacle is not None:
            lines.append(("OBSTACLE IN LANE", RED))
    if fps:
        lines.append((f"{fps:.0f} fps", WHITE))
    hud(out, lines)
    return out


def draw_ways(img, res):
    """Each way to go as its own rectangle and label; the path along the one being followed."""
    h, w = img.shape[:2]
    top = int(res.horizon + 0.03 * h)  # stay clear of the horizon itself
    for way in res.ways:
        area = np.zeros((h, w), np.uint8)
        cv2.fillPoly(area, [way.outline.round().astype(np.int32)], 1)
        area[:top] = 0
        chosen = way.name == res.way
        tint(img, area.astype(bool), WAY_COLORS[way.name], 0.5 if chosen else 0.3)
        edges = cv2.findContours(area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(img, edges, -1, WAY_COLORS[way.name], 3 if chosen else 1, cv2.LINE_AA)
        # Label where the way's centre line is last on screen.
        on_screen = [(x, y) for x, y in way.path if 0 <= x < w and top <= y < h]
        if on_screen:
            x, y = on_screen[-1]
            text = way.name.upper()
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = int(np.clip(x - tw / 2, 2, w - tw - 2))
            y = int(np.clip(y, top + th + 2, h - 2))
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WAY_COLORS[way.name], 1, cv2.LINE_AA)
        if chosen:
            path = way.path[way.path[:, 1] >= top]
            cv2.polylines(img, [path.round().astype(np.int32)], False, PATH, 2, cv2.LINE_AA)


def tint(img, mask, color, alpha):
    """Blend `color` into img where mask is set, in place."""
    blended = cv2.addWeighted(img, 1 - alpha, np.full_like(img, color), alpha, 0)
    np.copyto(img, blended, where=mask[..., None])


def hud(img, lines, scale=0.5):
    size = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for t, _ in lines]
    if not size:
        return
    lh = max(s[1] for s in size) + 8
    box = img[:lh * len(lines) + 8, :max(s[0] for s in size) + 16]
    box[:] = box // 3
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (8, (i + 1) * lh), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
