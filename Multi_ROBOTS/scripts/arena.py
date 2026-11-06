"""Shared surveyed map and grid planning; no follower exteroceptive sensors."""
import heapq
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = 0.25
BOUNDS = (-5, -7, 19, 7)
# x, y, width, depth, height
OBSTACLES = [(3, 0, 1.2, 2.8, 1.2), (7, 3, 1.8, 1.2, 1.4),
             (8, -2.2, 1.5, 2, 1.2), (12, 0.8, 1.2, 2.8, 1.5),
             (15, -3.5, 1.6, 1.2, 1.0), (1, 4, 1.5, 1.0, 1.0)]
WAYPOINTS = [(1, -3.8), (5, -4.5), (10, -4.8), (16.5, -1),
             (16, 4.8), (10, 4.8), (5, 4.8), (-1, 2), (0, 0)]


def clearance(x, y):
    distances = [min(x-BOUNDS[0], BOUNDS[2]-x, y-BOUNDS[1], BOUNDS[3]-y)]
    for ox, oy, w, d, _ in OBSTACLES:
        dx, dy = abs(x-ox)-w/2, abs(y-oy)-d/2
        distances.append(math.hypot(max(0, dx), max(0, dy)) + min(0, max(dx, dy)))
    return min(distances)


def plan(start, goal, radius=0.6, points=()):
    def free(x, y):
        return min([clearance(x, y)] + [math.hypot(x-p[0], y-p[1]) for p in points])
    def cell(p):
        return (round((p[0]-BOUNDS[0])/RES), round((p[1]-BOUNDS[1])/RES))
    def point(c):
        return (c[0]*RES+BOUNDS[0], c[1]*RES+BOUNDS[1])
    a, b = cell(start), cell(goal)
    if free(*point(b)) < radius:
        candidates = [(math.dist(point((b[0]+dx, b[1]+dy)), goal), (b[0]+dx, b[1]+dy))
                      for dx in range(-10, 11) for dy in range(-10, 11)
                      if free(*point((b[0]+dx, b[1]+dy))) >= radius]
        if not candidates:
            return []
        b = min(candidates)[1]
    queue, cost, parent = [(0, a)], {a: 0}, {}
    while queue:
        _, c = heapq.heappop(queue)
        if c == b:
            path = [c]
            while c in parent:
                c = parent[c]
                path.append(c)
            return [point(p) for p in reversed(path)]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            n = c[0]+dx, c[1]+dy
            if free(*point(n)) < radius:
                continue
            if dx and dy and (free(*point((c[0]+dx, c[1]))) < radius or
                              free(*point((c[0], c[1]+dy))) < radius):
                continue
            new = cost[c] + math.hypot(dx, dy)
            if new < cost.get(n, float('inf')):
                cost[n], parent[n] = new, c
                heapq.heappush(queue, (new+math.dist(n, b), n))
    return []


def export_map():
    width = round((BOUNDS[2]-BOUNDS[0])/RES)
    height = round((BOUNDS[3]-BOUNDS[1])/RES)
    pixels = bytes(0 if clearance(BOUNDS[0]+(x+.5)*RES, BOUNDS[1]+(y+.5)*RES) < 0.05 else 254
                   for y in reversed(range(height)) for x in range(width))
    (ROOT/'maps'/'arena.pgm').write_bytes(f'P5\n{width} {height}\n255\n'.encode()+pixels)
    (ROOT/'config'/'arena.yaml').write_text('image: ../maps/arena.pgm\nresolution: 0.25\norigin: [-5.0, -7.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    (ROOT/'maps'/'arena.json').write_text(json.dumps({'bounds': BOUNDS, 'obstacles': OBSTACLES,
                                            'waypoints': WAYPOINTS}, indent=2))
