import random
import math
from shapely.geometry import box
from geometry import compute_bounding_box, get_bounding_box_limits

def max_swap_distance(T, base=5):
    return max(1, int(base * T))

def move_random(rect_entry, state, best_state, T):
    id, b = rect_entry
    bw, bh = get_bounding_box_limits(state)
    x_min, y_min, x_max, y_max = map(int, b.bounds)
    w = x_max - x_min
    h = y_max - y_min
    max_shift = max(0, int(T))

    for _ in range(100):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        new_x = x_min + dx
        new_y = y_min + dy
        if new_x < 0 or new_y < 0 or new_x + w > bw or new_y + h > bh:
            continue
        moved = box(new_x, new_y, new_x + w, new_y + h)
        if all(moved.intersection(other_b).area == 0 for oid, other_b in state if oid != id):
            return (id, moved)
    return None

def swap_random(state, best_state, T, max_attempts=1000):
    bw, bh = get_bounding_box_limits(state)
    state_copy = state.copy()
    for _ in range(max_attempts):
        i, j = random.sample(range(len(state_copy)), 2)
        (id1, b1), (id2, b2) = state_copy[i], state_copy[j]
        cx1, cy1 = b1.centroid.coords[0]
        cx2, cy2 = b2.centroid.coords[0]
        distance = math.hypot(cx1 - cx2, cy1 - cy2)
        if distance > max_swap_distance(T):
            continue
        x1, y1, x1b, y1b = map(int, b1.bounds)
        x2, y2, x2b, y2b = map(int, b2.bounds)
        w1, h1 = x1b - x1, y1b - y1
        w2, h2 = x2b - x2, y2b - y2
        if (x2 + w1 > bw or y2 + h1 > bh) or (x1 + w2 > bw or y1 + h2 > bh):
            continue
        new_b1 = box(x2, y2, x2 + w1, y2 + h1)
        new_b2 = box(x1, y1, x1 + w2, y1 + h2)
        new_state = state_copy.copy()
        new_state[i] = (id1, new_b1)
        new_state[j] = (id2, new_b2)
        for m in range(len(new_state)):
            for n in range(m + 1, len(new_state)):
                if new_state[m][1].intersects(new_state[n][1]):
                    break
            else:
                continue
            break
        else:
            return new_state
    return None

def neighbour(state, boundaries, best_state, T,max_stuck=10):
    def try_move():
        indices = list(range(len(state)))
        random.shuffle(indices)
        for i in indices:
            new_rect = move_random(state[i], state, best_state, T)
            if new_rect:
                return state[:i] + [new_rect] + state[i+1:]
        return None
    def try_swap():
        return swap_random(state, best_state, T)
    moved = try_move()
    if moved:
        return moved
    else:
        swapped = try_swap()
        if swapped:
            return swapped
        else:
            return None
