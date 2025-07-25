import random
from shapely.geometry import box

def compute_bounding_box(state):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for _, b in state:
        x0, y0, x1, y1 = b.bounds
        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)
    return min_x, min_y, max_x, max_y

def compute_perimeter(state):
    min_x, min_y, max_x, max_y = compute_bounding_box(state)
    width = max_x - min_x
    height = max_y - min_y
    return 2 * (width + height)

def get_bounding_box_limits(state):
    min_x, min_y, max_x, max_y = compute_bounding_box(state)
    return int(max_x-min_x), int(max_y-min_y)

def random_initial_state(crate_dims, boundaries, max_attempts=1000):
    state = []
    bw, bh = boundaries
    crate_dims.sort(key=lambda x: x[1] * x[2], reverse=True)
    for crate_id, w, h in crate_dims:
        placed = False
        attempts = 0
        while not placed and attempts < max_attempts:
            x = random.randint(0, bw - w)
            y = random.randint(0, bh - h)
            new_box = box(x, y, x + w, y + h)
            if all(not new_box.intersects(other_box) for _, other_box in state):
                state.append((crate_id, new_box))
                placed = True
            attempts += 1
        if not placed:
            print(f"Warning: Could not place crate {crate_id} after {max_attempts} attempts")
    return state
