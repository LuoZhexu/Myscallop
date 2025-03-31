import scallopy
from scallopy import i32
import random

@scallopy.foreign_function
def contains(s1: str, s2: str) -> i32:
    if s1 in s2:
        return 1
    return 0

ctx = scallopy.ScallopContext()
ctx.register_foreign_function(contains)

# Define relations
ctx.add_relation("grid_node", (int, int))  # Represents all nodes in# the grid
ctx.add_relation("curr_position", (int, int))  # Current position of the agent
ctx.add_relation("goal_position", (int, int))  # Goal position to reach
ctx.add_relation("is_enemy", (int, int))  # Positions occupied by enemies
# Path information: (start_x, start_y, current_x, current_y, length,
# path_string)
ctx.add_relation("path", (int, int, int, int, int, str))
# Edges between nodes: (x1, y1, x2, y2, direction)
ctx.add_relation("edge", (int, int, int, int, int))

# Track visited nodes to avoid cycles
ctx.add_relation("visited", (int, int, int, int, str, int))

# Rules to mark nodes as visited
ctx.add_rule("visited(x, y, xp, yp, S, $contains($format(\"({}, {})\", xp, "
             "yp), S)) ="
             "edge(x, y, xp, yp, _), path(x, y, x, y, 0, S)")
ctx.add_rule("visited(x, y, xpp, ypp, S, $contains($format(\"({}, {})\", xpp, "
             "ypp), S)) ="
             "path(x, y, xp, yp, p, S), edge(xp, yp, xpp, ypp, _)")

# Additional relations for grid boundaries and actions
ctx.add_relation("max_x", (int,))  # Maximum x-coordinate of the grid
ctx.add_relation("max_y", (int,))  # Maximum y-coordinate of the grid
ctx.add_relation("action", (int,))  # Actions: 1-up, 2-right, 3-down, 4-left

# Randomly generate grid dimensions (between 3x3 and 7x7)
grid_width = random.randint(3, 7)
grid_height = random.randint(3, 7)
print("Grid size: {} x {}".format(grid_width, grid_height))
ctx.add_facts("max_x", [(grid_width - 1,)])
ctx.add_facts("max_y", [(grid_height - 1,)])

# Generate all grid nodes
grid_nodes = [(x, y) for x in range(grid_width) for y in range(grid_height)]
ctx.add_facts("grid_node", grid_nodes)

# Randomly place enemies (1/3 to 1/2 of the grid nodes)
num_enemies = random.randint(len(grid_nodes) // 3, len(grid_nodes) // 2)
enemies = random.sample(grid_nodes, num_enemies)
print("Enemies at:", enemies)
ctx.add_facts("is_enemy", enemies)

# Randomly select start and goal positions (must not be on enemies)
possible_positions = [pos for pos in grid_nodes if pos not in enemies]
start_pos = random.choice(possible_positions)
goal_pos = random.choice(possible_positions)

# Ensure start and goal are different
while goal_pos == start_pos:
    goal_pos = random.choice(possible_positions)
print("Start position:", start_pos)
print("Goal position:", goal_pos)
ctx.add_facts("curr_position", [start_pos])
ctx.add_facts("goal_position", [goal_pos])

# Define valid nodes (non-enemy nodes)
ctx.add_rule("node(x, y) = grid_node(x, y), not is_enemy(x, y)")

# Define edges between nodes (up, right, down, left)
ctx.add_rule("edge(x, y, x, yp, 1) = node(x, y), node(x, yp), max_y(ym), "
             "yp == y + 1, y != ym")  # Up
ctx.add_rule("edge(x, y, xp, y, 2) = node(x, y), node(xp, y), max_x(xm), "
             "xp == x + 1, x != xm")  # Right
ctx.add_rule("edge(x, y, x, yp, 3) = node(x, y), node(x, yp), yp == y - "
             "1, y != 0")  # Down
ctx.add_rule("edge(x, y, xp, y, 4) = node(x, y), node(xp, y), xp == x - 1, "
             "x != 0")  # Left

# Initialize path from the current position
ctx.add_rule("path(x, y, x, y, 0, $format(\"({}, {})\", x, y)) = "
             "curr_position(x, y)")

# Extend path by one step (first step)
ctx.add_rule("path(x, y, xp, yp, 1, $string_concat(S, \" \", $format(\"({}, "
             "{})\", xp, yp))) = edge(x, y, xp, yp, _), path(x, y, x, y, 0, "
             "S), visited(x, y, xp, yp, S, 0)")

# Extend path recursively (subsequent steps)
ctx.add_rule("path(x, y, xpp, ypp, p + 1, $string_concat(S, \" \", $format("
             "\"({}, {})\", xpp, ypp))) = path(x, y, xp, yp, p, S), edge(xp, "
             "yp, xpp, ypp, _), visited(x, y, xpp, ypp, S, 0), "
             "not goal_position(xp, yp)")

# Determine next possible positions and score actions based on path to goal
ctx.add_rule("next_position(a, xp, yp) = curr_position(x, y), edge(x, y, xp, "
             "yp, a)")
ctx.add_rule("action_score(a) = next_position(a, x, y), goal_position(gx, "
             "gy), path(x, y, gx, gy, p, S)")

ctx.run()

# Extract and analyze paths
path_list = list(ctx.relation("path"))
print("All paths found:", path_list)

# Filter paths that start at start_pos and end at goal_pos
valid_paths = [p for p in path_list if p[0:2] == start_pos
               and p[2:4] == goal_pos]
if not valid_paths:
    print("No valid path from start to goal found!")
else:
    # Find the shortest path
    best = min(valid_paths, key=lambda t: t[4])
    print("Shortest path from {} to {} is:".format(start_pos, goal_pos))
    print("Path length:", best[4])
    print("Path:", best[5])

print("done")
