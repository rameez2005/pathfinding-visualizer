import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from collections import deque
import heapq
import time

# Grid size
GRID_SIZE = 10

# Cell types
EMPTY = 0
WALL = 1
START = 2
TARGET = 3
EXPLORED = 4
FRONTIER = 5
PATH = 6

# Colors
COLORS = {
    EMPTY: 'white',
    WALL: 'gray',
    START: 'lightgreen',
    TARGET: 'salmon',
    EXPLORED: 'lightblue',
    FRONTIER: 'yellow',
    PATH: 'purple'
}

# Movement order: Up, Right, Bottom, Bottom-Right, Left, Top-Left
MOVEMENTS = [
    (-1, 0),  # Up
    (0, 1),  # Right
    (1, 0),  # Bottom
    (1, 1),  # Bottom-Right
    (0, -1),  # Left
    (-1, -1)  # Top-Left
]


class Grid:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.start = (2, 0)
        self.target = (7, 7)
        self.explored = set()
        self.frontier = []
        self.path = []
        self.searching = False
        self.algorithm = None
        self.step_delay = 0.3

        # Initialize with walls
        self.reset()

    def reset(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.explored = set()
        self.frontier = []
        self.path = []
        self.searching = False
        self.algorithm = None

        # Set start and target
        self.start = (2, 0)
        self.target = (7, 7)
        self.grid[2][0] = START
        self.grid[7][7] = TARGET

        # Add walls
        walls = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                 (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
                 (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8)]

        for row, col in walls:
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                self.grid[row][col] = WALL

    def clear_walls(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == WALL:
                    self.grid[i][j] = EMPTY

    def is_valid(self, pos):
        row, col = pos
        return (0 <= row < GRID_SIZE and
                0 <= col < GRID_SIZE and
                self.grid[row][col] != WALL)


class BFS:
    def __init__(self, grid):
        self.grid = grid
        self.queue = deque()
        self.parent = {}

    def start(self):
        start = self.grid.start
        self.queue.append(start)
        self.parent[start] = None
        self.grid.grid[start[0]][start[1]] = FRONTIER

    def step(self):
        if not self.queue:
            self.grid.searching = False
            return False

        current = self.queue.popleft()

        # Mark as explored
        if current != self.grid.start:
            self.grid.grid[current[0]][current[1]] = EXPLORED
        self.grid.explored.add(current)

        # Check if target reached
        if current == self.grid.target:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = self.parent[current]
            path.reverse()
            self.grid.path = path

            # Mark path
            for pos in path:
                if pos != self.grid.start and pos != self.grid.target:
                    self.grid.grid[pos[0]][pos[1]] = PATH
            self.grid.searching = False
            return False

        # Expand neighbors in specified order
        for dr, dc in MOVEMENTS:
            new_pos = (current[0] + dr, current[1] + dc)

            if (self.grid.is_valid(new_pos) and
                    new_pos not in self.parent):

                self.queue.append(new_pos)
                self.parent[new_pos] = current
                if new_pos != self.grid.target:
                    self.grid.grid[new_pos[0]][new_pos[1]] = FRONTIER

        return True


class DFS:
    def __init__(self, grid):
        self.grid = grid
        self.stack = []
        self.parent = {}

    def start(self):
        start = self.grid.start
        self.stack.append(start)
        self.parent[start] = None
        self.grid.grid[start[0]][start[1]] = FRONTIER

    def step(self):
        if not self.stack:
            self.grid.searching = False
            return False

        current = self.stack.pop()

        # Mark as explored
        if current != self.grid.start:
            self.grid.grid[current[0]][current[1]] = EXPLORED
        self.grid.explored.add(current)

        # Check if target reached
        if current == self.grid.target:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = self.parent[current]
            path.reverse()
            self.grid.path = path

            # Mark path
            for pos in path:
                if pos != self.grid.start and pos != self.grid.target:
                    self.grid.grid[pos[0]][pos[1]] = PATH
            self.grid.searching = False
            return False

        # Expand neighbors in reverse order to maintain priority with stack
        for dr, dc in reversed(MOVEMENTS):
            new_pos = (current[0] + dr, current[1] + dc)

            if (self.grid.is_valid(new_pos) and
                    new_pos not in self.parent):

                self.stack.append(new_pos)
                self.parent[new_pos] = current
                if new_pos != self.grid.target:
                    self.grid.grid[new_pos[0]][new_pos[1]] = FRONTIER

        return True

# Uniform Cost Search (UCS) implementation
class UCS:
    def __init__(self, grid):
        self.grid = grid
        self.pq = []  # Priority queue: (cost, position)
        self.cost = {}  # Cost to reach each position
        self.parent = {}

    def start(self):
        start = self.grid.start
        heapq.heappush(self.pq, (0, start))
        self.cost[start] = 0
        self.parent[start] = None
        self.grid.grid[start[0]][start[1]] = FRONTIER

    def step(self):
        if not self.pq:
            self.grid.searching = False
            return False

        cost, current = heapq.heappop(self.pq)

        # Skip if already explored (stale entry in priority queue)
        if current in self.grid.explored:
            return True

        # Mark as explored
        if current != self.grid.start:
            self.grid.grid[current[0]][current[1]] = EXPLORED
        self.grid.explored.add(current)

        # Check if target reached
        if current == self.grid.target:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = self.parent[current]
            path.reverse()
            self.grid.path = path

            # Mark path
            for pos in path:
                if pos != self.grid.start and pos != self.grid.target:
                    self.grid.grid[pos[0]][pos[1]] = PATH
            self.grid.searching = False
            return False

        # Expand neighbors
        for dr, dc in MOVEMENTS:
            new_pos = (current[0] + dr, current[1] + dc)

            if self.grid.is_valid(new_pos):
                # Diagonal movement costs more
                step_cost = 1.4 if dr != 0 and dc != 0 else 1.0
                new_cost = cost + step_cost

                if new_pos not in self.cost or new_cost < self.cost[new_pos]:
                    self.cost[new_pos] = new_cost
                    self.parent[new_pos] = current
                    heapq.heappush(self.pq, (new_cost, new_pos))

                    if new_pos != self.grid.target and new_pos not in self.grid.explored:
                        self.grid.grid[new_pos[0]][new_pos[1]] = FRONTIER

        return True

# Depth-Limited Search (DLS) implementation
class DLS:
    def __init__(self, grid, limit=7):
        self.grid = grid
        self.limit = limit
        self.stack = []  # (position, depth)
        self.parent = {}
        self.depth = {}

    def start(self):
        start = self.grid.start
        self.stack.append((start, 0))
        self.parent[start] = None
        self.depth[start] = 0
        self.grid.grid[start[0]][start[1]] = FRONTIER

    def step(self):
        if not self.stack:
            self.grid.searching = False
            return False

        current, depth = self.stack.pop()

        # Skip stale entries (node was already reached at a shallower depth)
        if depth > self.depth.get(current, float('inf')):
            return True

        # Mark as explored
        if current != self.grid.start:
            self.grid.grid[current[0]][current[1]] = EXPLORED
        self.grid.explored.add(current)

        # Check if target reached
        if current == self.grid.target:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = self.parent[current]
            path.reverse()
            self.grid.path = path

            # Mark path
            for pos in path:
                if pos != self.grid.start and pos != self.grid.target:
                    self.grid.grid[pos[0]][pos[1]] = PATH
            self.grid.searching = False
            return False

        # Expand neighbors if within depth limit
        if depth < self.limit:
            for dr, dc in reversed(MOVEMENTS):
                new_pos = (current[0] + dr, current[1] + dc)

                if (self.grid.is_valid(new_pos) and
                        (new_pos not in self.depth or depth + 1 < self.depth[new_pos])):

                    self.stack.append((new_pos, depth + 1))
                    self.parent[new_pos] = current
                    self.depth[new_pos] = depth + 1
                    if new_pos != self.grid.target:
                        self.grid.grid[new_pos[0]][new_pos[1]] = FRONTIER

        return True

# Iterative Deepening DFS (IDDFS) implementation
class IDDFS:
    def __init__(self, grid):
        self.grid = grid
        self.current_limit = 0
        self.dls = None
        self.max_limit = GRID_SIZE * GRID_SIZE

    def start(self):
        self.current_limit = 0
        self.dls = DLS(self.grid, self.current_limit)
        self.dls.start()

    def step(self):
        if not self.dls or not self.dls.stack:
            # Increase depth limit
            self.current_limit += 1
            if self.current_limit > self.max_limit:
                self.grid.searching = False
                return False

            # Reset grid for new iteration
            self.reset_grid()
            self.dls = DLS(self.grid, self.current_limit)
            self.dls.start()
            return True

        # Perform DLS step
        result = self.dls.step()

        # Check if target found
        if self.grid.path:
            self.grid.searching = False
            return False

        return result

    def reset_grid(self):
        # Reset explored and frontier cells
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid.grid[i][j] not in [WALL, START, TARGET]:
                    self.grid.grid[i][j] = EMPTY
        self.grid.explored = set()
        self.grid.frontier = []
        self.grid.path = []

        # Reset start and target
        self.grid.grid[self.grid.start[0]][self.grid.start[1]] = START
        self.grid.grid[self.grid.target[0]][self.grid.target[1]] = TARGET

# Bidirectional Search implementation
class Bidirectional:
    def __init__(self, grid):
        self.grid = grid
        self.queue_f = deque()
        self.queue_b = deque()
        self.parent_f = {}
        self.parent_b = {}
        self.meeting_point = None
        self.forward_turn = True  # Flag to alternate directions

    def start(self):
        # Forward search from start
        start = self.grid.start
        self.queue_f.append(start)
        self.parent_f[start] = None
        self.grid.grid[start[0]][start[1]] = FRONTIER

        # Backward search from target
        target = self.grid.target
        self.queue_b.append(target)
        self.parent_b[target] = None
        self.grid.grid[target[0]][target[1]] = FRONTIER

    def step(self):
        # Properly alternate between forward and backward
        if self.forward_turn and self.queue_f:
            result = self._expand_forward()
            self.forward_turn = False
            return result
        elif not self.forward_turn and self.queue_b:
            result = self._expand_backward()
            self.forward_turn = True
            return result
        elif self.queue_f:
            result = self._expand_forward()
            return result
        elif self.queue_b:
            result = self._expand_backward()
            return result
        else:
            self.grid.searching = False
            return False

    def _expand_forward(self):
        current = self.queue_f.popleft()

        # Mark as explored
        if current != self.grid.start:
            self.grid.grid[current[0]][current[1]] = EXPLORED
        self.grid.explored.add(current)

        # Check for meeting point
        if current in self.parent_b:
            self.meeting_point = current
            self.construct_path()
            return False

        # Expand forward
        for dr, dc in MOVEMENTS:
            new_pos = (current[0] + dr, current[1] + dc)

            if (self.grid.is_valid(new_pos) and
                    new_pos not in self.parent_f):

                self.queue_f.append(new_pos)
                self.parent_f[new_pos] = current
                if new_pos != self.grid.target:
                    self.grid.grid[new_pos[0]][new_pos[1]] = FRONTIER

        return True

    def _expand_backward(self):
        current = self.queue_b.popleft()

        # Mark as explored
        if current != self.grid.target:
            self.grid.grid[current[0]][current[1]] = EXPLORED
        self.grid.explored.add(current)

        # Check for meeting point
        if current in self.parent_f:
            self.meeting_point = current
            self.construct_path()
            return False

        # Expand backward
        for dr, dc in MOVEMENTS:
            new_pos = (current[0] + dr, current[1] + dc)

            if (self.grid.is_valid(new_pos) and
                    new_pos not in self.parent_b):

                self.queue_b.append(new_pos)
                self.parent_b[new_pos] = current
                if new_pos != self.grid.start:
                    self.grid.grid[new_pos[0]][new_pos[1]] = FRONTIER

        return True

    def construct_path(self):
        # Construct path from start to meeting point
        path_f = []
        current = self.meeting_point
        while current is not None:
            path_f.append(current)
            current = self.parent_f[current]
        path_f.reverse()

        # Construct path from meeting point to target (excluding meeting point)
        path_b = []
        current = self.parent_b[self.meeting_point]
        while current is not None:
            path_b.append(current)
            current = self.parent_b[current]

        # Combine paths
        self.grid.path = path_f + path_b

        # Mark path
        for pos in self.grid.path:
            if pos != self.grid.start and pos != self.grid.target:
                self.grid.grid[pos[0]][pos[1]] = PATH
        self.grid.searching = False


class PathfinderGUI:
    def __init__(self):
        self.grid = Grid()
        self.algorithm = None
        self.fig, (self.ax, self.button_ax) = plt.subplots(1, 2, figsize=(14, 8),
                                                           gridspec_kw={'width_ratios': [1, 0.3]})
        self.setup_ui()

    def setup_ui(self):
        # Setup main grid
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Pathfinder Visualization")

        # Setup button panel
        self.button_ax.axis('off')
        self.button_ax.set_facecolor('lightgray')

        # Create buttons
        button_props = dict(transform=self.button_ax.transAxes, fontsize=10)

        algorithms = ['BFS', 'DFS', 'UCS', 'DLS', 'IDDFS', 'Bidirectional']
        y_positions = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35]

        self.buttons = []
        for algo, y in zip(algorithms, y_positions):
            btn = Button(plt.axes([0.72, y, 0.2, 0.05]), algo, color='lightblue')
            btn.on_clicked(lambda event, a=algo: self.start_algorithm(a))
            self.buttons.append(btn)

        # Control buttons
        self.reset_btn = Button(plt.axes([0.72, 0.25, 0.2, 0.05]), 'Reset', color='lightgreen')
        self.reset_btn.on_clicked(lambda event: self.reset())

        self.clear_btn = Button(plt.axes([0.72, 0.18, 0.2, 0.05]), 'Clear Walls', color='orange')
        self.clear_btn.on_clicked(lambda event: self.clear_walls())

        self.step_btn = Button(plt.axes([0.72, 0.11, 0.2, 0.05]), 'Step', color='yellow')
        self.step_btn.on_clicked(lambda event: self.step())

        self.animate_btn = Button(plt.axes([0.72, 0.04, 0.2, 0.05]), 'Animate', color='pink')
        self.animate_btn.on_clicked(lambda event: self.animate())

        # Legend
        self.create_legend()

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Initial draw
        self.update_display()

    def create_legend(self):
        legend_ax = plt.axes([0.72, 0.92, 0.2, 0.07])
        legend_ax.axis('off')

        items = [
            ('white', 'Empty'), ('gray', 'Wall'), ('lightgreen', 'Start'),
            ('salmon', 'Target'), ('lightblue', 'Explored'), ('yellow', 'Frontier'),
            ('purple', 'Path')
        ]

        for i, (color, label) in enumerate(items[:4]):
            rect = plt.Rectangle((0.05, 0.75 - i * 0.25), 0.1, 0.1,
                                 facecolor=color, edgecolor='black', transform=legend_ax.transAxes)
            legend_ax.add_patch(rect)
            legend_ax.text(0.2, 0.8 - i * 0.25, label, transform=legend_ax.transAxes, fontsize=8)

        for i, (color, label) in enumerate(items[4:]):
            rect = plt.Rectangle((0.55, 0.75 - i * 0.25), 0.1, 0.1,
                                 facecolor=color, edgecolor='black', transform=legend_ax.transAxes)
            legend_ax.add_patch(rect)
            legend_ax.text(0.7, 0.8 - i * 0.25, label, transform=legend_ax.transAxes, fontsize=8)

    def start_algorithm(self, algo_name):
        # Only clear search state (explored/frontier/path), preserve walls
        self.clear_search()
        self.grid.searching = True

        if algo_name == 'BFS':
            self.algorithm = BFS(self.grid)
        elif algo_name == 'DFS':
            self.algorithm = DFS(self.grid)
        elif algo_name == 'UCS':
            self.algorithm = UCS(self.grid)
        elif algo_name == 'DLS':
            self.algorithm = DLS(self.grid, limit=7)
        elif algo_name == 'IDDFS':
            self.algorithm = IDDFS(self.grid)
        elif algo_name == 'Bidirectional':
            self.algorithm = Bidirectional(self.grid)

        self.algorithm.start()
        self.update_display()

    def reset(self):
        self.grid.reset()
        self.algorithm = None
        self.update_display()

    def clear_walls(self):
        self.grid.clear_walls()
        self.update_display()

    def clear_search(self):
        """Clear only search state (explored/frontier/path), preserving walls."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid.grid[i][j] in [EXPLORED, FRONTIER, PATH]:
                    self.grid.grid[i][j] = EMPTY
        self.grid.explored = set()
        self.grid.frontier = []
        self.grid.path = []
        self.grid.searching = False
        self.algorithm = None
        # Ensure start and target are still marked
        self.grid.grid[self.grid.start[0]][self.grid.start[1]] = START
        self.grid.grid[self.grid.target[0]][self.grid.target[1]] = TARGET
        self.update_display()

    def step(self):
        if self.algorithm and self.grid.searching:
            self.algorithm.step()
            self.update_display()

    def animate(self):
        if not self.algorithm or not self.grid.searching:
            return

        while self.grid.searching:
            self.algorithm.step()
            self.update_display()
            plt.pause(self.grid.step_delay)

    def on_click(self, event):
        if event.inaxes == self.ax and not self.grid.searching:
            col = int(event.xdata)
            row = GRID_SIZE - 1 - int(event.ydata)

            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                if event.button == 1:  # Left click: toggle walls
                    if self.grid.grid[row][col] == EMPTY:
                        self.grid.grid[row][col] = WALL
                    elif self.grid.grid[row][col] == WALL:
                        self.grid.grid[row][col] = EMPTY
                elif event.button == 3:  # Right click: set target
                    if (row, col) != self.grid.start:
                        # Clear old target
                        old = self.grid.target
                        self.grid.grid[old[0]][old[1]] = EMPTY
                        # Set new target
                        self.grid.target = (row, col)
                        self.grid.grid[row][col] = TARGET
                elif event.button == 2:  # Middle click: set start
                    if (row, col) != self.grid.target:
                        # Clear old start
                        old = self.grid.start
                        self.grid.grid[old[0]][old[1]] = EMPTY
                        # Set new start
                        self.grid.start = (row, col)
                        self.grid.grid[row][col] = START
                self.update_display()

    def update_display(self):
        self.ax.clear()

        # Draw grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = COLORS[self.grid.grid[i][j]]
                rect = plt.Rectangle((j, GRID_SIZE - 1 - i), 1, 1,
                                     facecolor=color, edgecolor='black', linewidth=0.5)
                self.ax.add_patch(rect)

                # Add labels
                if self.grid.grid[i][j] == START:
                    self.ax.text(j + 0.5, GRID_SIZE - 1 - i + 0.5, 'S',
                                 ha='center', va='center', fontsize=12, fontweight='bold')
                elif self.grid.grid[i][j] == TARGET:
                    self.ax.text(j + 0.5, GRID_SIZE - 1 - i + 0.5, 'T',
                                 ha='center', va='center', fontsize=12, fontweight='bold')

        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Update title
        if self.algorithm and self.grid.searching:
            self.ax.set_title(f"Running: {type(self.algorithm).__name__}", color='blue')
        elif self.algorithm and not self.grid.searching and self.grid.path:
            self.ax.set_title("Path Found!", color='green')
        elif self.algorithm and not self.grid.searching:
            self.ax.set_title("No Path Found", color='red')
        else:
            self.ax.set_title("Left-click: walls | Right-click: target | Middle-click: start")

        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


if __name__ == "__main__":
    gui = PathfinderGUI()
    gui.run()