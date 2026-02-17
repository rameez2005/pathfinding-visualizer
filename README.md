# pathfinding-visualizer

# Pathfinding Algorithm Visualizer

An interactive grid-based pathfinding visualizer built with Python and Matplotlib. Visualize how different search algorithms explore a grid and find the shortest path from start to target, with support for custom wall placement and step-by-step animation.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-TkAgg-orange.svg)

## Algorithms Implemented

| Algorithm | Description |
|---|---|
| **BFS** (Breadth-First Search) | Explores all neighbors at the current depth before moving deeper. Guarantees the shortest path in unweighted graphs. |
| **DFS** (Depth-First Search) | Explores as far as possible along each branch before backtracking. Does not guarantee the shortest path. |
| **UCS** (Uniform Cost Search) | Expands the least-cost node first. Handles weighted edges (diagonal moves cost 1.4, straight moves cost 1.0). |
| **DLS** (Depth-Limited Search) | DFS with a maximum depth limit to prevent infinite exploration. |
| **IDDFS** (Iterative Deepening DFS) | Repeatedly runs DLS with increasing depth limits (0, 1, 2, ...) until the target is found. Combines BFS's completeness with DFS's space efficiency. |
| **Bidirectional Search** | Runs BFS simultaneously from the start and target, meeting in the middle for faster convergence. |

## Features

- **Interactive Wall Placement** — Click on the grid to add/remove walls before running an algorithm
- **Step-by-Step Execution** — Step through the algorithm one node at a time
- **Animated Visualization** — Watch the algorithm explore the grid in real-time
- **Color-Coded Cells** — Easily distinguish between explored nodes, frontier, path, walls, start, and target
- **6-Directional Movement** — Supports Up, Right, Down, Down-Right, Left, and Up-Left movements
- **Persistent Walls** — Manually placed walls are preserved when switching between algorithms

## Grid Legend

| Color | Meaning |
|---|---|
| ⬜ White | Empty cell |
| 🔲 Gray | Wall (impassable) |
| 🟩 Green | Start position (S) |
| 🟥 Salmon | Target position (T) |
| 🟦 Light Blue | Explored node |
| 🟨 Yellow | Frontier (queued for exploration) |
| 🟪 Purple | Final path |

## Requirements

- Python 3.x
- NumPy
- Matplotlib (with TkAgg backend)

## Installation

```bash
# Clone the repository
git clone https://github.com/rameez2005/pathfinding-visualizer.git
cd pathfinding-visualizer

# Install dependencies
pip install numpy matplotlib
```

## Usage

```bash
python pathfinder.py
```

### Controls

| Button | Action |
|---|---|
| **BFS / DFS / UCS / DLS / IDDFS / Bidirectional** | Start the selected algorithm |
| **Reset** | Reset the grid to its default state (restores default walls) |
| **Clear Walls** | Remove all walls from the grid |
| **Step** | Execute one step of the current algorithm |
| **Animate** | Run the algorithm continuously with animation |
| **Left Click** on grid | Toggle wall on/off at that cell |

## Grid Configuration

- **Grid Size:** 10×10
- **Start Position:** (2, 0) — marked with "S"
- **Target Position:** (7, 7) — marked with "T"
- **Default Walls:** Pre-configured L-shaped wall barrier

## Project Structure

```
pathfinding-visualizer/
├── pathfinder.py       # Main application file
├── README.md           # This file

```

## How It Works

1. The grid initializes with a start node, target node, and default walls
2. Click on empty cells to add custom walls, or click walls to remove them
3. Select an algorithm to begin the search
4. Use **Step** to advance one iteration at a time, or **Animate** to watch it run
5. The algorithm explores the grid — explored cells turn blue, frontier cells turn yellow
6. When the target is found, the path is highlighted in purple

