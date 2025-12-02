# streamlit_app.py
# The Manhattan Project: Visualizing Sudoku Physics
# Architect: Herudero 
# AI Engineer: GPT-5.1 Thinking, Gemini 3

import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

# --- GLOBAL CONSTANTS: DISTANCES, UNITS, PEERS ---

# Manhattan distance from center cell (4,4 in 0-index → (5,5) for humans)
rows, cols = np.indices((9, 9))
DIST = np.abs(rows - 4) + np.abs(cols - 4)

# Standard Sudoku units (9 rows, 9 cols, 9 boxes)
UNITS = []
for i in range(9):
    UNITS.append([(i, c) for c in range(9)])  # rows
    UNITS.append([(r, i) for r in range(9)])  # cols

for r in range(0, 9, 3):
    for c in range(0, 9, 3):
        UNITS.append([(r + dr, c + dc) for dr in range(3) for dc in range(3)])  # boxes

# PEERS: for each cell, all cells in same row/col/box
PEERS = {}
for r in range(9):
    for c in range(9):
        p = set()
        for u in UNITS:
            if (r, c) in u:
                p.update(u)
        p.discard((r, c))
        PEERS[(r, c)] = list(p)


# --- SOLVER CLASS -----------------------------------------------------------

class ManhattanSolver:
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)
        self.candidates = {}
        self.manhattan_status = {}
        self.log = []
        self.initial_setup()

    def initial_setup(self):
        # Initialize candidates: 1..9 for empty cells, none for filled
        for r in range(9):
            for c in range(9):
                if self.grid[r, c] != 0:
                    self.candidates[(r, c)] = set()
                else:
                    self.candidates[(r, c)] = set(range(1, 10))

        # Propagate initial constraints from givens
        for r in range(9):
            for c in range(9):
                if self.grid[r, c] != 0:
                    self.eliminate_peers(r, c, self.grid[r, c])

    def eliminate_peers(self, r, c, val):
        # Remove val from all peers' candidate sets
        for pr, pc in PEERS[(r, c)]:
            if (pr, pc) in self.candidates and val in self.candidates[(pr, pc)]:
                self.candidates[(pr, pc)].discard(val)

    def place(self, r, c, val):
        # Place a digit and propagate constraints
        self.grid[r, c] = val
        self.candidates[(r, c)] = set()
        self.eliminate_peers(r, c, val)
        self.log.append(f"Placed {val} at R{r+1}C{c+1}")

    # --- SIP-A: Singles -----------------------------------------------------

    def apply_naked_singles(self):
        # Cells with exactly one candidate
        progress = False
        for (r, c), vals in list(self.candidates.items()):
            if self.grid[r, c] == 0 and len(vals) == 1:
                val = next(iter(vals))
                self.place(r, c, val)
                progress = True
        return progress

    def apply_hidden_singles(self):
        # Unique candidate for a digit in a unit (row/col/box)
        progress = False
        for unit in UNITS:
            counts = Counter()
            locs = {}
            for r, c in unit:
                if self.grid[r, c] == 0:
                    for val in self.candidates[(r, c)]:
                        counts[val] += 1
                        locs[val] = (r, c)
            for val, count in counts.items():
                if count == 1:
                    r, c = locs[val]
                    self.place(r, c, val)
                    self.log.append(f"Hidden Single: {val} at R{r+1}C{c+1}")
                    progress = True
        return progress

    # --- SIP-C: Naked pairs & triples --------------------------------------

    def apply_naked_pairs(self):
        progress = False
        for unit in UNITS:
            pairs_map = defaultdict(list)
            for r, c in unit:
                if self.grid[r, c] == 0 and len(self.candidates[(r, c)]) == 2:
                    key = tuple(sorted(list(self.candidates[(r, c)])))
                    pairs_map[key].append((r, c))

            for vals, cells in pairs_map.items():
                if len(cells) == 2:
                    v1, v2 = vals
                    pair_locs = set(cells)
                    for r, c in unit:
                        if (r, c) not in pair_locs and self.grid[r, c] == 0:
                            changed = False
                            if v1 in self.candidates[(r, c)]:
                                self.candidates[(r, c)].discard(v1)
                                changed = True
                            if v2 in self.candidates[(r, c)]:
                                self.candidates[(r, c)].discard(v2)
                                changed = True
                            if changed:
                                progress = True
        return progress

    def apply_naked_triples(self):
        """SIP-C: Naked triples elimination."""
        progress = False
        for unit in UNITS:
            # Collect cells with 2 or 3 candidates
            candidates_in_unit = []
            for r, c in unit:
                if self.grid[r, c] == 0 and 2 <= len(self.candidates[(r, c)]) <= 3:
                    candidates_in_unit.append(((r, c), self.candidates[(r, c)]))

            if len(candidates_in_unit) < 3:
                continue

            for triple in combinations(candidates_in_unit, 3):
                cells = [cell for cell, _ in triple]
                all_vals = set()
                for _, vals in triple:
                    all_vals.update(vals)

                # If exactly 3 values span these 3 cells, it's a naked triple
                if len(all_vals) == 3:
                    triple_locs = set(cells)
                    for r, c in unit:
                        if (r, c) not in triple_locs and self.grid[r, c] == 0:
                            changed = False
                            for val in all_vals:
                                if val in self.candidates[(r, c)]:
                                    self.candidates[(r, c)].discard(val)
                                    changed = True
                            if changed:
                                progress = True
        return progress

    # --- SIP-D: Box/line interplay -----------------------------------------

    def apply_box_line_reduction(self):
        """
        SIP-D: Box-to-line (claiming).
        If digit d in a box is confined to a single row/col,
        remove d from that row/col outside the box.
        """
        progress = False
        for box_r in [0, 3, 6]:
            for box_c in [0, 3, 6]:
                box_cells = [(box_r + dr, box_c + dc)
                             for dr in range(3) for dc in range(3)]

                for d in range(1, 10):
                    possible = [(r, c) for r, c in box_cells
                                if self.grid[r, c] == 0 and d in self.candidates[(r, c)]]
                    if not possible:
                        continue

                    # All candidates for d in this box are in one row
                    rows = set(r for r, c in possible)
                    if len(rows) == 1:
                        target_row = list(rows)[0]
                        for c in range(9):
                            if (target_row, c) not in box_cells and self.grid[target_row, c] == 0:
                                if d in self.candidates[(target_row, c)]:
                                    self.candidates[(target_row, c)].discard(d)
                                    self.log.append(
                                        f"Box-Line: Removed {d} from R{target_row+1}C{c+1}"
                                    )
                                    progress = True

                    # All candidates for d in this box are in one column
                    cols = set(c for r, c in possible)
                    if len(cols) == 1:
                        target_col = list(cols)[0]
                        for r in range(9):
                            if (r, target_col) not in box_cells and self.grid[r, target_col] == 0:
                                if d in self.candidates[(r, target_col)]:
                                    self.candidates[(r, target_col)].discard(d)
                                    self.log.append(
                                        f"Box-Line: Removed {d} from R{r+1}C{target_col+1}"
                                    )
                                    progress = True
        return progress

    def apply_line_box_reduction(self):
        """
        SIP-D: Line-to-box (enhanced pointing/claiming).
        If digit d in a row/col is confined to one box, remove d from the rest
        of that box outside the row/col.
        """
        progress = False

        # Rows
        for r in range(9):
            for d in range(1, 10):
                possible = [(r, c) for c in range(9)
                            if self.grid[r, c] == 0 and d in self.candidates[(r, c)]]
                if not possible:
                    continue

                boxes = set((r // 3, c // 3) for _, c in possible)
                if len(boxes) == 1:
                    box_r, box_c = list(boxes)[0]
                    box_r_start, box_c_start = box_r * 3, box_c * 3
                    for dr in range(3):
                        for dc in range(3):
                            cell = (box_r_start + dr, box_c_start + dc)
                            if cell[0] != r and self.grid[cell] == 0:
                                if d in self.candidates[cell]:
                                    self.candidates[cell].discard(d)
                                    self.log.append(
                                        f"Line-Box: Removed {d} from R{cell[0]+1}C{cell[1]+1}"
                                    )
                                    progress = True

        # Columns
        for c in range(9):
            for d in range(1, 10):
                possible = [(r, c) for r in range(9)
                            if self.grid[r, c] == 0 and d in self.candidates[(r, c)]]
                if not possible:
                    continue

                boxes = set((r // 3, c // 3) for r, _ in possible)
                if len(boxes) == 1:
                    box_r, box_c = list(boxes)[0]
                    box_r_start, box_c_start = box_r * 3, box_c * 3
                    for dr in range(3):
                        for dc in range(3):
                            cell = (box_r_start + dr, box_c_start + dc)
                            if cell[1] != c and self.grid[cell] == 0:
                                if d in self.candidates[cell]:
                                    self.candidates[cell].discard(d)
                                    self.log.append(
                                        f"Line-Box: Removed {d} from R{cell[0]+1}C{cell[1]+1}"
                                    )
                                    progress = True
        return progress

    # --- SIP-E: Manhattan invariant (status + pruning) ---------------------

    def compute_manhattan_status(self):
        """
        Compute Manhattan-40 status for each digit without pruning.
        Used only for visualization in the UI.
        """
        self.manhattan_status = {}
        for d in range(1, 10):
            placed_positions = list(zip(*np.where(self.grid == d)))
            current_sum = sum(DIST[r, c] for r, c in placed_positions)
            remaining_count = 9 - len(placed_positions)

            if remaining_count == 0:
                self.manhattan_status[d] = {
                    "status": "Complete",
                    "budget": 0
                }
                continue

            target_sum = 40
            remaining_budget = target_sum - current_sum

            candidate_cells = []
            for r in range(9):
                for c in range(9):
                    if self.grid[r, c] == 0 and d in self.candidates[(r, c)]:
                        candidate_cells.append(((r, c), DIST[r, c]))

            candidate_cells.sort(key=lambda x: x[1])
            costs = [x[1] for x in candidate_cells]

            if len(costs) < remaining_count:
                self.manhattan_status[d] = {
                    "status": "Insufficient candidates",
                    "budget": remaining_budget
                }
                continue

            min_possible = sum(costs[:remaining_count])
            max_possible = sum(costs[-remaining_count:])

            self.manhattan_status[d] = {
                "status": f"Range: [{min_possible}, {max_possible}]",
                "budget": remaining_budget
            }

    def apply_manhattan_invariant(self):
        """
        SIP-E: Physics-inspired global pruning.

        Uses the 40-sum Manhattan invariant as a conservative filter:
        - Does NOT guarantee solving every puzzle.
        - Tries to remove candidate cells where forcing digit d into that cell
          would make it impossible to reach exactly 40 for that digit.
        """
        progress = False
        self.manhattan_status = {}

        for d in range(1, 10):
            placed_positions = list(zip(*np.where(self.grid == d)))
            current_sum = sum(DIST[r, c] for r, c in placed_positions)
            remaining_count = 9 - len(placed_positions)

            if remaining_count == 0:
                self.manhattan_status[d] = {"status": "Complete", "budget": 0}
                continue

            target_sum = 40
            remaining_budget = target_sum - current_sum

            candidate_cells = []
            for r in range(9):
                for c in range(9):
                    if self.grid[r, c] == 0 and d in self.candidates[(r, c)]:
                        candidate_cells.append(((r, c), DIST[r, c]))

            candidate_cells.sort(key=lambda x: x[1])
            costs = [x[1] for x in candidate_cells]

            if len(costs) < remaining_count:
                self.manhattan_status[d] = {
                    "status": "Insufficient candidates",
                    "budget": remaining_budget
                }
                self.log.append(
                    f"CRITICAL: Digit {d} has only {len(costs)} spots for {remaining_count} remaining placements."
                )
                continue

            min_possible = sum(costs[:remaining_count])
            max_possible = sum(costs[-remaining_count:])

            self.manhattan_status[d] = {
                "status": f"Range: [{min_possible}, {max_possible}]",
                "budget": remaining_budget
            }

            if min_possible > remaining_budget:
                self.log.append(
                    f"CRITICAL: Digit {d} budget impossible! (Min {min_possible} > {remaining_budget})"
                )
                continue

            for (r, c), cost in candidate_cells:
                if remaining_count == 1:
                    # Last placement: cost must equal remaining_budget
                    if cost != remaining_budget:
                        if d in self.candidates[(r, c)]:
                            self.candidates[(r, c)].discard(d)
                            self.log.append(
                                f"Manhattan Prune: Removed {d} from R{r+1}C{c+1} "
                                f"(cost {cost} != required {remaining_budget})"
                            )
                            progress = True
                    continue

                other_costs = costs.copy()
                try:
                    other_costs.remove(cost)
                except ValueError:
                    continue

                cheapest_rest = sum(other_costs[:remaining_count - 1])
                if cost + cheapest_rest > remaining_budget:
                    if d in self.candidates[(r, c)]:
                        self.candidates[(r, c)].discard(d)
                        self.log.append(
                            f"Manhattan Prune: Removed {d} from R{r+1}C{c+1} "
                            f"(cost {cost} too expensive for budget {remaining_budget})"
                        )
                        progress = True

        return progress

    # --- Wave controller ----------------------------------------------------

    def solve_step(self):
        """
        One 'wave' of the engine: tries all logic layers in priority order.
        Returns True if any progress was made, False if stalled.
        """
        if self.apply_naked_singles():       return True
        if self.apply_hidden_singles():      return True
        if self.apply_naked_pairs():         return True
        if self.apply_naked_triples():       return True
        if self.apply_box_line_reduction():  return True
        if self.apply_line_box_reduction():  return True
        if self.apply_manhattan_invariant(): return True  # Physics wave
        return False


# --- STREAMLIT UI ----------------------------------------------------------

st.set_page_config(page_title="Manhattan Project - Sudoku Physics", layout="wide")
st.title("🧩 The Manhattan Project: Sudoku Physics")

st.markdown(
    """
**Architect:** Herudero | **Physics Engine:** Manhattan-40 Invariant

This demonstration tests the **Manhattan-40 Invariant** for Sudoku.

Unlike standard solvers that rely only on local logic, this system also tracks a
**Distance Budget** for every digit relative to the center cell (5,5). Each digit must
end up with a total Manhattan distance of exactly 40 from the center.

The engine uses this as a **global consistency check**: if a candidate placement would
force a digit to overspend its remaining budget, that candidate is pruned by the
"laws of geometry".
"""
)

with st.sidebar:
    st.header("Control Panel")

    puzzle_choice = st.selectbox(
        "Select Puzzle",
        ["Empty", "Easy", "Medium", "Hard", "AI Escargot"],
    )

    puzzles = {
        "Empty": np.zeros((9, 9), dtype=int),
        "Easy": np.array([
            [0, 3, 4, 0, 0, 8, 9, 1, 2],
            [0, 7, 2, 1, 9, 0, 0, 4, 8],
            [1, 9, 0, 3, 4, 2, 5, 0, 0],
            [0, 5, 0, 7, 0, 0, 0, 0, 3],
            [0, 2, 6, 0, 5, 3, 0, 0, 0],
            [0, 1, 0, 0, 0, 4, 8, 5, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 7, 4, 0, 0, 0, 3, 5],
            [0, 0, 0, 2, 0, 0, 1, 7, 0],
        ]),
        "Medium": np.array([
            [0, 0, 0, 2, 6, 0, 7, 0, 1],
            [6, 8, 0, 0, 7, 0, 0, 9, 0],
            [1, 9, 0, 0, 0, 4, 5, 0, 0],
            [8, 2, 0, 1, 0, 0, 0, 4, 0],
            [0, 0, 4, 6, 0, 2, 9, 0, 0],
            [0, 5, 0, 0, 0, 3, 0, 2, 8],
            [0, 0, 9, 3, 0, 0, 0, 7, 4],
            [0, 4, 0, 0, 5, 0, 0, 3, 6],
            [7, 0, 3, 0, 1, 8, 0, 0, 0],
        ]),
        "Hard": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 8, 5],
            [0, 0, 1, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 7, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 1, 0, 0],
            [0, 9, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 7, 3],
            [0, 0, 2, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 9],
        ]),
        "AI Escargot": np.array([
            [1, 0, 0, 0, 0, 7, 0, 9, 0],
            [0, 3, 0, 0, 2, 0, 0, 0, 8],
            [0, 0, 9, 6, 0, 0, 5, 0, 0],
            [0, 0, 5, 3, 0, 0, 9, 0, 0],
            [0, 1, 0, 0, 8, 0, 0, 0, 2],
            [6, 0, 0, 0, 0, 4, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 7, 0, 0, 0, 3, 0, 0],
        ]),
    }

    if st.button("Load Puzzle"):
        st.session_state.grid = puzzles[puzzle_choice].copy()
        st.session_state.solver = ManhattanSolver(st.session_state.grid)

# First-time init
if "grid" not in st.session_state:
    st.session_state.grid = puzzles["Easy"].copy()
    st.session_state.solver = ManhattanSolver(st.session_state.grid)

col_grid, col_stats = st.columns([1, 1])

with col_grid:
    st.subheader("The Grid")

    grid_display = st.session_state.solver.grid.astype(str)
    grid_display[grid_display == "0"] = "."
    st.dataframe(
        pd.DataFrame(grid_display),
        height=350,
        use_container_width=True,
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Run One Wave (Step)", use_container_width=True):
            changed = st.session_state.solver.solve_step()
            if changed:
                st.success("Wave Propagated!")
            else:
                if np.all(st.session_state.solver.grid > 0):
                    st.balloons()
                    st.success("Solved!")
                else:
                    st.warning("Stalled. No further progress with current logic.")
    with col_b2:
        if st.button("Auto-Solve", use_container_width=True):
            with st.spinner("Calculating Physics..."):
                limit = 300
                while limit > 0:
                    if not st.session_state.solver.solve_step():
                        break
                    limit -= 1
            st.rerun()

with col_stats:
    st.subheader("Manhattan Physics Engine")

    # Recompute status for visualization only
    st.session_state.solver.compute_manhattan_status()

    status_data = []
    for d in range(1, 10):
        stat = st.session_state.solver.manhattan_status.get(
            d, {"status": "Waiting", "budget": "?"}
        )
        status_data.append(
            {
                "Digit": d,
                "Budget Left": stat.get("budget", 0),
                "Range / Status": stat.get("status", ""),
            }
        )

    st.table(pd.DataFrame(status_data))

    with st.expander("Physics Log"):
        for log_entry in reversed(st.session_state.solver.log):
            st.text(log_entry)

st.markdown("---")
st.caption(
    "Active Modules: SIP-A (Singles), SIP-C (Naked Pairs & Triples), "
    "SIP-D (Box/Line Interplay), SIP-E (Manhattan Invariant – experimental global pruning)."
)
