import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import random, heapq, time
from colorsys import hsv_to_rgb

# ----------------------------
#   CONFIGURATION CONSTANTS
# ----------------------------
ROWS, COLS = 25, 25
CELL_SIZE = 25

DARK_BG     = "#2b2b2b"
ACCENT_1    = "#6C5CE7"
ACCENT_2    = "#00B894"
ACCENT_3    = "#FF7675"
OPTIMAL_PATH= "#FFD700"
TEXT_WHITE  = "#ECF0F1"
PANEL_BG    = "#3d3d3d"
WALL_COLOR  = "#4a4a4a"
PATH_COLOR  = "#2d2d2d"

# ----------------------------
#   MODERN MAZE CLASS
# ----------------------------
class ModernMaze(tk.Canvas):
    def __init__(self, parent, rows, cols, cell_size):
        super().__init__(parent,
                         width=cols * cell_size,
                         height=rows * cell_size,
                         bg=PATH_COLOR,
                         highlightthickness=0)
        self.cell_size = cell_size
        self.rows = rows
        self.cols = cols
        self.start = (rows - 2, 1)
        self.goal = (1, cols - 2)
        self.generate_maze()
        self.draw_maze()

    def generate_maze(self):
        # Use iterative DFS to avoid recursion depth issues.
        self.grid = [[{'n': True, 'e': True, 's': True, 'w': True}
                      for _ in range(self.cols)] for _ in range(self.rows)]
        visited = set()
        stack = []
        start_cell = (0, 0)
        visited.add(start_cell)
        stack.append(start_cell)

        while stack:
            r, c = stack[-1]
            # Collect all unvisited neighbors
            neighbors = []
            for dr, dc, d in [(-1, 0, 'n'), (1, 0, 's'), (0, -1, 'w'), (0, 1, 'e')]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and ((nr, nc) not in visited):
                    neighbors.append((nr, nc, d))
            if neighbors:
                nr, nc, d = random.choice(neighbors)
                self.grid[r][c][d] = False
                opp = {'n': 's', 's': 'n', 'e': 'w', 'w': 'e'}[d]
                self.grid[nr][nc][opp] = False
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # Reinforce boundaries
        for r in range(self.rows):
            self.grid[r][0]['w'] = True
            self.grid[r][-1]['e'] = True
        for c in range(self.cols):
            self.grid[0][c]['n'] = True
            self.grid[-1][c]['s'] = True

    def draw_maze(self):
        self.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c * self.cell_size, r * self.cell_size
                cell = self.grid[r][c]
                if cell['n']:
                    self.create_line(x, y, x + self.cell_size, y, fill=WALL_COLOR, width=2)
                if cell['s']:
                    self.create_line(x, y + self.cell_size, x + self.cell_size, y + self.cell_size, fill=WALL_COLOR, width=2)
                if cell['e']:
                    self.create_line(x + self.cell_size, y, x + self.cell_size, y + self.cell_size, fill=WALL_COLOR, width=2)
                if cell['w']:
                    self.create_line(x, y, x, y + self.cell_size, fill=WALL_COLOR, width=2)
        # start & goal markers
        self.create_oval(*self.get_center(self.start), fill=ACCENT_2, outline="")
        self.create_oval(*self.get_center(self.goal), fill=ACCENT_3, outline="")

    def get_center(self, pos):
        r, c = pos
        x = c * self.cell_size + self.cell_size // 2
        y = r * self.cell_size + self.cell_size // 2
        return (x - 8, y - 8, x + 8, y + 8)

# ----------------------------
#   A* SOLVER (Unidirectional)
# ----------------------------
class AStarSolver:
    def __init__(self, maze):
        self.maze = maze
        self.start, self.goal = maze.start, maze.goal
        self.open_set = []
        self.open_set_members = set()
        heapq.heappush(self.open_set, (self.heuristic(self.start), self.start))
        self.open_set_members.add(self.start)
        self.came_from = {}
        self.g_score = {(r, c): float('inf') for r in range(maze.rows) for c in range(maze.cols)}
        self.g_score[self.start] = 0
        self.explored = set()
        self.nodes_expanded = 0
        self.optimal_path = []

    def heuristic(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def get_neighbors(self, pos):
        r, c = pos
        for d, (dr, dc) in {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1)}.items():
            if not self.maze.grid[r][c][d]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols:
                    yield (nr, nc)

    def step(self):
        if not self.open_set:
            return False
        current = heapq.heappop(self.open_set)[1]
        self.open_set_members.discard(current)
        self.nodes_expanded += 1
        self.explored.add(current)
        if current == self.goal:
            self.reconstruct_path(current)
            return True
        for neigh in self.get_neighbors(current):
            tentative_g = self.g_score[current] + 1
            if tentative_g < self.g_score[neigh]:
                self.came_from[neigh] = current
                self.g_score[neigh] = tentative_g
                f = tentative_g + self.heuristic(neigh)
                if neigh not in self.explored and neigh not in self.open_set_members:
                    heapq.heappush(self.open_set, (f, neigh))
                    self.open_set_members.add(neigh)
        return None

    def reconstruct_path(self, cur):
        path = [cur]
        while cur in self.came_from:
            cur = self.came_from[cur]
            path.append(cur)
        path.reverse()
        self.optimal_path = path

# ----------------------------
#   BIDIRECTIONAL A* SOLVER
# ----------------------------
class BiAStarSolver:
    def __init__(self, maze):
        self.maze = maze
        self.start, self.goal = maze.start, maze.goal

        # Forward search
        self.open_set_f = []
        self.open_set_members_f = set()
        self.came_from_f = {}
        self.g_f = {(r, c): float('inf') for r in range(maze.rows) for c in range(maze.cols)}
        self.g_f[self.start] = 0
        heapq.heappush(self.open_set_f, (self.heuristic(self.start, self.goal), self.start))
        self.open_set_members_f.add(self.start)
        self.explored_f = set()

        # Backward search
        self.open_set_b = []
        self.open_set_members_b = set()
        self.came_from_b = {}
        self.g_b = {(r, c): float('inf') for r in range(maze.rows) for c in range(maze.cols)}
        self.g_b[self.goal] = 0
        heapq.heappush(self.open_set_b, (self.heuristic(self.goal, self.start), self.goal))
        self.open_set_members_b.add(self.goal)
        self.explored_b = set()

        self.meeting_point = None
        self.optimal_path = []
        self.nodes_expanded = 0

    def heuristic(self, p, t):
        return abs(p[0] - t[0]) + abs(p[1] - t[1])

    def get_neighbors(self, pos):
        r, c = pos
        for d, (dr, dc) in {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1)}.items():
            if not self.maze.grid[r][c][d]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols:
                    yield (nr, nc)

    def step(self):
        # Expand one node from forward search
        if self.open_set_f:
            if self._expand(self.open_set_f, self.open_set_members_f, self.g_f, self.came_from_f,
                             self.explored_f, self.goal, forward=True):
                return True
        # Expand one node from backward search
        if self.open_set_b:
            if self._expand(self.open_set_b, self.open_set_members_b, self.g_b, self.came_from_b,
                             self.explored_b, self.start, forward=False):
                return True
        return None

    def _expand(self, open_set, open_set_members, g_score, came_from, explored, target, forward):
        current = heapq.heappop(open_set)[1]
        open_set_members.discard(current)
        self.nodes_expanded += 1
        explored.add(current)
        if current in self.explored_f and current in self.explored_b:
            self.meeting_point = current
            self._reconstruct()
            return True
        for neigh in self.get_neighbors(current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score[neigh]:
                came_from[neigh] = current
                g_score[neigh] = tentative_g
                f_score = tentative_g + self.heuristic(neigh, target)
                if neigh not in open_set_members:
                    heapq.heappush(open_set, (f_score, neigh))
                    open_set_members.add(neigh)
                if forward and neigh in self.explored_b:
                    self.meeting_point = neigh
                    self._reconstruct()
                    return True
                if not forward and neigh in self.explored_f:
                    self.meeting_point = neigh
                    self._reconstruct()
                    return True
        return False

    def _reconstruct(self):
        # Reconstruct forward path
        forward_path = []
        node = self.meeting_point
        while node in self.came_from_f:
            forward_path.append(node)
            node = self.came_from_f[node]
        forward_path.append(self.start)
        forward_path.reverse()
        # Reconstruct backward path
        backward_path = []
        node = self.meeting_point
        while node in self.came_from_b:
            node = self.came_from_b[node]
            backward_path.append(node)
        self.optimal_path = forward_path + backward_path

# ----------------------------
#   MAIN TKINTER UI (Same UI as Before)
# ----------------------------
class ModernUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A* Pathfinding Visualizer")
        self.geometry("1280x800")
        self.configure(bg=DARK_BG)

        self.simulation_job = None
        self.start_time = None
        self.bi_start_time = None
        self.maze = None
        self.solver_mode = tk.StringVar(value="Bidirectional")
        self.solver_mode.trace("w", lambda *a: self.reset_solver(soft=True))
        self.uni_stats = None
        self.bi_stats = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=5)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self.create_stats_panel()
        self.create_maze_panel()
        self.create_control_panel()
        self.create_comparison_panel()

        self.speed = 100
        self.simulating = False

        self.reset_simulation(new_maze=True)

    def create_stats_panel(self):
        panel = ttk.LabelFrame(self, text="Statistics", style="Panel.TFrame")
        panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.stats_text = scrolledtext.ScrolledText(panel, width=25, height=18,
                                                     font=("Fira Code", 10),
                                                     bg=PANEL_BG, fg=TEXT_WHITE)
        self.stats_text.pack(fill="both", expand=True)

    def create_maze_panel(self):
        container = ttk.LabelFrame(self, text="Maze", style="Panel.TFrame")
        container.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        container.grid_propagate(False)
        self.maze_container = container

    def create_control_panel(self):
        panel = ttk.LabelFrame(self, text="Controls", style="Panel.TFrame")
        panel.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        ttk.Label(panel, text="Solver Mode", style="Light.TLabel").pack(anchor="w", pady=5)
        ttk.Radiobutton(panel, text="Bidirectional", variable=self.solver_mode, value="Bidirectional").pack(anchor="w")
        ttk.Radiobutton(panel, text="Unidirectional", variable=self.solver_mode, value="Unidirectional").pack(anchor="w")
        ttk.Label(panel, text="Simulation Speed", style="Light.TLabel").pack(fill="x", pady=5)
        self.speed_slider = ttk.Scale(panel, from_=10, to=500,
                                      command=lambda v: self.set_speed(int(float(v))))
        self.speed_slider.set(100)
        self.speed_slider.pack(fill="x", pady=5)
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill="x", pady=10)
        self.start_btn = ttk.Button(btn_frame, text="▶ START", style="Accent.TButton",
                                    command=self.toggle_simulation)
        self.start_btn.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="⏹ RESET", style="Danger.TButton",
                   command=lambda: self.reset_simulation(new_maze=True)).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="↻ RESTART", style="Success.TButton",
                   command=self.restart_simulation).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="⏩ STEP", style="Success.TButton",
                   command=self.step_simulation).pack(fill="x", pady=2)

    def create_comparison_panel(self):
        frame = ttk.Frame(self, style="Panel.TFrame")
        frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self.uni_frame = ttk.LabelFrame(frame, text="Unidirectional (A*)", style="Panel.TFrame")
        self.uni_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.uni_label = ttk.Label(self.uni_frame, text="No data yet.", style="Light.TLabel")
        self.uni_label.pack(fill="x", pady=5)
        self.bi_frame = ttk.LabelFrame(frame, text="Bidirectional (A*)", style="Panel.TFrame")
        self.bi_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.bi_label = ttk.Label(self.bi_frame, text="No data yet.", style="Light.TLabel")
        self.bi_label.pack(fill="x", pady=5)

    def update_comparison_panel(self):
        if self.uni_stats:
            ne, pl, et = self.uni_stats
            self.uni_label.config(text=f"Nodes Expanded: {ne}\nOptimal Path: {pl}\nTime: {et:.4f} s")
        else:
            self.uni_label.config(text="No data yet.")
        if self.bi_stats:
            ne, pl, et = self.bi_stats
            self.bi_label.config(text=f"Nodes Expanded: {ne}\nOptimal Path: {pl}\nTime: {et:.4f} s")
        else:
            self.bi_label.config(text="No data yet.")

    def set_speed(self, s):
        self.speed = max(10, min(500, s))

    def reset_simulation(self, new_maze=False, soft=False):
        if self.simulation_job:
            self.after_cancel(self.simulation_job)
            self.simulation_job = None
        self.simulating = False
        self.start_time = None
        self.bi_start_time = None
        if not soft:
            self.uni_stats = None
            self.bi_stats = None
            self.update_comparison_panel()
        if new_maze or not self.maze:
            for w in self.maze_container.winfo_children():
                w.destroy()
            self.maze = ModernMaze(self.maze_container, ROWS, COLS, CELL_SIZE)
            self.maze.pack(padx=20, pady=20)
        self.reset_solver()
        self.update_stats()
        self.start_btn.config(text="▶ START", state="normal")

    def reset_solver(self, soft=False):
        if not self.maze:
            return
        if self.solver_mode.get() == "Bidirectional":
            self.solver = BiAStarSolver(self.maze)
        else:
            self.solver = AStarSolver(self.maze)
        self.maze.delete("path")
        self.simulating = False
        self.start_btn.config(text="▶ START")
        self.update_stats()

    def restart_simulation(self):
        if self.simulation_job:
            self.after_cancel(self.simulation_job)
            self.simulation_job = None
        if self.solver_mode.get() == "Bidirectional":
            self.solver = BiAStarSolver(self.maze)
            self.bi_start_time = time.time()
        else:
            self.solver = AStarSolver(self.maze)
            self.start_time = time.time()
        self.maze.delete("path")
        self.update_stats()
        self.simulating = True
        self.start_btn.config(text="⏸ PAUSE")
        self.run_simulation()

    def toggle_simulation(self):
        self.simulating = not self.simulating
        self.start_btn.config(text="⏸ PAUSE" if self.simulating else "▶ START")
        if self.simulating:
            if self.solver_mode.get() == "Bidirectional" and self.bi_start_time is None:
                self.bi_start_time = time.time()
            if self.solver_mode.get() == "Unidirectional" and self.start_time is None:
                self.start_time = time.time()
            self.run_simulation()

    def run_simulation(self):
        if not self.simulating:
            return
        result = self.solver.step()
        # Update UI every 10 steps to reduce overhead
        if self.solver.nodes_expanded % 10 == 0:
            self.draw_paths()
            self.update_stats()
        if result:
            self.simulating = False
            if isinstance(self.solver, AStarSolver):
                elapsed = time.time() - self.start_time
            else:
                elapsed = time.time() - self.bi_start_time
            mode = "Unidirectional" if isinstance(self.solver, AStarSolver) else "Bidirectional"
            messagebox.showinfo("Path Found!",
                                f"[{mode}] Optimal path found with {len(self.solver.optimal_path)} steps!\n"
                                f"Nodes Expanded: {self.solver.nodes_expanded}\n"
                                f"Time: {elapsed:.4f} seconds")
            self.start_btn.config(text="▶ START")
            self.maze.delete("path")
            for i in range(1, len(self.solver.optimal_path)):
                x1, y1 = self.maze.get_center(self.solver.optimal_path[i-1])[:2]
                x2, y2 = self.maze.get_center(self.solver.optimal_path[i])[:2]
                self.maze.create_line(x1+8, y1+8, x2+8, y2+8,
                                      fill=OPTIMAL_PATH, width=3, tag="path")
            stats = (self.solver.nodes_expanded, len(self.solver.optimal_path), elapsed)
            if isinstance(self.solver, AStarSolver):
                self.uni_stats = stats
            else:
                self.bi_stats = stats
            self.update_comparison_panel()
        elif not ((hasattr(self.solver, 'open_set') and self.solver.open_set) or
                  (hasattr(self.solver, 'open_set_f') and (self.solver.open_set_f or self.solver.open_set_b))):
            self.simulating = False
            messagebox.showwarning("No Path", "No valid path exists!")
        else:
            self.simulation_job = self.after(self.speed, self.run_simulation)

    def step_simulation(self):
        self.solver.step()
        self.draw_paths()
        self.update_stats()

    def draw_paths(self):
        self.maze.delete("path")
        def draw(came_from, offset=0):
            hue = (self.solver.nodes_expanded + offset) % 100 / 100
            for node, parent in came_from.items():
                r, g, b = hsv_to_rgb(hue, 0.9, 0.9)
                col = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                x1, y1 = self.maze.get_center(parent)[:2]
                x2, y2 = self.maze.get_center(node)[:2]
                self.maze.create_line(x1+8, y1+8, x2+8, y2+8, fill=col, width=2, tag="path")
        if isinstance(self.solver, AStarSolver):
            draw(self.solver.came_from)
        else:
            draw(self.solver.came_from_f, offset=0)
            draw(self.solver.came_from_b, offset=50)

    def update_stats(self):
        self.stats_text.config(state="normal")
        self.stats_text.delete(1.0, tk.END)
        mode = "Bidirectional" if self.solver_mode.get() == "Bidirectional" else "Unidirectional"
        self.stats_text.insert(tk.END, "=== Simulation Stats ===\n")
        self.stats_text.insert(tk.END, f"Solver Mode: {mode}\n")
        if hasattr(self.solver, 'nodes_expanded'):
            self.stats_text.insert(tk.END, f"Nodes Expanded: {self.solver.nodes_expanded}\n")
        if hasattr(self.solver, 'open_set'):
            self.stats_text.insert(tk.END, f"Open Nodes: {len(self.solver.open_set)}\n")
        else:
            on = len(self.solver.open_set_f) + len(self.solver.open_set_b)
            self.stats_text.insert(tk.END, f"Open Nodes: {on}\n")
        self.stats_text.insert(tk.END, f"Optimal Path Length: {len(getattr(self.solver, 'optimal_path', []))}\n")
        self.stats_text.config(state="disabled")

if __name__ == "__main__":
    app = ModernUI()
    app.mainloop()
