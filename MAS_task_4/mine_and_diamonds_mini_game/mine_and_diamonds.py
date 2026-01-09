from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple

# Configuration of the mine
POSITIONS = [1, 2, 3, 4]  # linear positions
EXITS = {2, 4}  # positions adjacent to an exit (robots can exit from here)
DIAMOND_POSITIONS = {2, 4}  # initial diamond locations
START_POS = {"R1": 1, "R2": 3}  # starting positions


@dataclass
class GameState:
    """Represents the state of the game at a given point."""
    pos: Dict[str, Optional[int]]
    exited: Dict[str, bool]
    utility: Dict[str, int]
    stay_used: Dict[str, bool]
    adjacent_count: Dict[str, int]
    diamonds: Set[int]
    turn: str

    def to_dict(self):
        """Convert to dictionary for easier serialization/display."""
        return {
            "pos": self.pos.copy(),
            "exited": self.exited.copy(),
            "utility": self.utility.copy(),
            "stay_used": self.stay_used.copy(),
            "adjacent_count": self.adjacent_count.copy(),
            "diamonds": self.diamonds.copy(),
            "turn": self.turn
        }


@dataclass
class TreeNode:
    """Represents a node in the game tree."""
    state: GameState
    action: Optional[Tuple[str, str, str]] = None  # (player, action, event_description)
    children: List['TreeNode'] = field(default_factory=list)
    node_id: int = 0
    is_terminal: bool = False
    pruned: bool = False  # Whether this was pruned for being nonsensical

    def __repr__(self):
        return f"TreeNode(id={self.node_id}, action={self.action}, terminal={self.is_terminal})"


class MineGameTree:
    """Builds and stores the complete game tree with strategic pruning."""

    def __init__(self, prune_nonsensical: bool = True):
        self.root = None
        self.node_counter = 0
        self.all_nodes = []  # For easy access to all nodes
        self.terminal_nodes = []
        self.prune_nonsensical = prune_nonsensical
        self.pruned_count = 0

    def initial_state(self) -> GameState:
        """Create the initial game state."""
        return GameState(
            pos={"R1": START_POS["R1"], "R2": START_POS["R2"]},
            exited={"R1": False, "R2": False},
            utility={"R1": 0, "R2": 0},
            stay_used={"R1": False, "R2": False},
            adjacent_count={
                "R1": (1 if START_POS["R1"] in EXITS else 0),
                "R2": (1 if START_POS["R2"] in EXITS else 0)
            },
            diamonds=set(DIAMOND_POSITIONS),
            turn="R1"
        )

    def legal_actions_for(self, player: str, state: GameState) -> List[str]:
        """Return list of legal action symbols for player in current state."""
        if state.exited[player]:
            return []
        pos = state.pos[player]
        actions = []

        # Forced exit: if adjacent_count >= 2, must exit
        if pos in EXITS and state.adjacent_count[player] >= 2:
            return ["E"]

        # Otherwise list normal available actions:
        if pos > min(POSITIONS):
            actions.append("L")
        if pos < max(POSITIONS):
            actions.append("R")
        if not state.stay_used[player]:
            actions.append("S")
        if pos in EXITS:
            actions.append("E")
        return actions

    def filter_sensible_actions(self, player: str, actions: List[str], state: GameState) -> List[str]:
        """
        Filter actions to only those that make strategic sense.
        Pruning rules:
        1. If all diamonds collected and at exit with positive utility -> must exit
        2. If all diamonds collected and not at exit -> only moves towards exit are sensible
        3. If all diamonds collected -> no staying (pointless delay)
        4. If robot has negative utility and could exit -> staying is pointless
        """
        if not self.prune_nonsensical or not actions:
            return actions

        pos = state.pos[player]
        other = "R2" if player == "R1" else "R1"
        all_diamonds_collected = len(state.diamonds) == 0
        at_exit = pos in EXITS
        has_positive_utility = state.utility[player] > 0
        other_exited = state.exited[other]

        # Rule 1: All diamonds collected, at exit, positive utility -> MUST exit
        if all_diamonds_collected and at_exit and has_positive_utility:
            return ["E"]

        # Rule 2: All diamonds collected, at exit, other player exited -> exit (no reason to stay)
        if all_diamonds_collected and at_exit and other_exited:
            return ["E"]

        # Rule 3: All diamonds collected -> filter out staying (pointless)
        if all_diamonds_collected:
            actions = [a for a in actions if a != "S"]

        # Rule 4: All diamonds collected, not at exit -> only moves towards nearest exit
        if all_diamonds_collected and not at_exit and not other_exited:
            # Find direction to nearest exit
            sensible_moves = []
            if "E" in actions:
                sensible_moves.append("E")

            # Calculate distance to exits
            dist_to_exits = {exit_pos: abs(pos - exit_pos) for exit_pos in EXITS}
            nearest_exit = min(dist_to_exits, key=dist_to_exits.get)

            if nearest_exit < pos and "L" in actions:
                sensible_moves.append("L")
            elif nearest_exit > pos and "R" in actions:
                sensible_moves.append("R")
            elif nearest_exit == pos and "E" in actions:
                sensible_moves.append("E")

            if sensible_moves:
                return sensible_moves

        # Rule 5: If other robot exited and all diamonds collected, only exit if at exit
        if other_exited and all_diamonds_collected:
            if at_exit:
                return ["E"]
            # Otherwise move toward exit
            sensible_moves = []
            dist_to_exits = {exit_pos: abs(pos - exit_pos) for exit_pos in EXITS}
            nearest_exit = min(dist_to_exits, key=dist_to_exits.get)

            if nearest_exit < pos and "L" in actions:
                sensible_moves.append("L")
            elif nearest_exit > pos and "R" in actions:
                sensible_moves.append("R")

            if sensible_moves:
                return sensible_moves

        return actions

    def apply_action(self, player: str, action: str, state: GameState) -> Tuple[GameState, str]:
        """Apply action for player on a deep copy of state and return (new_state, event_description)."""
        s = GameState(**{k: deepcopy(v) for k, v in state.to_dict().items()})

        if s.exited[player]:
            return s, "no-op"

        pos = s.pos[player]
        other = "R2" if player == "R1" else "R1"
        event = []

        if action == "S":
            s.stay_used[player] = True
            event.append(f"{player} stays at {pos}")
        elif action == "E":
            s.utility[player] += 50
            s.exited[player] = True
            s.pos[player] = None
            event.append(f"{player} exits (+50), final utility {s.utility[player]}")
        elif action == "L" or action == "R":
            new_pos = pos - 1 if action == "L" else pos + 1

            # Increment adjacent_count if moving into an exit position
            if new_pos in EXITS:
                s.adjacent_count[player] += 1

            # Check for crush
            if s.pos.get(other) == new_pos and not s.exited[other]:
                s.utility[other] = -100
                s.pos[player] = new_pos
                event.append(f"{player} moves to {new_pos} and crushes {other} (-100 for {other})")
                s.exited[other] = True
                s.pos[other] = None

                # Crusher collects diamond if present
                if new_pos in s.diamonds:
                    s.diamonds.remove(new_pos)
                    s.utility[player] += 100
                    event.append(f"{player} collects diamond at {new_pos} (+100)")
            else:
                # Normal movement
                s.pos[player] = new_pos
                event.append(f"{player} moves to {new_pos}")
                if new_pos in s.diamonds:
                    s.diamonds.remove(new_pos)
                    s.utility[player] += 100
                    event.append(f"{player} collects diamond at {new_pos} (+100)")

        # Advance turn
        next_player = "R2" if player == "R1" else "R1"
        if not s.exited[next_player]:
            s.turn = next_player
        else:
            if not s.exited[player]:
                s.turn = player

        return s, "; ".join(event)

    def is_terminal(self, state: GameState) -> bool:
        """Check if state is terminal."""
        if state.exited["R1"] and state.exited["R2"]:
            return True

        for p in ("R1", "R2"):
            if self.legal_actions_for(p, state):
                return False
        return True

    def should_prune_subtree(self, state: GameState) -> bool:
        """
        Determine if we should prune this entire subtree.
        Prune if: all diamonds collected and both robots have exited or are at exits.
        """
        if not self.prune_nonsensical:
            return False

        # If both robots exited, it's terminal (not pruned, just terminal)
        if state.exited["R1"] and state.exited["R2"]:
            return False

        # If all diamonds collected and at least one robot has exited
        if len(state.diamonds) == 0:
            # Check remaining robot
            for p in ["R1", "R2"]:
                if not state.exited[p]:
                    # If remaining robot is at an exit and has positive utility, they should exit
                    # This will be handled by the terminal check
                    return False

        return False

    def build_tree(self):
        """Build the complete game tree starting from initial state."""
        initial = self.initial_state()
        self.root = TreeNode(state=initial, node_id=self.node_counter)
        self.node_counter += 1
        self.all_nodes.append(self.root)

        self._explore(self.root)
        return self.root

    def _explore(self, node: TreeNode):
        """Recursively explore and build the game tree."""
        # Check for early pruning
        if self.should_prune_subtree(node.state):
            node.is_terminal = True
            node.pruned = True
            self.pruned_count += 1
            self.terminal_nodes.append(node)
            return

        if self.is_terminal(node.state):
            node.is_terminal = True
            self.terminal_nodes.append(node)
            return

        player = node.state.turn
        if player not in ("R1", "R2"):
            node.is_terminal = True
            self.terminal_nodes.append(node)
            return

        actions = self.legal_actions_for(player, node.state)

        # Filter to sensible actions
        actions = self.filter_sensible_actions(player, actions, node.state)

        if not actions:
            node.is_terminal = True
            self.terminal_nodes.append(node)
            return

        for action in actions:
            new_state, event = self.apply_action(player, action, node.state)
            child = TreeNode(
                state=new_state,
                action=(player, action, event),
                node_id=self.node_counter
            )
            self.node_counter += 1
            self.all_nodes.append(child)
            node.children.append(child)

            self._explore(child)

    def get_statistics(self) -> Dict:
        """Get statistics about the game tree."""
        return {
            "total_nodes": len(self.all_nodes),
            "terminal_nodes": len(self.terminal_nodes),
            "pruned_nodes": self.pruned_count,
            "max_depth": self._max_depth(self.root),
            "branching_factors": self._compute_branching_factors()
        }

    def _max_depth(self, node: TreeNode, depth: int = 0) -> int:
        """Compute maximum depth of the tree."""
        if not node.children:
            return depth
        return max(self._max_depth(child, depth + 1) for child in node.children)

    def _compute_branching_factors(self) -> Dict:
        """Compute branching factor statistics."""
        non_terminal = [node for node in self.all_nodes if not node.is_terminal]
        if not non_terminal:
            return {"avg": 0, "max": 0, "min": 0}

        branching = [len(node.children) for node in non_terminal]
        return {
            "avg": sum(branching) / len(branching),
            "max": max(branching),
            "min": min(branching)
        }

    def print_summary(self):
        """Print a summary of the game tree."""
        stats = self.get_statistics()
        print(f"Game Tree Statistics:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Terminal nodes (outcomes): {stats['terminal_nodes']}")
        if self.prune_nonsensical:
            print(f"  Pruned nodes (nonsensical strategies): {stats['pruned_nodes']}")
        print(f"  Maximum depth: {stats['max_depth']}")
        print(f"  Branching factors - avg: {stats['branching_factors']['avg']:.2f}, "
              f"max: {stats['branching_factors']['max']}, min: {stats['branching_factors']['min']}")

        print(f"\nSample terminal outcomes:")
        for i, node in enumerate(self.terminal_nodes[:10]):
            print(f"\nOutcome #{i + 1}:")
            print(f"  Utilities: R1={node.state.utility['R1']}, R2={node.state.utility['R2']}")
            print(f"  Remaining diamonds: {node.state.diamonds}")
            print(f"  Exited: R1={node.state.exited['R1']}, R2={node.state.exited['R2']}")
            if node.pruned:
                print(f"  (Pruned subtree)")


# Main execution
if __name__ == "__main__":
    print("Building tree WITH pruning...")
    game_tree_pruned = MineGameTree(prune_nonsensical=True)
    root_pruned = game_tree_pruned.build_tree()
    game_tree_pruned.print_summary()

    print("\n" + "="*60)
    print("\nBuilding tree WITHOUT pruning...")
    game_tree_full = MineGameTree(prune_nonsensical=False)
    root_full = game_tree_full.build_tree()
    game_tree_full.print_summary()