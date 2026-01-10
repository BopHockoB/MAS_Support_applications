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
        Filter actions to allow diverse strategies (hostile vs defensive).

        Allows multiple playstyles:
        - HOSTILE: Move toward opponent to crush them, block their path, stay to ambush
        - DEFENSIVE: Collect diamonds safely, exit early with positive score, avoid opponent
        - OPPORTUNISTIC: Balance between diamond collection and opponent interaction

        Only removes truly irrational moves that no strategy would justify.
        """
        if not self.prune_nonsensical or not actions:
            return actions

        pos = state.pos[player]
        other = "R2" if player == "R1" else "R1"
        other_pos = state.pos.get(other)
        other_exited = state.exited[other]

        at_exit = pos in EXITS
        my_utility = state.utility[player]
        other_utility = state.utility[other]

        all_diamonds_collected = len(state.diamonds) == 0
        diamonds_remain = len(state.diamonds) > 0

        rational_moves = []

        # MANDATORY: If forced to act in certain situations
        # Rule 1: All diamonds gone, at exit, have positive utility -> MUST exit (no other rational option)
        if all_diamonds_collected and at_exit and my_utility > 0 and other_exited:
            return ["E"]

        # Rule 2: All diamonds gone, both players have exited or no moves benefit anyone
        if all_diamonds_collected and other_exited and at_exit:
            return ["E"]

        # Now allow diverse strategies by categorizing moves:

        for action in actions:
            reasons = []  # Track why this move is rational

            if action == "E":
                # DEFENSIVE: Exit to secure positive score
                if my_utility > 0:
                    reasons.append("defensive_exit")
                # DEFENSIVE: Exit to avoid further negative utility
                if my_utility < 0 and not diamonds_remain:
                    reasons.append("cut_losses")
                # NEUTRAL: Only sensible move if diamonds gone
                if all_diamonds_collected:
                    reasons.append("game_over")

            elif action == "L" or action == "R":
                new_pos = pos - 1 if action == "L" else pos + 1

                # HOSTILE: Move toward opponent to crush them
                if other_pos and not other_exited:
                    current_dist = abs(pos - other_pos)
                    new_dist = abs(new_pos - other_pos)
                    if new_dist < current_dist:
                        reasons.append("hostile_approach")
                    # Can crush opponent in one move
                    if new_pos == other_pos:
                        reasons.append("hostile_crush")

                # DEFENSIVE/OPPORTUNISTIC: Move toward diamonds
                if diamonds_remain:
                    for diamond_pos in state.diamonds:
                        current_dist = abs(pos - diamond_pos)
                        new_dist = abs(new_pos - diamond_pos)
                        if new_dist < current_dist:
                            reasons.append("collect_diamond")
                            break

                # DEFENSIVE: Move toward exit when diamonds collected
                if all_diamonds_collected and my_utility > 0:
                    for exit_pos in EXITS:
                        current_dist = abs(pos - exit_pos)
                        new_dist = abs(new_pos - exit_pos)
                        if new_dist < current_dist:
                            reasons.append("defensive_exit_approach")
                            break

                # DEFENSIVE: Move away from opponent to avoid crush
                if other_pos and not other_exited:
                    current_dist = abs(pos - other_pos)
                    new_dist = abs(new_pos - other_pos)
                    if new_dist > current_dist and my_utility > other_utility:
                        reasons.append("defensive_evade")

                # HOSTILE: Block opponent's path to diamond
                if diamonds_remain and other_pos and not other_exited:
                    for diamond_pos in state.diamonds:
                        # Check if new position blocks opponent's direct path
                        if other_pos < diamond_pos <= new_pos or other_pos > diamond_pos >= new_pos:
                            reasons.append("hostile_block")
                            break

            elif action == "S":
                # HOSTILE: Stay to ambush opponent
                if other_pos and not other_exited:
                    if abs(pos - other_pos) <= 2:  # Opponent nearby
                        reasons.append("hostile_ambush")

                # DEFENSIVE: Stay at strategic position (near diamond)
                if pos in state.diamonds:
                    reasons.append("defensive_guard_diamond")

                # HOSTILE: Stay to block opponent's access
                if other_pos and diamonds_remain:
                    for diamond_pos in state.diamonds:
                        if pos == diamond_pos or (other_pos < diamond_pos < pos) or (other_pos > diamond_pos > pos):
                            reasons.append("hostile_block")
                            break

            # Include move if there's ANY rational strategy that supports it
            if reasons:
                rational_moves.append(action)

        # If no moves had strategic justification, allow all (better safe than sorry)
        if not rational_moves:
            rational_moves = actions

        # Only absolute filter: Don't stay if all diamonds collected AND both players should exit
        if all_diamonds_collected and at_exit and my_utility > 0:
            rational_moves = [a for a in rational_moves if a != "S"]

        return rational_moves if rational_moves else actions

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

            # Check for crush - but crushed robot does NOT die
            if s.pos.get(other) == new_pos and not s.exited[other]:
                s.utility[other] -= 100
                s.pos[player] = new_pos
                event.append(f"{player} moves to {new_pos} and crushes {other} (-100 for {other})")
                # Crushed robot stays at the same position and can continue playing

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

        # Filter to sensible actions (allows diverse strategies)
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
    print("Building tree with diverse strategies (hostile + defensive)...")
    game_tree_pruned = MineGameTree(prune_nonsensical=True)
    root_pruned = game_tree_pruned.build_tree()
    game_tree_pruned.print_summary()

    print("\n" + "=" * 60)
    print("\nBuilding tree WITHOUT pruning...")
    game_tree_full = MineGameTree(prune_nonsensical=False)
    root_full = game_tree_full.build_tree()
    game_tree_full.print_summary()