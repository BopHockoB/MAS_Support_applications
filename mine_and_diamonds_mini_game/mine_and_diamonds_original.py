# Modeling the mine game and enumerating all distinct terminal histories (outcomes).
# It implements reasonable interpretations of ambiguous rules
# (see assumptions printed after the enumeration).

from copy import deepcopy

# Configuration of the mine
POSITIONS = [1, 2, 3, 4]  # linear positions
EXITS = {2, 4}  # positions adjacent to an exit (robots can exit from here)
DIAMOND_POSITIONS = {2, 4}  # initial diamond locations
START_POS = {"R1": 1, "R2": 3}  # starting positions (from conversation)

# Actions available (dependent on position and state)
# - "L": move left (pos-1)
# - "R": move right (pos+1)
# - "S": stay (allowed at most once per robot)
# - "E": exit (allowed only if pos in EXITS)
# Note: if a robot is forced to exit (second time adjacent to exit), only E is allowed.

# We'll represent a state as a dictionary:
# {
#   "pos": {"R1": int or None, "R2": int or None},  # None if exited
#   "exited": {"R1": bool, "R2": bool},              # True if exited
#   "utility": {"R1": int, "R2": int},
#   "stay_used": {"R1": bool, "R2": bool},           # whether Stay was already used
#   "adjacent_count": {"R1": int, "R2": int},       # how many times robot has moved into an exit position
#   "diamonds": set of remaining diamond positions (subset of {2,4}),
#   "turn": "R1" or "R2"                             # whose turn it is
# }

# We count terminal histories (distinct full play sequences) as separate outcomes.
terminal_histories = []  # list of (history_actions, terminal_state) pairs


def initial_state():
    ds = {
        "pos": {"R1": START_POS["R1"], "R2": START_POS["R2"]},
        "exited": {"R1": False, "R2": False},
        "utility": {"R1": 0, "R2": 0},
        "stay_used": {"R1": False, "R2": False},
        # Starting adjacency counts: starting in EXITS counts as first time adjacent
        "adjacent_count": {"R1": (1 if START_POS["R1"] in EXITS else 0),
                           "R2": (1 if START_POS["R2"] in EXITS else 0)},
        "diamonds": set(DIAMOND_POSITIONS),
        "turn": "R1"  # R1 acts first
    }
    return ds


def legal_actions_for(player, state):
    """
    Return list of legal action symbols for player in current state.
    If player has exited, return empty list.
    If forced exit by adjacency rule, return ["E"] only.
    """
    if state["exited"][player]:
        return []
    pos = state["pos"][player]
    actions = []

    # Forced exit: if adjacent_count >= 2, must exit
    if pos in EXITS and state["adjacent_count"][player] >= 2:
        return ["E"]

    # Otherwise list normal available actions:
    if pos > min(POSITIONS):
        actions.append("L")
    if pos < max(POSITIONS):
        actions.append("R")
    if not state["stay_used"][player]:
        actions.append("S")
    if pos in EXITS:
        actions.append("E")
    return actions


def apply_action(player, action, state):
    """
    Apply action for player on a deep copy of state and return (new_state, event_description).
    event_description is a short text describing important events (diamond collected, crush, exit).
    """
    s = deepcopy(state)
    if s["exited"][player]:
        return s, "no-op"
    pos = s["pos"][player]
    other = "R2" if player == "R1" else "R1"
    event = []

    if action == "S":
        s["stay_used"][player] = True
        event.append(f"{player} stays at {pos}")
        # no immediate payoff unless crushed later
    elif action == "E":
        # Exit: robot collects +50 on top of accumulated utility and is removed from game
        s["utility"][player] += 50
        s["exited"][player] = True
        s["pos"][player] = None
        event.append(f"{player} exits (+50), final utility {s['utility'][player]}")
    elif action == "L" or action == "R":
        new_pos = pos - 1 if action == "L" else pos + 1

        # Increment adjacent_count if moving into an exit position
        if new_pos in EXITS:
            s["adjacent_count"][player] += 1

        # If other robot occupies new_pos => crush the occupant
        if s["pos"].get(other) == new_pos and not s["exited"][other]:
            # occupant crushed: loses all previous utility and gets -100 penalty
            s["utility"][other] = -100

            # moving robot moves into new position; moving robot gets no utility change from crush
            s["pos"][player] = new_pos
            event.append(f"{player} moves to {new_pos} and crushes {other} (-100 for {other})")
            # Mark crushed robot as exited (they're out of the game)
            s["exited"][other] = True
            s["pos"][other] = None

            # If there was a diamond at new_pos and hasn't been collected yet, check whether crusher collects it:
            # Rule: "If a robot steps on a diamond it will collect it" -- moving robot steps on the location, so yes.
            if new_pos in s["diamonds"]:
                s["diamonds"].remove(new_pos)
                s["utility"][player] += 100
                event.append(f"{player} collects diamond at {new_pos} (+100)")
        else:
            # normal movement; if diamond present, collect
            s["pos"][player] = new_pos
            event.append(f"{player} moves to {new_pos}")
            if new_pos in s["diamonds"]:
                s["diamonds"].remove(new_pos)
                s["utility"][player] += 100
                event.append(f"{player} collects diamond at {new_pos} (+100)")
    else:
        raise ValueError("Unknown action")

    # Advance turn to the other player if they haven't exited
    next_player = "R2" if player == "R1" else "R1"
    if not s["exited"][next_player]:
        s["turn"] = next_player
    else:
        # If next player has exited, keep turn with current player if they haven't exited
        if not s["exited"][player]:
            s["turn"] = player
        # else both have exited, game is terminal

    return s, "; ".join(event)


def is_terminal(state):
    """Terminal if both players have exited OR if current player has no legal actions."""
    # If both exited, terminal
    if state["exited"]["R1"] and state["exited"]["R2"]:
        return True

    # Check if any player has legal actions
    for p in ("R1", "R2"):
        if legal_actions_for(p, state):
            return False
    return True


# Depth-first exploration of histories (count terminal histories)
def explore(state, history_actions):
    # If state is terminal, record it
    if is_terminal(state):
        terminal_histories.append((list(history_actions), deepcopy(state)))
        return
    player = state["turn"]
    # If None or invalid, treat as terminal
    if player not in ("R1", "R2"):
        terminal_histories.append((list(history_actions), deepcopy(state)))
        return
    actions = legal_actions_for(player, state)
    # If no legal actions, terminal
    if not actions:
        terminal_histories.append((list(history_actions), deepcopy(state)))
        return
    for a in actions:
        new_state, event = apply_action(player, a, state)
        explore(new_state, history_actions + [(player, a, event)])


# Run the exhaustive enumeration
start = initial_state()
explore(start, [])

# Show summary
num_terminal = len(terminal_histories)
print(f"Number of distinct terminal histories (outcomes found): {num_terminal}\n")

# Print a few sample terminal histories and their payoffs
print("Sample terminal histories:\n")
for i, (hist, st) in enumerate(terminal_histories):
    print(f"Outcome #{i + 1}:")
    for step in hist:
        print("  ", step)
    print("  Terminal utilities:", st["utility"], "Remaining diamonds:", st["diamonds"], "Exited:", st["exited"])
    print()

print(f"\nTotal number of distinct terminal histories: {num_terminal}")