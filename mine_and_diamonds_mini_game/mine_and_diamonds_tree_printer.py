"""
Visualization utilities for the Mine Game Tree.
Requires: pip install networkx matplotlib graphviz
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional


def tree_to_networkx(root, max_depth: Optional[int] = None, vert_gap: float = 0.2, width: float = 1.0):
    """
    Convert the game tree to a NetworkX directed graph.

    Args:
        root: Root TreeNode of the game tree
        max_depth: Maximum depth to include (None for full tree)
        vert_gap: Vertical gap between layers (default 0.2)
        width: Horizontal width/spacing (default 1.0, larger = more spread out)

    Returns:
        tuple: (graph, pos, labels, colors)
    """
    G = nx.DiGraph()
    labels = {}
    colors = []

    def add_nodes(node, depth=0):
        if max_depth is not None and depth > max_depth:
            return

        # Add node with robot positions
        r1_pos = node.state.pos["R1"]
        r2_pos = node.state.pos["R2"]
        pos_str = f"({r1_pos if r1_pos else 'X'},{r2_pos if r2_pos else 'X'})"

        if node.action:
            player, action, event = node.action
            # Check if this is a crush event
            if "crushes" in event.lower():
                node_label = f"{player}:{action} CRUSH!\n{pos_str}"
            else:
                node_label = f"{player}:{action}\n{pos_str}"
        elif depth == 0:
            node_label = f"Start\n{pos_str}"
        else:
            node_label = pos_str

        G.add_node(node.node_id)
        labels[node.node_id] = node_label

        # Color nodes: crush events in red, terminal in coral, pruned in gray, others in blue
        if node.action and "crushes" in node.action[2].lower():
            colors.append('red')
        elif hasattr(node, 'pruned') and node.pruned:
            colors.append('lightgray')
        elif node.is_terminal:
            colors.append('lightcoral')
        else:
            colors.append('lightblue')

        # Add edges to children
        for child in node.children:
            if max_depth is None or depth < max_depth:
                G.add_edge(node.node_id, child.node_id)
                add_nodes(child, depth + 1)

    add_nodes(root)

    # Use hierarchical layout with better spacing
    pos = hierarchy_pos(G, root.node_id, width=width * 10, vert_gap=vert_gap, min_node_gap=0.3)

    return G, pos, labels, colors


def hierarchy_pos(G, root, width=1., vert_gap=0.2, min_node_gap=0.15):
    """
    Improved hierarchical layout that guarantees minimum spacing between nodes.
    Uses a two-pass algorithm: first assigns positions, then adjusts to prevent overlaps.
    """
    # First pass: assign levels and collect nodes per level
    levels = {}
    node_to_level = {}

    def assign_levels(node, level=0):
        node_to_level[node] = level
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
        for child in G.neighbors(node):
            assign_levels(child, level + 1)

    assign_levels(root)

    # Second pass: position nodes level by level
    pos = {}
    max_level = max(levels.keys())

    for level in range(max_level + 1):
        nodes_at_level = levels[level]
        num_nodes = len(nodes_at_level)

        if level == 0:
            # Root node centered
            pos[root] = (0, 0)
        else:
            # Calculate positions for this level
            # Space them evenly, ensuring minimum gap
            total_width = max(num_nodes * min_node_gap, width)

            if num_nodes == 1:
                # Single node - center it under parent
                node = nodes_at_level[0]
                parent = list(G.predecessors(node))[0]
                parent_x = pos[parent][0]
                pos[node] = (parent_x, -level * vert_gap)
            else:
                # Multiple nodes - distribute with minimum spacing
                start_x = -(total_width / 2)
                dx = total_width / (num_nodes - 1) if num_nodes > 1 else 0

                for i, node in enumerate(sorted(nodes_at_level)):
                    x = start_x + i * dx
                    y = -level * vert_gap
                    pos[node] = (x, y)

    return pos

def plot_tree(root, max_depth: Optional[int] = None,
              figsize=(20, 12), save_path: Optional[str] = None,
              node_size=2800, font_size=6, vert_gap=0.2, width=1.0):
    """
    Plot the game tree with utility information on terminal nodes.

    Args:
        root: Root TreeNode of the game tree
        max_depth: Maximum depth to include (None for full tree)
        figsize: Figure size (width, height)
        save_path: Path to save the plot (None to just display)
        node_size: Size of nodes (default 2800)
        font_size: Font size for labels (default 6)
        vert_gap: Vertical gap between layers (default 0.2, smaller = closer layers)
        width: Horizontal spacing (default 1.0, larger = more spread out nodes)
    """
    G = nx.DiGraph()
    labels = {}
    colors = []

    def add_nodes(node, depth=0):
        if max_depth is not None and depth > max_depth:
            return

        # Get robot positions
        r1_pos = node.state.pos["R1"]
        r2_pos = node.state.pos["R2"]
        pos_str = f"({r1_pos if r1_pos else 'X'},{r2_pos if r2_pos else 'X'})"

        # Create label
        if node.is_terminal:
            u1 = node.state.utility['R1']
            u2 = node.state.utility['R2']
            if hasattr(node, 'pruned') and node.pruned:
                node_label = f"Pruned\n({u1},{u2})"
            else:
                node_label = f"({u1},{u2})"
        elif node.action:
            player, action, event = node.action
            # Check if this is a crush event
            if "crushes" in event.lower():
                node_label = f"{player}:{action} \n{pos_str}"
            else:
                node_label = f"{player}:{action}\n{pos_str}"
        else:
            node_label = f"Start\n{pos_str}"

        G.add_node(node.node_id)
        labels[node.node_id] = node_label

        # Color by event type and terminal status
        if node.action and "crushes" in node.action[2].lower():
            colors.append('salmon')
        elif hasattr(node, 'pruned') and node.pruned:
            colors.append('lightgray')
        elif node.is_terminal:
            colors.append('lightgreen')
        else:
            colors.append('lightblue')

        # Add edges to children
        for child in node.children:
            if max_depth is None or depth < max_depth:
                G.add_edge(node.node_id, child.node_id)
                add_nodes(child, depth + 1)

    add_nodes(root)
    pos = hierarchy_pos(G, root.node_id, width=width * 10, vert_gap=vert_gap, min_node_gap=0.3)

    plt.figure(figsize=figsize)
    nx.draw(G, pos,
            labels=labels,
            node_color=colors,
            node_size=node_size,
            font_size=font_size,
            font_weight='bold',
            arrows=True,
            arrowsize=10,
            edge_color='gray',
            width=1.5,
            alpha=0.9)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

# Example usage
if __name__ == "__main__":
    # Import the game tree builder
    from mine_and_diamonds import MineGameTree

    # Build the pruned tree (only sensible strategies)
    print("Building pruned game tree")
    game_tree = MineGameTree(prune_nonsensical=True)
    root = game_tree.build_tree()
    game_tree.print_summary()


    # Plot with utilities
    print("\nPlotting pruned tree with utilities (depth ≤ 10)...")
    plot_tree(root, max_depth=10, figsize=(30, 21),
              node_size=4000, font_size=12,
              save_path="game_tree_utilities.png")