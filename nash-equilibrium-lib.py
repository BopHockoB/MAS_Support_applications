import nashpy as nash
import numpy as np
# Payoff matrices: A for Player 1, B for Player 2
R1 = np.array([[4, 7, 12],
               [8, 9, 7],
               [6, 5, 7]])

R2 = np.array([[10, 4, 5],
               [7, 5, 5],
               [11, 4, 6]])

game = nash.Game(R1, R2) # Computes all Nash equilibria
equilibria = list(game.support_enumeration())

print(equilibria)