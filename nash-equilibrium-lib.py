import nashpy as nash
import numpy as np


R1 = np.array([[4, 7, 12],
               [8, 9, 7],
               [6, 5, 7]])

R2 = np.array([[10, 4, 5],
               [7, 5, 5],
               [11, 4, 6]])


game = nash.Game(R2, R1)
equilibria = list(game.support_enumeration())

"""
A "support" is the set of pure strategies that receive strictly positive
probability in a mixed strategy.
All strategies in the support must give the same expected payoff.
All strategies outside the support must give no higher payoff.

Because we do not know the support in advance, Nashpy tries them all.

Each pair (S1, S2) is a candidate support like:
S1 = {row 2, row 3}
S2 = {col 1, col 2}
  
Any strategy excluded from the support must have probability zero, 
otherwise it would belong to the support.

In a Nash equilibrium all strategies in a player's support
must hold the same expected payoff.

Therefore:
  For Player 1:
      Expected payoff of row i = Expected payoff of row k
      for all i, k in S1
      where row i sum_j A[i, j] * q_j
      same for k

  For Player 2:
      Expected payoff of column j = Expected payoff of column l
      for all j, l in S2
      where column j sum_i B[i, j] * p_i
      same for l

It gives us a system of linear equations, where strategies must sum to 1:

  sum_{i in S1} p_i = 1
  sum_{j in S2} q_j = 1

Possible outcomes:
  - No solution (discard this support)
  - Negative values (discard (invalid probabilities)
  - Valid solution

For strategies outside the support:
    Expected payoff of excluded row/column 
    <= payoff of row/columns in selected strategies

If any excluded strategy gives a higher payoff
its assumed support cant be a Nash equilibrium

If all conditions are satisfied meaning
then The solution is a valid NE

Nashpy stores it and continues checking other supports.

After all support pairs have been tested every valid equilibrium is returned
"""
print(equilibria)
