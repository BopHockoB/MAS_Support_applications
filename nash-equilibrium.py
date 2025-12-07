import numpy as np

payoffs = np.array([
    [(4, 10), (7, 4), (12, 5)],
    [(8, 7),  (9, 5), (7, 5)],
    [(6, 11), (5, 4), (7, 6)]
])

payoff_robot1 = np.array([[cell[0] for cell in row] for row in payoffs])
payoff_robot2 = np.array([[cell[1] for cell in row] for row in payoffs])


n_rows, n_cols = payoff_robot1.shape

#Find best responses for Robot 1
# A best response for Robot 1 is the strategy (row) that gives the highest payoff
# given a specific strategy of Robot 2 (column)
best_responses_r1 = []
for j in range(n_cols):
    col = payoff_robot1[:, j] # All payoffs for Robot 1 if Robot 2 plays column j
    max_val = np.max(col) # Find the highest payoff in this column

    # Identify all rows that give this maximum payoff
    best_rows = [i for i in range(n_rows) if col[i] == max_val]
    best_responses_r1.append(best_rows)

# Find best responses for Robot 2
# A best response for Robot 2 is the strategy (column) that gives the highest payoff
# given a specific strategy of Robot 1 (row)
best_responses_r2 = []
for i in range(n_rows):
    row = payoff_robot2[i, :] # All payoffs for Robot 2 if Robot 1 plays row i
    max_val = np.max(row) # Find the highest payoff in this row

    # Identify all columns that give this maximum payoff
    best_cols = [j for j in range(n_cols) if row[j] == max_val]
    best_responses_r2.append(best_cols)

#Find pure strategy Nash equilibria
nash_equilibria = []
for i in range(n_rows):
    for j in range(n_cols):
        if i in best_responses_r1[j] and j in best_responses_r2[i]:
            # if both robots are playing best responses -> Nash equilibrium
            nash_equilibria.append(((i, j), payoffs[i][j]))


for eq in nash_equilibria:
    row_index, col_index = eq[0]
    payoff = eq[1]

    strategies = (f"{row_index+1}", f"{col_index+1}")
    print(f"Best strategies: {strategies}, Payoffs: {payoff}")
