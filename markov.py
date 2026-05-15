import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('/content/Iris.csv')

# Extract unique states from Species column
states = list(df['Species'].unique())

print("States:", states)

# Example transition matrix (3 species)
transition_matrix = [
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.3, 0.3, 0.4]
]

# Start from first state
current_state = 0

print("\nInitial State:", states[current_state])

# Simulate 10 transitions
for i in range(10):
    current_state = np.random.choice(
        [0, 1, 2],
        p=transition_matrix[current_state]
    )
    print("Step", i+1, ":", states[current_state])
