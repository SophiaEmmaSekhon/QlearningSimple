# QlearningSimple
Simple Q Learning Exercise.

Clean Q-learning GridWorld with action masking (no illegal moves).

Usage (defaults use a 4x4 grid, start=(3,0), goal=(0,3), lava at (1,1)):

    python main.py
    python main.py --rows 5 --cols 6 --start 4,0 --goal 0,5 --lava 1,1;2,2;3,3 --episodes 1000

Key ideas:
- States are grid cells encoded as a single integer s in [0, rows*cols).
- Actions are UP=0, RIGHT=1, DOWN=2, LEFT=3.
- A boolean action mask ensures the agent only selects/computes over valid actions.
- Invalid actions are never chosen, and Q-updates only bootstrap over valid next actions.
