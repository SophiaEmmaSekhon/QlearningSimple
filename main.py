#!/usr/bin/env python3
"""
main.py — Clean Q-learning GridWorld with action masking (no illegal moves).

Usage (defaults use a 4x4 grid, start=(3,0), goal=(0,3), lava at (1,1)):

    python main.py
    python main.py --rows 5 --cols 6 --start 4,0 --goal 0,5 --lava 1,1;2,2;3,3 --episodes 1000

Key ideas:
- States are grid cells encoded as a single integer s in [0, rows*cols).
- Actions are UP=0, RIGHT=1, DOWN=2, LEFT=3.
- A boolean action mask ensures the agent only selects/computes over valid actions.
- Invalid actions are never chosen, and Q-updates only bootstrap over valid next actions.
"""

from __future__ import annotations

import argparse
import re
from typing import Iterable, List, Sequence, Set, Tuple
import numpy as np

# ----- Action encoding -----
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_CHARS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}
ACTION_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}


# ----- Coordinate/index helpers -----
def to_state(r: int, c: int, cols: int) -> int:
    """Pack (row, col) -> state index (row-major)."""
    return r * cols + c


def to_rc(s: int, cols: int) -> Tuple[int, int]:
    """Unpack state index -> (row, col) (row-major)."""
    return divmod(s, cols)


# ----- Mask construction -----
def build_action_mask(rows: int, cols: int) -> np.ndarray:
    """
    Build boolean mask of shape (rows*cols, 4). mask[s, a] is True iff action a is valid in state s.
    Valid means: the action keeps the agent within grid bounds.
    """
    n_states = rows * cols
    mask = np.zeros((n_states, 4), dtype=bool)
    for s in range(n_states):
        r, c = divmod(s, cols)
        if r > 0:
            mask[s, UP] = True
        if c < cols - 1:
            mask[s, RIGHT] = True
        if r < rows - 1:
            mask[s, DOWN] = True
        if c > 0:
            mask[s, LEFT] = True
    return mask


# ----- Environment step -----
def step_env(
    s: int,
    a: int,
    rows: int,
    cols: int,
    goal_state: int,
    lava_states: Set[int],
    step_penalty: float,
    goal_reward: float,
    lava_penalty: float,
) -> Tuple[int, float, bool]:
    """
    Apply action a in state s and return (s_next, reward, done).

    Assumes the caller only passes valid actions (stays on the board).
    Episode ends when entering goal or a lava cell.
    """
    r, c = to_rc(s, cols)

    # Move according to the action (assumed valid by caller).
    if a == UP:
        r -= 1
    elif a == RIGHT:
        c += 1
    elif a == DOWN:
        r += 1
    elif a == LEFT:
        c -= 1
    else:
        raise ValueError(f"Unknown action id: {a}")

    s_next = to_state(r, c, cols)

    # Terminal transitions
    if s_next == goal_state:
        return s_next, goal_reward, True
    if s_next in lava_states:
        return s_next, lava_penalty, True

    # Non-terminal step
    return s_next, step_penalty, False


# ----- Policy/action selection -----
def epsilon_greedy(
    Q: np.ndarray, s: int, epsilon: float, mask: np.ndarray, rng: np.random.Generator
) -> int:
    """
    Select an action using ε-greedy *over valid actions only*.
    """
    valid_idx = np.flatnonzero(mask[s])
    if valid_idx.size == 0:
        raise RuntimeError(f"No valid actions in state {s}.")
    if rng.random() < epsilon:
        return int(rng.choice(valid_idx))
    qvals = Q[s, valid_idx]
    max_q = qvals.max()
    best = valid_idx[qvals == max_q]
    return int(rng.choice(best))


# ----- Q-learning update -----
def q_update(
    Q: np.ndarray,
    s: int,
    a: int,
    r: float,
    s_next: int,
    done: bool,
    alpha: float,
    gamma: float,
    mask: np.ndarray,
) -> None:
    """
    Standard tabular Q-learning update with action masking for the bootstrap term.
    """
    if done:
        target = r
    else:
        next_valid = np.flatnonzero(mask[s_next])
        if next_valid.size == 0:
            best_next = 0.0
        else:
            best_next = float(Q[s_next, next_valid].max())
        target = r + gamma * best_next

    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])


# ----- Training loop -----
def train(
    rows: int,
    cols: int,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    lava_rcs: Sequence[Tuple[int, int]] = (),
    episodes: int = 600,
    max_steps: int = 50,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    step_penalty: float = -0.01,
    goal_reward: float = 10.0,
    lava_penalty: float = -1.0,
    seed: int = 53257,
    log_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray, int, int, Set[int], List[float]]:
    """
    Train a Q-learner in the masked GridWorld.

    Returns:
        Q:           (n_states, 4) Q-table with -inf for invalid actions, learned values for valid ones.
        mask:        (n_states, 4) boolean action mask.
        start_s:     encoded start state
        goal_s:      encoded goal state
        lava_states: set of encoded lava states
        returns:     list of per-episode total returns
    """
    assert rows > 0 and cols > 0, "Grid dimensions must be positive."
    n_states = rows * cols

    # Encode key cells
    start_s = to_state(*start_rc, cols)
    goal_s = to_state(*goal_rc, cols)
    lava_states = {to_state(r, c, cols) for (r, c) in lava_rcs}

    # Basic validation
    def _in_bounds(r: int, c: int) -> bool:
        return (0 <= r < rows) and (0 <= c < cols)

    assert _in_bounds(*start_rc), "Start outside grid."
    assert _in_bounds(*goal_rc), "Goal outside grid."
    for rc in lava_rcs:
        assert _in_bounds(*rc), f"Lava outside grid: {rc}"
    assert start_s != goal_s, "Start cannot equal goal."
    assert start_s not in lava_states, "Start cannot be lava."
    assert goal_s not in lava_states, "Goal cannot be lava."

    # Build mask and initialize Q
    mask = build_action_mask(rows, cols)
    Q = np.full((n_states, 4), -np.inf, dtype=np.float32)
    Q[mask] = 0.0  # only valid entries start at 0; invalid actions stay at -inf

    rng = np.random.default_rng(seed)
    ep_returns: List[float] = []

    for ep in range(1, episodes + 1):
        s = start_s
        done = False
        total = 0.0

        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, mask, rng)
            s_next, r, done = step_env(
                s,
                a,
                rows,
                cols,
                goal_s,
                lava_states,
                step_penalty,
                goal_reward,
                lava_penalty,
            )
            q_update(Q, s, a, r, s_next, done, alpha, gamma, mask)
            s = s_next
            total += r
            if done:
                break

        ep_returns.append(total)

        # ε decay
        if epsilon > epsilon_min:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Optional logging
        if log_every and ep % log_every == 0:
            recent = ep_returns[-log_every:]
            avg_recent = sum(recent) / len(recent)
            print(
                f"Episode {ep:5d} | avg return (last {log_every}): {avg_recent:7.3f} | "
                f"epsilon: {epsilon:6.3f}"
            )

    return Q, mask, start_s, goal_s, lava_states, ep_returns


# ----- Rendering -----
def render_policy(
    Q: np.ndarray,
    mask: np.ndarray,
    rows: int,
    cols: int,
    goal_s: int,
    lava_states: Set[int],
    start_s: int | None = None,
) -> None:
    """Print the greedy policy as arrow glyphs; mark Goal 'G', Lava 'X', Start 'S'."""
    grid = np.full((rows, cols), "•", dtype="<U1")
    n_states = rows * cols

    for s in range(n_states):
        r, c = to_rc(s, cols)
        if s == goal_s:
            grid[r, c] = "G"
        elif s in lava_states:
            grid[r, c] = "X"
        elif start_s is not None and s == start_s:
            # We'll overwrite with arrow after computing, but mark later to keep arrow visible.
            pass
        else:
            valid_idx = np.flatnonzero(mask[s])
            if valid_idx.size == 0:
                ch = "•"
            else:
                best = valid_idx[np.argmax(Q[s, valid_idx])]
                ch = ACTION_CHARS[best]
            grid[r, c] = ch

    # Ensure we label Start cell (if provided) after arrows
    if start_s is not None:
        r, c = to_rc(start_s, cols)
        grid[r, c] = "S"

    print("\nLearned greedy policy:")
    for r in range(rows):
        print(" ".join(grid[r]))


# ----- CLI parsing -----
def _parse_rc(text: str) -> Tuple[int, int]:
    """Parse 'r,c' into (r, c) integers."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected format 'r,c'")
    try:
        r = int(parts[0].strip())
        c = int(parts[1].strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("Row/col must be integers") from e
    return r, c


def _parse_rc_list(text: str) -> List[Tuple[int, int]]:
    """
    Parse a list like 'r1,c1;r2,c2 r3,c3' into [(r1,c1), (r2,c2), (r3,c3)].
    Accepts semicolons and/or spaces as separators.
    """
    text = text.strip()
    if not text:
        return []
    chunks = re.split(r"[; ]+", text)
    out: List[Tuple[int, int]] = []
    for ch in chunks:
        if ch:
            out.append(_parse_rc(ch))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Q-learning GridWorld with action masking (tabular)."
    )
    p.add_argument("--rows", type=int, default=4, help="Number of rows in the grid.")
    p.add_argument("--cols", type=int, default=4, help="Number of columns in the grid.")
    p.add_argument("--start", type=_parse_rc, default=(3, 0), help="Start cell 'r,c'.")
    p.add_argument("--goal", type=_parse_rc, default=(0, 3), help="Goal cell 'r,c'.")
    p.add_argument(
        "--lava",
        type=_parse_rc_list,
        default=_parse_rc_list("1,1"),
        help="Lava cells as 'r,c;r,c;...'. Empty for none.",
    )
    p.add_argument("--episodes", type=int, default=600, help="Training episodes.")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode.")
    p.add_argument("--alpha", type=float, default=0.1, help="Learning rate α.")
    p.add_argument("--gamma", type=float, default=0.95, help="Discount γ.")
    p.add_argument("--epsilon", type=float, default=1.0, help="Initial ε for ε-greedy.")
    p.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum ε.")
    p.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative ε decay per episode.",
    )
    p.add_argument(
        "--step-penalty", type=float, default=-0.01, help="Reward per non-terminal step."
    )
    p.add_argument("--goal-reward", type=float, default=10.0, help="Reward at goal.")
    p.add_argument("--lava-penalty", type=float, default=-1.0, help="Reward at lava.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility (0 = random seed).")
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log average return every N episodes (0 to disable).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    
    # Use random seed if none specified
    if args.seed == 0:
        import time
        args.seed = int(time.time() * 1000000) % 2**32
    
    print(f"Using seed: {args.seed}")

    Q, mask, start_s, goal_s, lava_states, returns = train(
        rows=args.rows,
        cols=args.cols,
        start_rc=args.start,
        goal_rc=args.goal,
        lava_rcs=args.lava,
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        step_penalty=args.step_penalty,
        goal_reward=args.goal_reward,
        lava_penalty=args.lava_penalty,
        seed=args.seed,
        log_every=args.log_every,
    )

    # Show the learned greedy policy
    render_policy(
        Q=Q,
        mask=mask,
        rows=args.rows,
        cols=args.cols,
        goal_s=goal_s,
        lava_states=lava_states,
        start_s=start_s,
    )

    # Optional: print a tiny Q-snippet for the start state
    valid_idx = np.flatnonzero(mask[start_s])
    print("\nQ-values at Start (valid actions only):")
    for a in valid_idx:
        print(f"  {ACTION_NAMES[a]:>5s}: {Q[start_s, a]:8.4f}")

    # Final return summary
    if returns:
        avg_last_100 = sum(returns[-100:]) / min(100, len(returns))
        print(f"\nAverage return over last {min(100, len(returns))} episodes: {avg_last_100:.3f}")


if __name__ == "__main__":
    main()
