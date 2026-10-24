
import random

# ============================================================
# 1. STATE SPACE: S and ACTIONS: A
# ============================================================

ROWS = 20
COLS = 20

WALL = (10, 10)
GOAL = (0, 19)
TRAP = (1, 19)
START = (19, 0)

TERMINAL_STATES = {GOAL, TRAP}

STATES = {
    (r, c)
    for r in range(ROWS)
    for c in range(COLS)
    if (r, c) != WALL
}

ACTIONS = {
    "NORTH": (-1, 0),
    "SOUTH": (1, 0),
    "WEST": (0, -1),
    "EAST": (0, 1),
    "SOUTHEAST": (1, 1),
    "SOUTHWEST": (1, -1),
    "NORTHWEST": (-1, -1),
    "NORTHEAST": (-1, 1),
}


# ============================================================
# 2. STOCHASTIC MOVEMENT
# ============================================================

PERPENDICULAR = {
    "NORTH": ("NORTHWEST", "NORTHEAST"),
    "WEST": ("NORTHWEST", "SOUTHWEST"),
    "EAST": ("NORTHEAST", "SOUTHEAST"),
    "SOUTH": ("SOUTHWEST", "SOUTHEAST"),
    "NORTHEAST": ("NORTH", "EAST"),
    "SOUTHWEST": ("SOUTH", "WEST"),
    "NORTHWEST": ("NORTH", "WEST"),
    "SOUTHEAST": ("SOUTH", "EAST"),
}

INTENDED_PROBABILITY = 0.8
SLIP_PROBABILITY = 0.1


# ============================================================
# 2. REWARD FUNCTION: R(s,a,s')
# ============================================================

STRAIGHT_STEP_REWARD = -0.1
DIAGONAL_STEP_REWARD = -0.14
GOAL_REWARD = 10.0
TRAP_REWARD = -10.0


def get_reward(action, next_state):
    """
    Implements the reward function R(s,a,s').
    """

    if next_state == GOAL:
        return GOAL_REWARD

    if next_state == TRAP:
        return TRAP_REWARD

    if action in ["NORTH", "SOUTH", "WEST", "EAST"]:
        return STRAIGHT_STEP_REWARD

    return DIAGONAL_STEP_REWARD


# ============================================================
# 5. ENVIRONMENT CLASS
# ============================================================

class Environment:

    def __init__(self):
        """
        Initialize the environment.
        """
        self.state = START
        self.done = False

    def reset(self):
        """
        Start a new episode.
        S_0 = START
        """
        self.state = START
        self.done = False

        return self.state


    # --------------------------------------------------------
    # 6. DETERMINISTIC MOVEMENT FUNCTION: f(s,d)
    # --------------------------------------------------------
    #
    # Mathematical definition:
    #
    #             s + Delta(d)   if s + Delta(d) in S
    # f(s,d) =
    #             s              otherwise
    #
    # This function has NO randomness.

    def move(self, state, direction):

        # Extract displacement Delta(d)
        dr, dc = ACTIONS[direction]

        # Extract the current coordinates
        r, c = state

        # Calculate candidate s + Delta(d)
        candidate = (r + dr, c + dc)

        # If the candidate belongs to S, movement succeeds
        if candidate in STATES:
            return candidate

        # Otherwise, the robot remains at s
        return state


    # --------------------------------------------------------
    # 7. TRANSITION MODEL: P(s' | s,a)
    # --------------------------------------------------------
    #
    # The mathematical transition probability is:
    #
    # P(s'|s,a) =
    #     sum over d of:
    #         P(D=d|a) * I[f(s,d)=s']
    #
    # I[condition] is the indicator function:
    #
    # I[True]  = 1
    # I[False] = 0
    #
    # We enumerate all possible actual directions.
    #
    # This is useful for:
    # - Bellman expectation equation
    # - Policy evaluation
    # - Policy iteration
    # - Value iteration

    def transitions(self, state, action):
        """
        Return possible outcomes as:

        (probability, next_state, reward)

        This function does NOT perform random sampling.
        """

        if state in TERMINAL_STATES:
            return []

        # Possible actual directions.
        directions = [
            action,
            *PERPENDICULAR[action]
        ]

        probabilities = [
            INTENDED_PROBABILITY,
            SLIP_PROBABILITY,
            SLIP_PROBABILITY
        ]

        outcomes = []

        # The loop represents summation over directions d.
        for direction, probability in zip(
            directions, probabilities
        ):

            # s' = f(s,d)
            next_state = self.move(state, direction)

            # r = R(s,a,s')
            reward = get_reward(action, next_state)

            outcomes.append(
                (probability, next_state, reward)
            )

        return outcomes

    # --------------------------------------------------------
    # 8. STOCHASTIC ENVIRONMENT STEP
    # --------------------------------------------------------
    #
    # The environment executes:
    #
    # D_t ~ P(D | A_t)
    #
    # S_(t+1) = f(S_t, D_t)
    #
    # R_(t+1) = R(S_t, A_t, S_(t+1))
    #
    # Unlike transitions(), step() samples ONE outcome.
    #
    # This is the function the learning agent will use
    # when interacting with the environment.

    def step(self, action):

        if self.done:
            raise RuntimeError(
                "Episode finished. Call reset() first."
            )

        if action not in ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )

        directions = [
            action,
            *PERPENDICULAR[action]
        ]

        probabilities = [
            INTENDED_PROBABILITY,
            SLIP_PROBABILITY,
            SLIP_PROBABILITY
        ]

        # Sample the actual direction.
        #
        # D_t ~ P(D | A_t)
        actual_direction = random.choices(
            directions,
            weights=probabilities,
            k=1
        )[0]

        # Apply the deterministic movement function.
        #
        # S_(t+1) = f(S_t, D_t)
        next_state = self.move(
            self.state,
            actual_direction
        )

        # Calculate the immediate reward.
        #
        # R_(t+1) = R(S_t, A_t, S_(t+1))
        reward = get_reward(action,next_state)

        # Update the environment's current state.
        self.state = next_state

        # Check whether a terminal state was reached.
        self.done = next_state in TERMINAL_STATES

        # Return one observed transition:
        #
        # (S_(t+1), R_(t+1), terminal)
        return next_state, reward, self.done


# ============================================================
# 9. TEST
# ============================================================

if __name__ == "__main__":

    env = Environment()

    state = env.reset()

    print("Initial state:", state)

    for t in range(20):

        action = "NORTH"

        next_state, reward, done = env.step(action)

        print(
            f"t={t:02d} | "
            f"action={action:5s} | "
            f"next_state={next_state} | "
            f"reward={reward:+.2f}"
        )

        if done:
            print("Terminal state reached.")
            break