
import random

# ============================================================
# 1. STATE SPACE: S and ACTIONS: A
# ============================================================

ROWS = 32
COLS = 32

START = (30, 1)
GOAL = (1, 30)
FIRE = (27, 27)
KEY = (16, 15)

TRAP = FIRE

VISION_RADIUS = 3

STEP_REWARD = -0.04

GRASS_REWARDS = {
    "short": -0.01,
    "medium": -0.06,
    "tall": -0.15
}

def generate_garden(seed=42):

    rng = random.Random(seed)

    walls = set()
    grass = {}

    # --------------------------------------------------------
    # A. CREATE GARDEN BOUNDARIES
    # --------------------------------------------------------

    for r in range(ROWS):
        walls.add((r, 0))
        walls.add((r, COLS - 1))

    for c in range(COLS):
        walls.add((0, c))
        walls.add((ROWS - 1, c))

    # --------------------------------------------------------
    # B. GENERATE RANDOM HEDGES
    # --------------------------------------------------------
    #
    # Each hedge is a horizontal or vertical line.
    # Their lengths and positions are randomized.

    for _ in range(55):

        r = rng.randint(2, ROWS - 3)
        c = rng.randint(2, COLS - 3)

        length = rng.randint(2, 7)

        horizontal = rng.choice([True, False])

        for i in range(length):

            if horizontal:
                cell = (r, c + i)
            else:
                cell = (r + i, c)

            if (
                1 <= cell[0] < ROWS - 1
                and 1 <= cell[1] < COLS - 1
            ):
                walls.add(cell)

    # --------------------------------------------------------
    # C. GENERATE BUSH CLUSTERS
    # --------------------------------------------------------
    #
    # Small irregular obstacles make the garden feel
    # less artificial than perfectly straight walls.

    for _ in range(45):

        r = rng.randint(2, ROWS - 3)
        c = rng.randint(2, COLS - 3)

        cluster_size = rng.randint(2, 5)

        for _ in range(cluster_size):

            dr = rng.randint(-1, 1)
            dc = rng.randint(-1, 1)

            cell = (r + dr, c + dc)

            if (
                1 <= cell[0] < ROWS - 1
                and 1 <= cell[1] < COLS - 1
            ):
                walls.add(cell)

    # --------------------------------------------------------
    # D. GUARANTEE A PATH THROUGH THE GARDEN
    # --------------------------------------------------------
    #
    # Carve randomized Manhattan paths:
    #
    # START -> KEY
    # KEY   -> GOAL
    #
    # These paths guarantee that a robot can reach
    # the key and subsequently reach the door.

    protected_path = set()

    def carve_path(origin, destination):

        r, c = origin
        protected_path.add((r, c))
        walls.discard((r, c))

        while (r, c) != destination:

            choices = []

            if r < destination[0]:
                choices.append((r + 1, c))
            elif r > destination[0]:
                choices.append((r - 1, c))

            if c < destination[1]:
                choices.append((r, c + 1))
            elif c > destination[1]:
                choices.append((r, c - 1))

            # Randomly choose one valid step toward the target.
            r, c = rng.choice(choices)

            walls.discard((r, c))
            protected_path.add((r, c))

    carve_path(START, KEY)
    carve_path(KEY, GOAL)

    # --------------------------------------------------------
    # E. OPEN ADDITIONAL GARDEN PASSAGES
    # --------------------------------------------------------
    #
    # Remove some hedge cells to create alternative paths.

    removable = list(
        walls - protected_path - {
            (r, c)
            for r in range(ROWS)
            for c in range(COLS)
            if r in (0, ROWS - 1)
            or c in (0, COLS - 1)
        }
    )

    rng.shuffle(removable)

    for cell in removable[:65]:
        walls.discard(cell)

    # --------------------------------------------------------
    # F. CREATE GRASS FIELDS
    # --------------------------------------------------------
    #
    # Grass fields are patches, not single random cells.
    #
    # Each field has an associated negative reward.
    #
    # R(s,a,s') depends on the terrain at s'.

    for _ in range(28):

        r = rng.randint(1, ROWS - 5)
        c = rng.randint(1, COLS - 5)

        height = rng.randint(2, 5)
        width = rng.randint(2, 5)

        grass_type = rng.choice(
            ["short", "medium", "tall"]
        )

        for dr in range(height):
            for dc in range(width):

                cell = (r + dr, c + dc)

                if (
                    cell not in walls
                    and cell not in (START, KEY, GOAL)
                ):
                    grass[cell] = grass_type

    # --------------------------------------------------------
    # G. PLACE THE FIRE PIT
    # --------------------------------------------------------
    #
    # Place the fire pit on an accessible tile near
    # the bottom-right, away from the guaranteed route.
    #
    # FIRE terminates the episode with a negative reward.

    fire_candidates = [
        (r, c)
        for r in range(22, ROWS - 1)
        for c in range(22, COLS - 1)
        if (r, c) not in walls
        and (r, c) not in protected_path
        and (r, c) not in (START, KEY, GOAL)
    ]

    fire = (
        rng.choice(fire_candidates)
        if fire_candidates
        else FIRE
    )

    walls.discard(fire)
    grass.pop(fire, None)

    return walls, grass, fire


# ============================================================
# 3. INITIALIZE THE WORLD
# ============================================================

WALLS, GRASS, FIRE = generate_garden(seed=42)

# Keep TRAP as an alias for older simulation code.
TRAP = FIRE

# Terminal positions.
#
# IMPORTANT:
# GOAL terminates only if the robot possesses the key.
# FIRE always terminates the episode.
#
# This set identifies special terminal locations;
# the actual terminal condition is checked later.

TERMINAL_STATES = {GOAL, FIRE}

STATES = {
    (r, c)
    for r in range(ROWS)
    for c in range(COLS)
    if (r, c) not in WALLS
}

RL_STATES = {
    (r, c, has_key)
    for r, c in STATES
    for has_key in (False, True)
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