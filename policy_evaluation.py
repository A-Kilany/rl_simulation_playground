
# ============================================================
# POLICY EVALUATION
# ============================================================
#
# Goal:
# Calculate the value V^pi(s) of every state when the robo follows a policy pi.
#
# Iterative policy evaluation:
#
# V_(k+1)(s) = SUM_a pi(a|s) *
#                 SUM_s' P(s'|s,a) *
#                 [R(s,a,s') + gamma*V_k(s')]
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT MDP
# ------------------------------------------------------------

from environment import (
    Environment,
    STATES,
    ACTIONS,
    TERMINAL_STATES,
    ROWS,
    COLS,
    WALL,
    GOAL,
    TRAP
)

env = Environment()


# ------------------------------------------------------------
# 2. DISCOUNT FACTOR
# ------------------------------------------------------------

GAMMA = 0.99


# ------------------------------------------------------------
# 3. DEFINE THE POLICY
# ------------------------------------------------------------
#
# A policy pi maps a state to a probability distribution
# over actions.
#
# Here we use a UNIFORM RANDOM POLICY:
#
# pi(a|s) = 1 / |A| = 1/4 = 0.25
#
# For every non-terminal state:
#
# P(UP)    = 0.25
# P(DOWN)  = 0.25
# P(LEFT)  = 0.25
# P(RIGHT) = 0.25
#
# The policy chooses the REQUESTED action.
# The environment determines the ACTUAL movement.

policy = {}

for state in STATES:

    if state in TERMINAL_STATES:
        continue

    policy[state] = {}

    for action in ACTIONS:
        policy[state][action] = 1.0 / len(ACTIONS)


# ------------------------------------------------------------
# 4. INITIALIZE THE STATE-VALUE FUNCTION: V_0(s)
# ------------------------------------------------------------
#
# The expected discounted return starting from state s, following policy pi.
#
# Initially, we don't know the values.
#
# So we begin with:
#
# V_0(s) = 0 for every state s.
#
# These are estimates, not the true values.

V = {}

for state in STATES:
    V[state] = 0.0


# ------------------------------------------------------------
# 5. DEFINE A FUNCTION TO CALCULATE Q^pi(s,a)
# ------------------------------------------------------------
#
# The action-value function is:
#
# Q^pi(s,a) =
#     SUM over s' [
#         P(s'|s,a) * (R(s,a,s') + gamma*V^pi(s'))
#     ]
#
# Interpretation:
#
# 1. Request action a in state s.
# 2. Consider all possible next states s'.
# 3. Calculate immediate reward + discounted future value.
# 4. Multiply each outcome by its probability.
# 5. Sum the results.
#
# This function computes the expected action value using
# our CURRENT estimate of V.

def calculate_q(state, action, values):

    q = 0.0

    # Each outcome contains:
    #
    # probability = P(s'|s,a)
    # next_state  = s'
    # reward      = R(s,a,s')

    for probability, next_state, reward in (
        env.transitions(state, action)
    ):

        # Bellman one-step return:
        #
        # R(s,a,s') + gamma*V(s')

        one_step_return = (
            reward + GAMMA * values[next_state]
        )

        # Expected-value contribution:
        #
        # P(s'|s,a) * [R + gamma*V(s')]

        weighted_return = probability * one_step_return

        # Accumulate the SUM over possible outcomes.
        q += weighted_return

    return q


# ------------------------------------------------------------
# 6. CALCULATE ONE BELLMAN EXPECTATION UPDATE
# ------------------------------------------------------------

def bellman_update(state, values):

    # Terminal states have no future reward.
    #
    # IMPORTANT:
    # The +1 or -1 reward is received when ENTERING them.
    # Their continuation value is therefore zero.

    if state in TERMINAL_STATES:
        return 0.0

    expected_value = 0.0

    for action in ACTIONS:

        # pi(a|s)
        action_probability = policy[state][action]

        # Q_k(s,a)
        action_value = calculate_q(state, action, values)

        # pi(a|s) * Q_k(s,a)
        expected_value += (
            action_probability * action_value
        )

    return expected_value


# ------------------------------------------------------------
# 7. ITERATIVE POLICY EVALUATION
# ------------------------------------------------------------
#
# We repeatedly apply the Bellman expectation operator:
#
# V_(k+1) = T^pi(V_k)
#
# We start with V_0(s) = 0.
#
# Every iteration brings our values closer to V^pi.
#
# We stop when the maximum change is very small:
#
# delta = max over s |V_(k+1)(s) - V_k(s)|
#
# Stop when:
#
# delta < THRESHOLD

THRESHOLD = 0.000001
MAX_ITERATIONS = 100000

iteration = 0

while iteration < MAX_ITERATIONS:

    # Copy current values for synchronous updates.
    #
    # Every state in iteration k+1 must be calculated
    # using values from iteration k.
    #
    # We don't overwrite V until all states are updated.

    new_V = V.copy()

    # Track the largest absolute value change.
    delta = 0.0

    for state in STATES:

        # Bellman expectation update:
        #
        # V_(k+1)(s) = T^pi(V_k)(s)

        new_V[state] = bellman_update(state, V)

        # Change in this state's estimate.
        #
        # |V_(k+1)(s) - V_k(s)|

        change = abs(new_V[state] - V[state])

        # Maximum change across all states.
        delta = max(delta, change)

    # Replace V_k with V_(k+1).
    V = new_V

    iteration += 1

    # Print progress during the first few iterations.
    if iteration <= 5:
        print(
            f"Iteration {iteration}: "
            f"V(2,0) = {V[(2,0)]:.6f}, "
            f"delta = {delta:.6f}"
        )

    # Convergence test.
    if delta < THRESHOLD:
        break


# ------------------------------------------------------------
# 8. DISPLAY THE STATE-VALUES
# ------------------------------------------------------------
#
# Each number represents:
#
# V^pi(s) = expected discounted return from state s
#
# Negative values mean the random policy is expected to accumulate more penalties than rewards.

print("\n" + "=" * 42)
print("POLICY EVALUATION RESULTS")
print("=" * 42)

print(f"Iterations: {iteration}")
print(f"Final delta: {delta:.8f}")
print(f"Gamma: {GAMMA}")

if delta >= THRESHOLD:
    print("Warning: maximum iterations reached.")

print("\nState-value function V^pi(s):\n")

for r in range(ROWS):

    row_values = []

    for c in range(COLS):

        state = (r, c)

        if state == WALL:
            row_values.append("  WALL  ")

        elif state == GOAL:
            row_values.append("  GOAL  ")

        elif state == TRAP:
            row_values.append("  TRAP  ")

        else:
            row_values.append(f"{V[state]:8.4f}")

    print(" ".join(row_values))


# ------------------------------------------------------------
# 9. INSPECT ALL ACTION VALUES AT THE STARTING STATE
# ------------------------------------------------------------
#
# Recall:
#
# V^pi(s) = SUM_a pi(a|s)*Q^pi(s,a)
#
# The policy is random, so it averages the four Q-values.

start_state = (2, 0)

print("\nQ-values at starting state (2,0):\n")

for action in ACTIONS:

    q = calculate_q(start_state, action, V)

    print(f"Q({start_state}, {action:5s}) = {q:.6f}")

print(
    f"\nV^pi({start_state}) = {V[start_state]:.6f}"
)

# V^pi(s) = SUM_a pi(a|s) * Q^pi(s,a)

weighted_average = sum(
    policy[start_state][action]
    * calculate_q(start_state, action, V)
    for action in ACTIONS
)

print(
    "Weighted average of Q-values:",
    f"{weighted_average:.6f}"
)