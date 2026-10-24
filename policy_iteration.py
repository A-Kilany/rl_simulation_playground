
# ============================================================
# POLICY ITERATION
# ============================================================
#
# Goal:
# Find an optimal policy pi* for our Markov Decision Process.
#
# Algorithm:
#
# 1. Initialize a policy pi_0.
# 2. Evaluate pi_k to calculate V^(pi_k).
# 3. Improve the policy using its Q-values.
# 4. Repeat until the policy is stable.
#
# The two mathematical equations:
#
# POLICY EVALUATION:
#
# V^pi(s) = SUM_a pi(a|s) *
#              SUM_s' P(s'|s,a) *
#              [R(s,a,s') + gamma*V^pi(s')]
#
# POLICY IMPROVEMENT:
#
# pi_new(s) = argmax_a Q^pi(s,a)
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT THE ENVIRONMENT
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
    TRAP,
    START
)

env = Environment()

GAMMA = 0.9
THRESHOLD = 0.000001

# Sorting states makes iteration order reproducible.
states = sorted(STATES)

# List of available actions.
actions = list(ACTIONS)


# ------------------------------------------------------------
# 2. INITIALIZE THE POLICY: pi_0
# ------------------------------------------------------------
#
# This time, we'll represent a DETERMINISTIC policy.
#
# A deterministic policy maps each state to ONE action:
#
# pi(s) = a
#
# For example:
#
# pi((2,0)) = "UP"
#
# This is mathematically equivalent to:
#
# pi(UP|(2,0)) = 1
# pi(a|(2,0)) = 0 for all other actions.
#
# We initially choose UP everywhere.
#
# This is arbitrary. Policy iteration will improve it.

policy = {}

for state in states:

    if state in TERMINAL_STATES:
        continue

    policy[state] = "UP"


# ------------------------------------------------------------
# 3. INITIALIZE THE STATE-VALUE FUNCTION
# ------------------------------------------------------------
#
# V(s) = Expected discounted return from state s.
#
# Initially:
#
# V_0(s) = 0 for every state.
#
# Terminal states have continuation value zero because
# the reward is received upon entering them.

V = {}

for state in states:
    V[state] = 0.0


# ------------------------------------------------------------
# 4. CALCULATE THE ACTION-VALUE FUNCTION Q(s,a)
# ------------------------------------------------------------
#
# Mathematical equation:
#
# Q^pi(s,a) =
#     SUM_s' P(s'|s,a) *
#         [R(s,a,s') + gamma*V^pi(s')]
#
# This computes the expected return from:
#
# 1. Taking action a in state s.
# 2. Following the current policy afterward.
#
# The state-value estimates are passed as an argument.

def calculate_q(state, action, values):

    q = 0.0

    # Inner Bellman summation over next-state outcomes.
    for probability, next_state, reward in (
        env.transitions(state, action)
    ):

        # Immediate reward plus discounted future value:
        #
        # R(s,a,s') + gamma*V(s')

        one_step_return = (
            reward + GAMMA * values[next_state]
        )

        # Weight by transition probability P(s'|s,a).
        q += probability * one_step_return

    return q


# ------------------------------------------------------------
# 5. POLICY EVALUATION
# ------------------------------------------------------------
#
# For a deterministic policy:
#
# pi(a|s) = 1 for a = pi(s), otherwise 0.
#
# Therefore, the Bellman expectation equation simplifies:
#
# V^pi(s) =
#     SUM_s' P(s'|s,pi(s)) *
#       [R(s,pi(s),s') + gamma*V^pi(s')]
#
# Or more compactly:
#
# V^pi(s) = Q^pi(s,pi(s))
#
# We repeatedly update V until the changes are small.

def evaluate_policy(policy, values):

    iteration = 0
    max_iterations = 100000

    while iteration < max_iterations:

        # Use synchronous Bellman updates:
        #
        # Every V_(k+1)(s) is calculated using V_k.
        new_V = values.copy()

        delta = 0.0

        for state in states:

            # Once a terminal state is reached,
            # no additional rewards are collected.
            if state in TERMINAL_STATES:
                new_V[state] = 0.0
                continue

            # Action selected by our current policy:
            #
            # a = pi(s)
            action = policy[state]

            # Bellman expectation update:
            #
            # V_(k+1)(s) = Q_k(s,pi(s))
            new_V[state] = calculate_q(
                state, action, values
            )

            # Calculate absolute change:
            #
            # |V_(k+1)(s) - V_k(s)|
            change = abs(
                new_V[state] - values[state]
            )

            delta = max(delta, change)

        # Replace old state values.
        values = new_V

        iteration += 1

        # Convergence test:
        #
        # max_s |V_(k+1)(s) - V_k(s)| < threshold
        if delta < THRESHOLD:
            break

    return values, iteration, delta


# ------------------------------------------------------------
# 6. POLICY IMPROVEMENT
# ------------------------------------------------------------
#
# This is the new mathematical operation.
#
# pi_new(s) = argmax_a Q^pi(s,a)
#
# For every state:
#
# 1. Calculate Q(s,UP).
# 2. Calculate Q(s,DOWN).
# 3. Calculate Q(s,LEFT).
# 4. Calculate Q(s,RIGHT).
# 5. Choose the action with the largest Q-value.
#
# The argmax operator returns the ACTION, not its value.

def improve_policy(policy, values):

    new_policy = policy.copy()

    # Assume the policy is stable until a change is found.
    policy_stable = True

    changes = 0

    for state in states:

        if state in TERMINAL_STATES:
            continue

        old_action = policy[state]

        # Calculate Q(s,a) for every action.
        q_values = {}

        for action in actions:
            q_values[action] = calculate_q(
                state, action, values
            )

        # Mathematical argmax:
        #
        # a* = argmax_a Q(s,a)
        #
        # key=q_values.get tells max() to compare
        # dictionary VALUES rather than action names.
        best_action = max(
            actions,
            key=q_values.get
        )

        # Update the policy:
        #
        # pi_new(s) = a*
        new_policy[state] = best_action

        # If an action changes, the policy is not stable.
        if best_action != old_action:
            policy_stable = False
            changes += 1

    return new_policy, policy_stable, changes


# ------------------------------------------------------------
# 7. DISPLAY FUNCTIONS
# ------------------------------------------------------------
#
# These functions do not perform reinforcement learning.
# They simply print the state values and policy.

ARROWS = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}


def print_values(values):

    print("\nSTATE VALUES:\n")

    for r in range(ROWS):

        row = []

        for c in range(COLS):

            state = (r, c)

            if state == WALL:
                cell = "  WALL  "

            elif state == GOAL:
                cell = "  GOAL  "

            elif state == TRAP:
                cell = "  TRAP  "

            else:
                cell = f"{values[state]:8.4f}"

            row.append(cell)

        print(" ".join(row))


def print_policy(policy):

    print("\nPOLICY:\n")

    for r in range(ROWS):

        row = []

        for c in range(COLS):

            state = (r, c)

            if state == WALL:
                cell = " ■ "

            elif state == GOAL:
                cell = " +10"

            elif state == TRAP:
                cell = " -10"

            else:
                cell = f" {ARROWS[policy[state]]} "

            row.append(cell)

        print(" ".join(row))


# ------------------------------------------------------------
# 8. MAIN POLICY ITERATION ALGORITHM
# ------------------------------------------------------------
#
# Initialize pi_0 and V_0.
#
# Repeat:
#
#   A. Policy evaluation:
#      V^(pi_k) = evaluate(pi_k)
#
#   B. Policy improvement:
#      pi_(k+1)(s) = argmax_a Q^(pi_k)(s,a)
#
# Until:
#
#   pi_(k+1) = pi_k
#
# When no actions change, the policy is stable.
#
# For this finite discounted MDP, a stable greedy policy
# is optimal (up to numerical evaluation tolerance).

def policy_iteration():

    global policy, V

    outer_iteration = 0
    max_outer_iterations = 1000

    print("=" * 50)
    print("POLICY ITERATION")
    print("=" * 50)

    print("\nINITIAL POLICY:")
    print_policy(policy)

    while outer_iteration < max_outer_iterations:

        outer_iteration += 1

        print(
            f"\n{'=' * 50}\n"
            f"POLICY ITERATION {outer_iteration}\n"
            f"{'=' * 50}"
        )

        # STEP A: POLICY EVALUATION
        #
        # Calculate V^(pi_k).
        V, evaluation_iterations, delta = evaluate_policy(
            policy, V
        )

        print(
            f"\nEvaluation converged after "
            f"{evaluation_iterations} Bellman sweeps."
        )

        print(f"Final delta: {delta:.8f}")

        print_values(V)

        # STEP B: POLICY IMPROVEMENT
        #
        # pi_(k+1)(s) = argmax_a Q^(pi_k)(s,a)
        policy, stable, changes = improve_policy(
            policy, V
        )

        print(
            f"\nPolicy improvement changed "
            f"{changes} actions."
        )

        print_policy(policy)

        # STOPPING CONDITION:
        #
        # pi_(k+1) = pi_k
        if stable:

            print("\nOPTIMAL POLICY FOUND!")
            print(
                f"Policy iterations: {outer_iteration}"
            )
            break

    else:
        print("\nMaximum policy iterations reached.")

    return policy, V


# ------------------------------------------------------------
# 9. TEST THE OPTIMAL POLICY IN THE ENVIRONMENT
# ------------------------------------------------------------
#
# Policy iteration calculated the best REQUESTED actions.
#
# But the environment is still stochastic.
#
# Following the optimal policy does not guarantee
# exactly the same trajectory every time.
#
# Here we simulate one episode:
#
# S_t -> A_t = pi(S_t)
#     -> (S_(t+1), R_(t+1), done)
#
# We also calculate the actual discounted return:
#
# G_0 = SUM_t gamma^t * R_(t+1)

def simulate_episode(policy):

    state = env.reset()

    total_reward = 0.0
    discounted_return = 0.0

    max_steps = 100

    print("\nSIMULATING OPTIMAL POLICY:\n")

    for t in range(max_steps):

        # Choose the action prescribed by the policy.
        action = policy[state]

        # Execute one stochastic environment step.
        next_state, reward, done = env.step(action)

        # Undiscounted cumulative reward:
        total_reward += reward

        # Discounted return:
        #
        # G_0 = SUM_t gamma^t * R_(t+1)
        discounted_return += (GAMMA ** t) * reward

        print(
            f"Step {t + 1:02d}: "
            f"{state} -> {action:5s} -> "
            f"{next_state} | R = {reward:+.2f}"
        )

        state = next_state

        if done:
            print("\nEpisode finished!")
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    print(
        f"Discounted return: {discounted_return:.4f}"
    )


# ------------------------------------------------------------
# 10. EXECUTE THE ALGORITHM
# ------------------------------------------------------------
#
# __name__ == "__main__" means:
#
# Run this section only when executing this file directly.
#
# It will not run when another script imports this file.

if __name__ == "__main__":

    optimal_policy, optimal_values = policy_iteration()

    simulate_episode(optimal_policy)