"""Fullscreen, responsive policy-iteration gridworld visualizer.

Run: python simulation.py
Requires: pip install pygame
Place beside environment.py from the RL tutorial.
Controls: ESC quit, F fullscreen/windowed, SPACE pause, R restart,
          V show/hide values, C show/hide coordinates, UP/DOWN speed.

Math is coded below; Pygame handles only drawing and input.
"""
import math
import sys
import pygame
from environment import (
    Environment, STATES, ACTIONS, TERMINAL_STATES,
    ROWS, COLS, WALLS, GOAL, TRAP, START,
)

GAMMA = 0.9
TOLERANCE = 1e-9
FPS = 240
COLORS = {
    'bg': (13, 18, 31), 'panel': (22, 30, 47), 'cell': (37, 49, 70),
    'line': (62, 79, 103), 'text': (231, 237, 249), 'muted': (149, 164, 188),
    'blue': (92, 169, 255), 'black': (0, 0, 0), 'green': (80, 217, 154), 'red': (255, 108, 126),
    'yellow': (255, 207, 112), 'wall': (86, 96, 112),
}
ARROWS = {'NORTH': 'c', 
'SOUTH': 'd', 
'WEST': 'b', 
'EAST': 'a', 
'NORTHEAST': 'g', 
'NORTHWEST': 'e', 
'SOUTHEAST': 'h', 
'SOUTHWEST': 'f'}
DELTAS = {'NORTH': (-1, 0), 'SOUTH': (1, 0), 'WEST': (0, -1), 'EAST': (0, 1), 'NORTHEAST': (-1, 1), 'NORTHWEST': (-1, -1), 'SOUTHEAST': (1, 1), 'SOUTHWEST': (1, -1)}


def q_value(env, state, action, values):
    """Q(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ V(s')]."""
    return sum(
        probability * (reward + GAMMA * values[next_state])
        for probability, next_state, reward in env.transitions(state, action)
    )


def solve_policy_iteration(env):
    """Alternate Bellman policy evaluation and greedy policy improvement."""
    states = sorted(STATES)
    policy = {s: 'NORTH' for s in states if s not in TERMINAL_STATES}
    values = {s: 0.0 for s in states}
    history = []

    for outer in range(100):
        # Bellman expectation update for deterministic policy π:
        # V_(k+1)(s) = Q_k(s, π(s))
        for sweep in range(10000):
            updated = values.copy()
            for s in policy:
                updated[s] = q_value(env, s, policy[s], values)
            delta = max(abs(updated[s] - values[s]) for s in states)
            values = updated
            if delta < TOLERANCE:
                break

        # Greedy policy improvement: π_new(s) = argmax_a Q^π(s,a)
        changes = 0
        improved = policy.copy()
        for s in policy:
            scores = {a: q_value(env, s, a, values) for a in ACTIONS}
            best = max(scores, key=scores.get)
            improved[s] = best
            changes += (best != policy[s])
        policy = improved
        history.append((outer + 1, sweep + 1, changes))
        if changes == 0:
            break

    # Evaluate final policy again so its displayed values match it.
    for _ in range(10000):
        updated = values.copy()
        for s in policy:
            updated[s] = q_value(env, s, policy[s], values)
        delta = max(abs(updated[s] - values[s]) for s in states)
        values = updated
        if delta < TOLERANCE:
            break
    return policy, values, history


def calculate_layout(size):
    """Fit the whole grid AND a sidebar inside the current display size."""
    w, h = size
    margin = max(12, min(32, w // 55))
    header = 100 if h >= 600 else 65
    footer = 68 if h >= 600 else 42
    panel_width = min(300, max(185, int(w * 0.27)))
    panel_gap = max(12, margin)
    available_w = w - 2 * margin - panel_width - panel_gap
    available_h = h - header - footer - margin
    gap = 0
    cell = max(1, int(min(
        (available_w - gap * (COLS - 1)) / COLS,
        (available_h - gap * (ROWS - 1)) / ROWS,
    )))
    grid_w = COLS * cell + (COLS - 1) * gap
    grid_h = ROWS * cell + (ROWS - 1) * gap
    x0 = margin + max(0, (available_w - grid_w) // 2)
    y0 = header + max(0, (available_h - grid_h) // 2)
    panel_x = w - margin - panel_width
    return dict(w=w, h=h, x0=x0, y0=y0, gap=gap, cell=cell,
                grid_w=grid_w, grid_h=grid_h, panel_x=panel_x,
                panel_width=panel_width, margin=margin, header=header)


def cell_center(state, layout):
    r, c = state
    step = layout['cell'] + layout['gap']
    return (layout['x0'] + c * step + layout['cell'] / 2,
            layout['y0'] + r * step + layout['cell'] / 2)


def interpolate(start, end, t):
    t = max(0.0, min(1.0, t))
    smooth = t * t * (3 - 2 * t)
    return (start[0] + (end[0] - start[0]) * smooth,
            start[1] + (end[1] - start[1]) * smooth)


def draw_text(surface, font, message, position, color='text'):
    img = font.render(str(message), True, COLORS.get(color, color))
    surface.blit(img, position)
    return img


def main():
    pygame.init()
    pygame.display.set_caption('Robot in a Room | Policy Iteration Visualizer')
    desktop = pygame.display.get_desktop_sizes()[0]
    windowed_size = (min(1250, desktop[0]), min(850, desktop[1]))
    fullscreen = True
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    layout = calculate_layout(screen.get_size())
    clock = pygame.time.Clock()

    env = Environment()
    robot_img = pygame.image.load('robot.webp').convert_alpha()
    policy, values, history = solve_policy_iteration(env)
    state = env.reset()
    start_state = state
    target_state = state
    elapsed = 0.0
    duration = 0.55
    moving = False
    paused = False
    show_values = False  # In a large grid, values would otherwise clutter every cell.
    show_coords = False
    speed = 1.0
    episode = 1
    steps = 0
    cumulative = 0.0
    discounted = 0.0
    last_action = '—'
    last_direction = '—'
    last_reward = 0.0
    outcome = 'Navigating'
    trail = [state]
    episode_rest = 0.0
    running = True

    def reset_episode(reset_counter=False):
        nonlocal state, start_state, target_state, elapsed, moving
        nonlocal steps, cumulative, discounted, last_action, last_direction
        nonlocal last_reward, outcome, trail, episode_rest, episode
        state = env.reset()
        start_state = target_state = state
        elapsed = episode_rest = 0.0
        moving = False
        steps = 0
        cumulative = discounted = last_reward = 0.0
        last_action = last_direction = '—'
        outcome = 'Navigating'
        trail = [state]
        if reset_counter:
            episode = 1

    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                layout = calculate_layout(event.size)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
                    layout = calculate_layout(screen.get_size())
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_v:
                    show_values = not show_values
                elif event.key == pygame.K_c:
                    show_coords = not show_coords
                elif event.key == pygame.K_r:
                    reset_episode(reset_counter=True)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_UP):
                    speed = min(10.0, speed * 1.4)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
                    speed = max(0.2, speed / 1.4)

        if not paused:
            if moving:
                elapsed += dt * speed * 5
                if elapsed >= duration:
                    elapsed = duration
                    moving = False
                    state = target_state
            elif env.done:
                episode_rest += dt * speed * 5
                if episode_rest >= 1.3:
                    episode += 1
                    reset_episode()
            else:
                action = policy[state]
                old = state
                next_state, reward, done = env.step(action)
                last_action = action
                if next_state == old:
                    last_direction = 'Stayed (blocked)'
                else:
                    last_direction = next(
                        direction for direction, (dr, dc) in DELTAS.items()
                        if (old[0] + dr, old[1] + dc) == next_state
                    )
                last_reward = reward
                cumulative += reward
                discounted += (GAMMA ** steps) * reward
                steps += 1
                start_state, target_state = old, next_state
                elapsed = 0.0
                moving = True
                trail.append(next_state)
                if done:
                    outcome = 'GOAL REACHED' if next_state == GOAL else 'TRAP REACHED'

        # -------- Responsive drawing: no fixed coordinates inside small cells --------
        w, h = screen.get_size()
        if (w, h) != (layout['w'], layout['h']):
            layout = calculate_layout((w, h))
        screen.fill(COLORS['bg'])
        cw = layout['cell']
        x0, y0 = layout['x0'], layout['y0']
        gap = layout['gap']
        title_font = pygame.font.SysFont('arial', max(19, min(29, w // 35)), bold=True)
        small = pygame.font.SysFont('arial', max(12, min(15, w // 85)))
        info_font = pygame.font.SysFont('arial', max(12, min(17, w // 75)))
        arrow_font = pygame.font.Font('./arrows/Arrows.otf', max(9, min(33, int(cw * 0.62))))
        cell_font = pygame.font.SysFont('arial', max(12, min(17, cw // 5)))
        draw_text(screen, title_font, 'ROBOT IN A ROOM', (layout['margin'], 19))
        if h >= 600:
            draw_text(screen, small, 'Policy iteration  |  stochastic 80/10/10 movement  |  grid: %s x %s' % (ROWS, COLS),
                      (layout['margin'], 61), 'muted')

        for r in range(ROWS):
            for c in range(COLS):
                s = (r, c)
                rect = pygame.Rect(x0 + c * (cw + gap), y0 + r * (cw + gap), cw, cw)
                color = COLORS['cell']
                if s in WALLS:
                    color = COLORS['wall']
                elif s == GOAL:
                    color = (30, 98, 78)
                elif s == TRAP:
                    color = (112, 49, 68)
                pygame.draw.rect(screen, color, rect)
                if cw >= 12:
                    pygame.draw.rect(screen, COLORS['line'], rect, 1)
                if s in WALLS:
                    if cw >= 52:
                        label = cell_font.render('WALL', True, COLORS['text'])
                        screen.blit(label, label.get_rect(center=rect.center))
                elif s in (GOAL, TRAP):
                    if cw >= 19:
                        label = cell_font.render('+10' if s == GOAL else '-10', True,
                                                  COLORS['green'] if s == GOAL else COLORS['red'])
                        screen.blit(label, label.get_rect(center=rect.center))
                else:
                    if cw >= 13:
                        arrow = arrow_font.render(ARROWS[policy[s]], True, COLORS['blue'])
                        screen.blit(arrow, arrow.get_rect(center=rect.center))
                    if show_values and cw >= 64:
                        label = cell_font.render(f'{values[s]:.2f}', True, COLORS['muted'])
                        screen.blit(label, label.get_rect(midtop=(rect.centerx, rect.y + 3)))
                    if show_coords and cw >= 78:
                        label = cell_font.render(str(s), True, COLORS['muted'])
                        screen.blit(label, label.get_rect(midbottom=(rect.centerx, rect.bottom - 3)))

        if len(trail) > 1 and cw >= 6:
            # Draw transparent-looking thin lines so the route does not bury arrows.
            for a, b in zip(trail[:-1], trail[1:]):
                if a != b:
                    pygame.draw.line(screen, (100, 118, 148), cell_center(a, layout),
                                     cell_center(b, layout), max(1, cw // 16))
        current_pos = interpolate(cell_center(start_state, layout),
                                  cell_center(target_state, layout),
                                  elapsed / duration) if moving else cell_center(state, layout)

        px, py = round(current_pos[0]), round(current_pos[1])
        robot_size = max(4, int(cw * 0.8))
        robot_scaled = pygame.transform.smoothscale(
            robot_img, (robot_size, robot_size)
        )
        robot_rect = robot_scaled.get_rect(center=(px, py))
        screen.blit(robot_scaled, robot_rect)

        panel_x = layout['panel_x']
        panel_w = layout['panel_width']
        panel_top = layout['header']
        panel_h = min(455, max(340, h - panel_top - 75))
        pygame.draw.rect(screen, COLORS['panel'],
                         (panel_x, panel_top, panel_w, panel_h), border_radius=12)
        panel_padding = 16
        draw_text(screen, info_font, 'LIVE EPISODE', (panel_x + panel_padding, panel_top + 15), 'blue')
        info = [
            ('Status', outcome), ('Episode', episode), ('Steps', steps),
            ('Requested', last_action), ('Actual', last_direction),
            ('Last reward', f'{last_reward:+.2f}'),
            ('Total reward', f'{cumulative:+.2f}'),
            ('Discounted G', f'{discounted:+.3f}'), ('Speed', f'{speed:.1f}x'),
        ]
        line_height = max(27, min(36, (panel_h - 65) // len(info)))
        for i, (label, value) in enumerate(info):
            y = panel_top + 52 + i * line_height
            draw_text(screen, small, label, (panel_x + panel_padding, y), 'muted')
            rendered = small.render(str(value), True, COLORS['text'])
            max_value_w = int(panel_w * 0.58)
            if rendered.get_width() > max_value_w:
                # Keep values inside panel instead of covering the label.
                short_value = str(value)
                while len(short_value) > 1 and small.size(short_value + '…')[0] > max_value_w:
                    short_value = short_value[:-1]
                rendered = small.render(short_value + '…', True, COLORS['text'])
            screen.blit(rendered, (panel_x + panel_w - panel_padding - rendered.get_width(), y))

        # Bottom help bar sits below the grid, not at a fixed y=560.
        footer_y = min(h - 48, y0 + layout['grid_h'] + 12)
        draw_text(screen, small,
                  f'Policy iteration: {len(history)} rounds   |   gamma={GAMMA}   |   '
                  f'V=values  C=coordinates  SPACE=pause  R=reset  F=fullscreen  ESC=quit',
                  (layout['margin'], footer_y), 'muted')
        if footer_y + 21 < h:
            draw_text(screen, small, 'UP/DOWN: animation speed. Values and coordinates are hidden on small cells.',
                      (layout['margin'], footer_y + 20), 'muted')
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()