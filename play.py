# play.py
import os
import pygame
import torch
from utils import set_seed
from env import GridEnv
from agent import Agent
from config import (
    MAX_STEPS, SCREEN_SIZE, CELL_PIXELS, ASSET_DIR, GRID_SIZE,
    CAT_IMG, FOOD_IMG, GROUND_IMG, WALL_IMG, WATER_IMG, LAVA_IMG
)

def load_assets():
    def _load(name):
        img = pygame.image.load(os.path.join(ASSET_DIR, name))
        return pygame.transform.scale(img, (CELL_PIXELS, CELL_PIXELS))
    cat   = _load(CAT_IMG)
    cat_l = pygame.transform.flip(cat, True, False)
    return {
        "cat_r": cat, "cat_l": cat_l,
        "food": _load(FOOD_IMG),
        "ground": _load(GROUND_IMG),
        "wall": _load(WALL_IMG),
        "water": _load(WATER_IMG),
        "lava": _load(LAVA_IMG),
    }

def main():
    set_seed()
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock  = pygame.time.Clock()

    assets = load_assets()
    font   = pygame.font.SysFont(None, 24)

    # create env & agent, load weights
    env   = GridEnv()
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, device)
    agent.policy_net.load_state_dict(torch.load("dqn_cat_model.pth", map_location=device))
    agent.policy_net.eval()

    for ep in range(1, 11):  # run 10 demo episodes
        state   = env.reset()
        total_r = 0.0

        for step in range(1, MAX_STEPS + 1):
            # greedy play
            action = agent.select_action(state, eps=0.2)
            state, reward, done = env.step(action)
            total_r += reward

            # draw grid
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    screen.blit(assets["ground"], (x*CELL_PIXELS, y*CELL_PIXELS))
            for (ox,oy) in env.obstacles: screen.blit(assets["wall"], (ox*CELL_PIXELS, oy*CELL_PIXELS))
            for (wx,wy) in env.water:     screen.blit(assets["water"], (wx*CELL_PIXELS, wy*CELL_PIXELS))
            for (lx,ly) in env.lava:      screen.blit(assets["lava"], (lx*CELL_PIXELS, ly*CELL_PIXELS))

            fx,fy = env.food; screen.blit(assets["food"], (fx*CELL_PIXELS, fy*CELL_PIXELS))
            cx,cy = env.cat
            screen.blit(assets["cat_l"] if action==2 else assets["cat_r"],
                        (cx*CELL_PIXELS, cy*CELL_PIXELS))

            info = font.render(f"Play Ep {ep}/10  Step {step}  R {total_r:.1f}", True, (0,0,0))
            screen.blit(info, (5,5))
            pygame.display.flip()
            clock.tick(10)

            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    pygame.quit()
                    return

            if done:
                break

        print(f"[Demo Ep {ep:2d}] Reward = {total_r:.1f}")

    pygame.quit()

if __name__ == "__main__":
    main()
