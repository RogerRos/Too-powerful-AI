import pygame
import sys
import gym
import numpy as np
from gym import spaces

class PingPongEnv(gym.Env):
    def __init__(self):
        super(PingPongEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(400, 300, 3), dtype=np.uint8)

        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Ping Pong")
        self.clock = pygame.time.Clock()

        self.ball_radius = 7
        self.ball_speed_x = 5
        self.ball_speed_y = 5
        self.ball_pos = [200, 150]

        self.paddle_width = 5
        self.paddle_height = 50
        self.paddle_speed = 10

        self.player1_pos = [5, 125]
        self.player2_pos = [390, 125]

        self.player1_score = 0
        self.player2_score = 0

        self.font = pygame.font.Font(None, 36)

    def reset(self):
        self.ball_pos = [200, 150]
        self.ball_speed_x = 5
        self.ball_speed_y = 5
        self.player1_pos = [5, 125]
        self.player2_pos = [390, 125]
        self.player1_score = 0
        self.player2_score = 0
        return self._get_obs()

    def step(self, action):
        if action == 1:
            self.player1_pos[1] -= self.paddle_speed
        elif action == 2:
            self.player1_pos[1] += self.paddle_speed

        if self.ball_pos[1] < self.player2_pos[1]:
            self.player2_pos[1] -= self.paddle_speed
        if self.ball_pos[1] > self.player2_pos[1] + self.paddle_height:
            self.player2_pos[1] += self.paddle_speed

        self.ball_pos[0] += self.ball_speed_x
        self.ball_pos[1] += self.ball_speed_y

        if self.ball_pos[1] - self.ball_radius <= 0 or self.ball_pos[1] + self.ball_radius >= 300:
            self.ball_speed_y = -self.ball_speed_y

        if self.ball_pos[0] - self.ball_radius <= self.player1_pos[0] + self.paddle_width and \
           self.player1_pos[1] <= self.ball_pos[1] <= self.player1_pos[1] + self.paddle_height:
            self.ball_speed_x = -self.ball_speed_x
        if self.ball_pos[0] + self.ball_radius >= self.player2_pos[0] and \
           self.player2_pos[1] <= self.ball_pos[1] <= self.player2_pos[1] + self.paddle_height:
            self.ball_speed_x = -self.ball_speed_x

        reward = 0
        done = False
        if self.ball_pos[0] - self.ball_radius <= 0:
            reward = -1
            done = True
        if self.ball_pos[0] + self.ball_radius >= 400:
            reward = 1
            done = True

        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self):
        obs = np.array(pygame.surfarray.array3d(self.screen), dtype=np.uint8)
        return obs

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (*self.player1_pos, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255, 255, 255), (*self.player2_pos, self.paddle_width, self.paddle_height))
        pygame.draw.circle(self.screen, (255, 255, 255), self.ball_pos, self.ball_radius)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
        sys.exit()
