import numpy as np
from pingpong_env import PingPongEnv
from dqn_agent import DQNAgent
import tensorflow as tf
import threading
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
    except RuntimeError as e:
        print(e)

def train_agent(agent, env, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 400, 300, 3])
        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, 400, 300, 3])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time_step}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 50 == 0:
            agent.save(f"pingpong-dqn-{e}.h5")

def render_env(env):
    while True:
        env.render()
        time.sleep(1/60)

if __name__ == "__main__":
    env = PingPongEnv()
    state_shape = (400, 300, 3)
    action_size = env.action_space.n
    agent = DQNAgent(state_shape, action_size)

    train_thread = threading.Thread(target=train_agent, args=(agent, env))
    train_thread.start()

    render_env(env)
