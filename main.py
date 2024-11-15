import gym_cutting_stock
import gymnasium as gym
import logging
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # Logger mode selection
    mode = int(input("Choose logger mode: \n1. INFO\n2. DEBUG\nEnter choice: "))
    if mode == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    # Policy selection
    mode = int(input("Choose mode: 1. GreedyPolicy, 2. RandomPolicy, 3. Your policy: "))
    if mode == 1:
        policy = GreedyPolicy()
    elif mode == 2:
        policy = RandomPolicy()
    else:
        policy = Policy2210xxx()

    # Reset the environment for the first episode
    observation, info = env.reset(seed=42)

    # Validation log for initial products
    for idx, product in enumerate(observation['products']):
        logger.info(f"Initial Product {idx}: Size={product['size']}, Quantity={product['quantity']}")

    # Validation log for initial stocks
    logger.info(f"Initial number of stocks: {len(observation['stocks'])}")

    # Validation log for initial observation
    logger.debug(f"Initial observation: {observation}")

    ep = 0
    while ep < NUM_EPISODES:
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        logger.debug("Step info:")
        logger.debug(f"- Action: {action}")
        logger.debug(f"- Remaining products: {len([p for p in observation['products'] if p['quantity'] > 0])}")
        logger.debug(f"- Available stocks: {len(observation['stocks'])}")

        if terminated:
            logger.info(f"Episode {ep} terminated:")
            logger.info(f"- Final filled ratio: {info.get('filled_ratio', 'unknown')}")
            logger.info(
                f"- Reason: {'No more products' if not len([p for p in observation['products'] if p['quantity'] > 0]) else 'Cannot place products'}")

        if truncated:
            logger.info(f"Episode {ep} truncated:")
            logger.info(f"- Steps taken: {info.get('steps', 'unknown')}")

        if terminated or truncated:
            # Reset the environment for the next episode
            observation, info = env.reset(seed=ep)

            # Validation log for initial products
            for idx, product in enumerate(observation['products']):
                logger.info(f"Initial Product {idx}: Size={product['size']}, Quantity={product['quantity']}")

            # Validation log for initial stocks (optional)
            logger.info(f"Initial number of stocks: {len(observation['stocks'])}")

            # Validation log for initial observation
            logger.debug(f"Initial observation: {observation}")

            ep += 1

env.close()
