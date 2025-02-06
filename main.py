import random
from uno_env import UNOGame
from render import ASCIIUnoRenderer


if __name__ == "__main__":
    env = UNOGame(num_players=2, renderer=ASCIIUnoRenderer())
    obs = env.reset()

    for episode in range(2):
        print(f"=== Episode {episode+1} ===")
        done = False
        env.render()

        while not done:
            # 获取合法动作
            current_player = env.players[env.current_player_idx]
            valid_actions = [108]  # 摸牌动作
            for action in range(108):
                if env._is_valid_action(action, current_player):
                    valid_actions.append(action)

            # 简单用户输入
            if env.current_player_idx == 0:  # 人类玩家
                choice = env.renderer.user_input(env, valid_actions)
                action = valid_actions[choice]
            else:  # AI随机选择
                action = random.choice(
                    valid_actions[1:] if len(valid_actions) > 1 else valid_actions
                )  # 优先选择出牌

            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                winner = -1
                for i, player in enumerate(env.players):
                    if len(player.hand) == 0:
                        winner = i
                        break
                if winner >= 0:
                    print(f"Game Over! Player {winner+1} wins!")
                else:
                    print("Game Over! Draw!")
                input("Press Enter to continue...")
                break
        env.reset()
