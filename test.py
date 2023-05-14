from model.net import DQN
from tankbattle.env.engine import TankBattle
from tankbattle.env.utils import Utils

max_iterator = 100


def test(game, net, init_state, epsilon, device):
    total_reward = 0.0
    state = init_state
    while True:
        game.render()
        action = net.act(state, epsilon, device)
        next_state, reward, done, _ = game.step(action)
        next_state = Utils.resize_image(next_state)
        reward = reward[0]

        total_reward += reward

        if done:
            print(total_reward)
            break
        state = next_state


if __name__ == "__main__":
    game = TankBattle(render=True, player1_human_control=False, player2_human_control=False, two_players=False,
                      speed=60, debug=False, frame_skip=5)

    for i in range(max_iterator):
        init_state = game.reset()
        state = Utils.resize_image(init_state)
        origin_net = DQN(input_shape=state.shape, num_actions=game.get_num_of_actions())
        net = Utils.load_model(origin_net, "output/predicted_402_0.7647789637410702_50.pth")
        test(game, net, state, 0.7647789637410702, "cpu")


