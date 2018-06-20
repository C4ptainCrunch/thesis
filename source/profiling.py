import matplotlib.pyplot as plt # noqa

import import_ipynb  # noqa
from rules import Game
from players import GreedyUCTPlayer


def play_game(player, opponent):
    game = Game.start_game()
    opponent_action = -1

    while not game.game_finished:
        player_action = player.play(opponent_action)
        game, captures, finished = game.step(player_action)

        player, opponent = opponent, player
        opponent_action = player_action
    return game


print("Starting run")

BUDGET = 50

player = GreedyUCTPlayer(0, BUDGET)
opponent = GreedyUCTPlayer(1, BUDGET)

play_game(player, opponent)
