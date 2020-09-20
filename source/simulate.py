import math
from datetime import timedelta

import import_ipynb
from index import Game
from players import GreedyPlayer, MCTSPlayer, UCTPlayer


def play_game(player, opponent):
    game = Game()
    opponent_action = -1
    i = 0

    while not game.game_finished and i < 500:
        player_action = player.play(opponent_action)
        game, captures, finished = game.step(player_action)

        player, opponent = opponent, player
        opponent_action = player_action
        i += 1

    return game


player = UCTPlayer(0, timedelta(seconds=0.2), c=math.sqrt(2) / 2)
opponent = UCTPlayer(1, timedelta(seconds=0.2), c=1)
game = play_game(player, opponent)
print(game.winner)
