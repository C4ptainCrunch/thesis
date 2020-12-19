import json
import logging
import math
import os
import time
from datetime import timedelta
from datetime import timedelta as td

import import_ipynb
import numpy as np
from index import Game
from index import GreedyPlayer, MCTSPlayer, UCTPlayer

logging.basicConfig(
    format="%(asctime)s %(levelname)8s | %(message)s", level=logging.DEBUG
)

space = np.linspace(0, 1, 21)
for i in range(50):
    for eps1 in space:
        for eps2 in space:
            if eps1 > eps2:
                pass
            for side in range(2):
                player_tpl= f"GreedyPlayer(%s, {eps1})"
                opponent_tpl = f"GreedyPlayer(%s, {eps2})"

                if side == 1:
                    player_tpl, opponent_tpl = opponent_tpl, player_tpl

                player_str = player_tpl % '0'
                opponent_str = opponent_tpl % '1'


                logging.debug("Starting a new match")


                logging.info("Opposing players")
                logging.info("%s", player_str)
                logging.info("%s", opponent_str)


                player = eval(player_str)
                opponent = eval(opponent_str)

                game = Game()
                opponent_action = -1
                depth = 0

                start = time.perf_counter()
                while not game.game_finished and depth < 500:
                    player_action = player.play(opponent_action)
                    game, captures, finished = game.step(player_action)

                    player, opponent = opponent, player
                    opponent_action = player_action
                    depth += 1
                duration = round(time.perf_counter() - start, 4)

                pace = duration / depth
                logging.info("Game finished in %.2fs and %s turns (%.2fs/turn)", duration, depth, pace)

                logging.info("Final score %d - %d", game.captures[0], game.captures[1])
                if game.winner:
                    logging.info(
                        "Winner is %s: %s",
                        game.winner,
                        player_str if game.winner == 0 else opponent_str,
                    )
                else:
                    logging.info("Darw: no winner")


                print(
                    json.dumps(
                        {
                            "player": player_str,
                            "opponent": opponent_str,
                            "duration": duration,
                            "depth": depth,
                            "score": game.captures.tolist(),
                            "winner": game.winner,
                            "pool": "local-eps-matrix",
                            "side": side,
                            "success": True,
                            "version": 2,
                        }
                    )
                )
