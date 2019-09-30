========================================
Playing Mancala with MCTS and Alpha Zero
========================================

.. toctree::
   :maxdepth: 2


.. topic:: About this work

    This work is a thesis submitted to the Faculty of Sciences in partial
    fulfillment of the requirements for the degree of Master’s in Computer Science.

    This work was originally published on https://mancala.ml as a web page consisting in text
    and notebook cells mixed together.

    The source of this document along with the source code of all experiments
    are available on GitHub_.

    .. _GitHub: https://github.com/C4ptainCrunch/thesis

Mancala
-------

Mancala is a ancient family of games played played on many continents :cite:`deVoogt2008`.
The word mancala comes from the Arabic word "نقلة"transliterated "naqala"
meaning literally "to move". Mancala games usually consists of two
row of pits each containing a proportionate amount of seeds,
stones or shells. Usually, these games are played by two opponents who play sequentially.
The goal for each opponent is to capture as many seeds as possible before the other.

.. figure:: _static/intro-kalah.jpg

  A wooden Mancala game [#source_kalah]_

We will focus on Awalé (also sometimes called Oware,  Owari or Ayo), originating from
Ghana. There are too many other existing variations to list them all here, but a
few notable ones are Wari, Bao, Congkak and Kalah, a modern version invented by
William Julius Champion Jr. circa 1940.

.. [#source_kalah] Picture by Adam Cohn under Creative Commonds license https://www.flickr.com/photos/adamcohn/3076571304/


Awalé
-----

The subject of our study, Awalé is played on a board made of two row of six
pits. Each row is owned by a player. In the initial state of the game every hole
contains 4 seeds thus the game contains 48 seeds in total.

.. figure:: _static/awale.jpg

   A typical Awalé board in the start position.

The goal for both players is to capture more seeds than its opponent. As the
game has 48 seeds, capturing 25 is enough to win and end the game.

Each player plays alternatively, without the right to pass their turn. A
player's turn consists in choosing one of his non-empty pits, pick all seeds
contained in it and seed them one by one in every consecutive pits on the right
(rotating counter-clockwise). The player thus has at most 6 possible moves at
each turn.

If the pit chosen by the player contains more than 12 seeds, the sowing makes
more than a full circle and the starting hole is skipped during the second pass.
When the last sowed seed is placed in a pit that was containing no seed, more
than 2 seeds or in the current player's pits, the turn of the player is ended.

Otherwise, when the last sowed seed is placed in a pit owned by the opponent and
the pit contains then two or three seeds, the content of the pit is captured by
the player and removed of the game. If the pit preceding the captured pit also
contains two or three seeds, it is also captured. The capture continues until a
pit without two or three seeds is encountered. When the capture is ended the
next player's turn starts.

If the current player's opponent has no seed left in his half of the board, the
current player has to play a move that gives him seeds if such a move exists.
This rule is called the "let the opponent play".

When a player has captured more than 25 seeds the game ends and he wins. If both
players have captured 24 seeds, the game ends by a draw. If the current player's
pits are all empty, the game ends and the player with the most captures wins.
The last way to stop the game is when a position is encountered twice in the
same game (there is a cycle): the game ends and player with the most captures
wins.

Perfect information games
-------------------------

Now that we know the rules, we can see that Mancala games are :

* Sequential: the opponents play one after the other,
* Hold no secret information: each player has the same information about
  the game as the other
* Do not rely on randomness: the state of the game depends only on the actions
  taken sequentially by each player and an action has a deterministic result.

This type of game is called a sequential perfect information game
:cite:`osborne1994course`.

Other games in this category are for example Chess, Go, Checkers or even
Tic-tac-toe and Connect Four. This type of game is a particularly interesting
field to study in computer science and artificial intelligence as they are easy
to simulate.


Perfect information games as finite state machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO formal definition of FSM ?

When viewed from an external point of view, these types of games can be
modelized as finite states machines with boards being states (the initial board
is the initial state), each player's action being transitions and wins and draws
being terminal states.

.. TODO formal description of the game as a FSM ?

It might be tempting to try to enumerate every possible play of those games by
starting a game and recursively try each legal action until the end of the play
to find the best move for each state.

Unfortunately, most of the time, this is not a feasible approach due to the size
of the state space. As an example, Romein et al. claims that Awalé has
889,063,398,406 legal positions :cite:`romein2003solving` and the exact number
(:math:`\approx 2.08 \times 10^{170}`) of legal positions in Go is so big that
it has only recently been determined :cite:`tromp2016`. Such state space are too
big to be quickly enumerated.

Perfect information games as Markov decision processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of being viewed from an external point of view, these types of games can
also be seen from the point of view of a single player. He only knows the state
of the board and his own moves and is not aware of the moves from his opponent,
neither in advance or after the move has been played.

When viewed under this angle, a game looks like this:
 * the game is in a state :math:`A`,
 * the player plays his turn, the board changes deterministically,
 * the game is in state :math:`A'`,
 * his opponent plays and the board has multiple ways of changing,
 * the game is in state :math:`B`, :math:`B` is one of the 6 possible successors
   of :math:`A'`.

We can model this as a Markov decision process (MDP).

.. TODO More on MDP and why it is a MDP.



Solved games
------------



A strongly solved game is defined by Allis :cite:`Allis94searchingfor` as:

    For all legal positions, a strategy has been determined to
    obtain the game-theoretic value of the position, for both players, under
    reasonable resources.

A solved game is, of course much less interesting to study than an
unsolved one as we could just create an agent that has the knowledge of each
game-theoretic position values and can thus play perfectly.

Unfortunately for us, (:math:`m,n`)-Kalah (:math:`m` pits per side and :math:`n`
stones in each pit) has been solved in 2000 for :math:`m \leq 6`  and :math:`n
\leq 6` except (:math:`6,6`) by Jos Uiterwijk :cite:`irving2000solving` and in
2011 for :math:`n = 6, m=6` by Anders Carstensen :cite:`kalah66`.

J. W. Romein et al. :cite:`romein2003solving` also claims to have solved
Awalé by quasi-*brute-force* -- retrograde analysis,
but this claim has since been challenged by others like Víktor Bautista i Roca.
Roca claims that several endgames were incorrect and the results are invalid.
As both the database made by Romein and the claim from Roca are not available
anymore publicly we can not know who is right.

Nevertheless, these proofs for Kalah and Awalé both use a quasi-*brute-force*
method to solve the game and uses a database all possible states. The database
used by Romein et al. has 204 billion entries and weighs 178GiB. A database so
huge is of course not practical so we think that there is still room for
improvement if we can create an agent that has a policy that does not need a
exhaustive database, even if the agent is not capable of a perfect play.

We arbitrarily chose to work on Awalé as it might not have been solved but
the same work could most probably be done on Kalah and other variants.



.. .. topic:: How should you read this document ?
..
..     This document is a mix of text and Python code in the form of notebook
..     cells. Reading only the text and skipping all the code should be enough for
..     you to understand the whole work. But if you are interested in the
..     implementation work and the details of the simulations you are welcome to
..     read the notebook cells as well.
..
..     Some output and cells are hidden for the sake of brevity and readability.
..     Click on the button to reveal the full code and output that were used for
..     the simulations to write this work.

.. .. include:: nb_builds/rules.rst

Monte Carlo tree search
-----------------------

Many algorithms have been proposed and studied to play sequential
perfect information games.
A few examples are :math:`\alpha-\beta` pruning, Minimax,
Monte Carlo tree search (MCTS) and Alpha (Go) Zero :cite:`AlphaGoZero`.

We will focus on MCTS as it does not require any expert knowledge
about the given game to make reasonable decisions.

The principle of MCTS is simple : we represent the starting state of a game by
the root node of a tree. This node then has a children for each possible action
the current player can make. The n-th child of the node represents the state in
which the game would be if the payer had played the n-th possible action.

The maximum number of children of a node in the game is called the branching
factor. In a classical Awalé game the player can choose to sow his seeds from
one of his non-empty pits. As the player has 6 pits, the branching factor is 6
(this is very small compared to branching factor of 19 from the game of Go and
makes Awalé much easier to play with this method).

With this representation, if we build the complete tree, we will have computed
every possible state in the game and every leaf of the tree will be a final
state (end of a game). As said, previously, computing the complete tree is not
ideal for Alawé (it has :math:`\approx 8 \times 10^{11}` nodes) and
computationally impossible for games with a high branching factor.

To overcome this computational problem, the MCTS method constructs only a part
of the tree by sampling and tries to estimate the chance of winning based on
this information.

.. figure:: _static/mcts-algorithm.png

   The 4 steps of MCTS :cite:`chaslot2008monte`


The (partial) tree is constructed as follows:

* Selection: starting at the root node, recursively choose a child until
  a leaf :math:`L` is reached
* Expansion: if :math:`L` is not a terminal node\footnote{As the tree is
  not complete, a leaf could be a node that is missing its children, not
  necessarily a terminal state}, create a child :math:`C`
* Simulation: run a playout from :math:`C` until a terminal node :math:`T` is
  reached (play a full game)
* Backpropagation: update the counters described below of each ancestor
  of :math:`T`.

Each node holds 3 counters : the number of times a node has been used during a
sampling iteration (:math:`N`), the number of simulations using this node ended
with a win for the player 1 (:math:`W_1`) and player 2 (:math:`W_2`). From this
counters, a probability of winning if an action is chosen can be computed
immediately: :math:`\frac{W_1}{N}` or :math:`\frac{W_2}{N}`.

This sampling can be ran as many times as needed or allowed\footnote{Most of the
time, the agent is time constrained}, each time, refining the probability of
winning when choosing a child of the root node. When we are done sampling the
agent chooses the child with the highest probability of winning and plays the
corresponding action in the game.

Node Selection
--------------

In step 1 and 3 of the algorithm, we have to choose nodes.
There are multiples ways to choose those.

The most naïve method, in the vanilla MCTS we take a child at random each time.
This is easy to implement and has no bias but it is not effective as it explores
every part of the tree even if a part has no chance of leading to a win for the
player.


Upper Confidence Bounds for Trees
---------------------------------

A better method would be asymmetric and only explore interesting parts of the
tree. Kocsis and Szepervari :cite:`kocsis2006bandit` defined Upper Confidence
Bounds for Trees (UCT), a method mixing vanilla MCTS and Upper Confidence Bounds
(UCB).

Indeed, in step 1, selecting the node during the tree descent that maximizes the
probability of winning is analogous to the multi-armed bandit problem in which a
player has choose the slot machine that maximizes the estimated reward.

The UCB formula is the following, where :math:`N'` is the number of times the
parent of the node has been visited and :math:`c` a fixed parameter:

.. math::

    \frac{W_1}{N} + c \times \sqrt{\frac{ln N'}{N}}

:math:`c` can be tuned to balance exploitation of known wins and exploration of
less visited nodes. Kocsis et al. has shown that :math:`\frac{\sqrt{2}}{2}`
:cite:`kocsis2006bandit` is a good value when rewards are in :math:`[0, 1]`.

In step 3, the playouts are played at random as it is the first time these nodes
are seen and we do not have a generic evaluation function do direct the playout
towards "better" states.


Informed UCT
------------


Surprisingly,
increasing the bias in the random play-outs can
occasionally weaken the strength of a program using the
UCT algorithm even when the bias is correlated with Go
playing strength. One instance of this was reported by Gelly
and Silver [#GS07]_, and our group observed a drop in strength
when the random play-outs were encouraged to form patterns
commonly occurring in computer Go games [#Fly08]_.

.. [#GS07] Sylvain Gelly and David Silver. Combining online and offline
 knowledge in uct. In ICML ’07: Proceedings of the 24th
 Internatinoal Conference on Machine Learning, pages 273–280.
 ACM, 2007.

.. [#Fly08] Jennifer Flynn. Independent study quarterly reports.
 http://users.soe.ucsc.edu/~charlie/projects/SlugGo/, 2008

Alpha Zero
----------

To replace the random play in step 3, D. Silver et al. propose
:cite:`AlphaGoZero` to use a neural network to estimate the value of a
game state without having to play it. This can greatly enhances the performance
of the algorithm as much less playouts are required.


Bibliography
------------

.. bibliography:: refs.bib
   :style: custom



Although captured stones
contribute to a position’s final outcome, the best
move from a position does not depend on them.
We therefore consider the distribution of only
uncaptured stones [romein2003] -> faux : need proof
