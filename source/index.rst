  
========================================
Playing Mancala with MCTS and Alpha Zero
========================================

.. contents:: Table of Contents
   :depth: 2

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




  
Awalé
-----

The subject of our study, Awalé is played on a board made of two rows of six
pits. Each row is owned by a player that sits in front of it.
In the initial state of the game every pit contains 4 seeds thus the game contains
48 seeds in total.

.. figure:: /_static/awale.jpg

   A typical Awalé board in the start position.

The board can be schematized like this, every big circle representing a pit and every small disc representing a seed. The 6 pits from the top row belonging to the north player and the 6 others belonging to the south player.

Numbers in the bottom right of each pits are the count of stones in each pit for better readability.







  









.. raw:: html
    :file: index_files/index_4_0.svg








  
Rules of the game
-----------------

The goal for both players is to capture more seeds than its opponent. As the
game has 48 seeds, capturing 25 is enough to win and end the game.

Each player plays alternatively, without the right to pass their turn. A
player's turn consists in choosing one of his non-empty pits, pick all seeds
contained in the pit and seed them one by one in every consecutive pits on the right
(rotating counter-clockwise). The player thus has at most 6 possible moves at
each turn.

Usually, the player that starts the game is the oldest player. Here, we will start at random.

As an example, if we start from the initial state showed above, the first player to move is the south (on the bottom) and he chooses the 4th pit, the board will then have the following state.




  









.. raw:: html
    :file: index_files/index_6_0.svg








  
When the last sowed seed is placed in a pit owned by the opponent and after seeding
the pit contains two or three seeds, the content of the pit is captured by
the player and removed from the game. If the pit preceding the captured pit also
contains two or three seeds, it is also captured. The capture continues until a
pit without two or three seeds is encountered. When the capture is ended the
next player's turn starts.

Otherwise, when the last sowed seed is placed in a pit that now contains one seed, more
than 3 seeds or in the current player's own pits, the turn of the player is ended without
any capture.

For example, if the south player plays the 4th pit in the following configuration he will
be able to capture the opponent's 4th and 5th pits (highlighted in red in the second figure) 




  









.. raw:: html
    :file: index_files/index_8_0.svg








  









.. raw:: html
    :file: index_files/index_9_0.svg








  
If the pit chosen by the player contains more than 12 seeds, the sowing makes
more than a full revolution and the starting hole is skipped during the second
and subsequent passes.

If the current player's opponent has no seed left in his half of the board, the
current player has to play a move that gives him seeds if such a move exists.
This rule is called the "let the opponent play" or "don't starve your opponent".

This rule has for second consequence that if a player plays a move that could capture
every seed of the opponent, he may play this move but he may not capture the seeds as
it would also prevent the opponent of playing.

In the following example, the south player has to play the fifth pit because playing the first would leave the opponent without any move to play.




  









.. raw:: html
    :file: index_files/index_11_0.svg








  
When a player has captured more than 25 seeds the game ends and he wins. If both
players have captured 24 seeds, the game ends by a draw. If the current player
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



.. topic:: How should you read this document ?

    This document is a mix of text and Python code in the form of notebook
    cells. Reading only the text and skipping all the code should be enough for
    you to understand the whole work. But if you are interested in the
    implementation work and the details of the simulations you are welcome to
    read the notebook cells as well.

    Some output and cells are hidden for the sake of brevity and readability.
    Click on the button to reveal the full code and output that were used for
    the simulations to write this work.







  
Implementation of the rules
---------------------------

Some rules have deliberately been excuded from this implementation :

-  Loops in the game state are not checked (this speeds up considerably
   the computations and we never encountered a loop in practice)
-  You are authorized to starve your opponent. This was made so the
   rules are a little bit simpler and should not change the complexity
   of the game.

We first define a dataclass with the minmal attributes needed to store the game state




  


  .. code:: ipython3

    from dataclasses import dataclass
    from typing import Tuple
    
    @dataclass
    class Game:
        pits: np.array
        current_player: int
        captures: Tuple[int, int]






  
We add a static method to start a new game




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        @classmethod
        def new(klass):
            return klass(
                # A 6x2 matrix filled with 4
                pits=np.ones(6 * 2, dtype=int) * 4,
                current_player=0,
                captures=(0, 0)
            )






  
Next, we add some convenience methods that will be usefull later




  


  .. code:: ipython3

    class Game(Game):
        ...
    
        @property
        def view_from_current_player(self):
            if self.current_player == 0:
                return self.pits
            else:
                return np.roll(self.pits, 6)
        
        @property
        def current_player_pits(self):
            if self.current_player == 0:
                return self.pits[:6]
            else:
                return self.pits[6:]
    
        @property
        def current_opponent(self):
            return (self.current_player + 1) % 2
        
        @property
        def adverse_pits_idx(self):
            if self.current_player == 1:
                return list(range(6))
            else:
                return list(range(6, 6 * 2))
        






  
Now we start implementing the rules




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        @property
        def legal_actions(self):
            our_pits = self.current_player_pits
            return [x for x in range(6) if our_pits[x] != 0]
        
        @property
        def game_finished(self):
            no_moves_left = np.sum(self.current_player_pits) == 0
            
            half_seeds = 6 * 4
            enough_captures = self.captures[0] > half_seeds or self.captures[1] > half_seeds
            
            draw = self.captures[0] == half_seeds and self.captures[1] == half_seeds
            
            return no_moves_left or enough_captures or draw
        
        @property
        def winner(self):
            if not self.game_finished:
                return None
            return np.argmax(self.captures)






  
We can now add the ``step()`` functions that plays a turn

The main method you are interested in is ``Game.step(i)`` to play the
i-th pit in the current sate. This will return the new state, the amount
of seeds captured and a boolean informing you if the game is finished.




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        def step(self, action):
            assert 0 <= action < 6, "Illegal action"
            
            target_pit = action if self.current_player == 0 else action - 6
            
            seeds = self.pits[target_pit]
            assert seeds != 0, "Illegal action: pit % is empty" % target_pit
            
            # copy attributes
            pits = np.copy(self.pits)
            captures = np.copy(self.captures)
            
            # empty the target pit
            pits[target_pit] = 0
            
            # fill the next pits
            pit_to_sow = target_pit
            while seeds > 0:
                pit_to_sow = (pit_to_sow + 1) % (6 * 2)
                if pit_to_sow != target_pit: # do not fill the target pit ever
                    pits[pit_to_sow] += 1
                    seeds -= 1
            
            # Capture
            # -------
            
            # count the captures of the play
            round_captures = 0
            if pit_to_sow in self.adverse_pits_idx:
                # if the last seed was in a adverse pit
                # we can try to collect seeds
                while pits[pit_to_sow] in (2, 3):
                    # if the pit contains 2 or 3 seeds, we capture them
                    captures[self.current_player] += pits[pit_to_sow]
                    round_captures += pits[pit_to_sow]
                    pits[pit_to_sow] = 0
                    
                    # go backwards
                    pit_to_sow = (pit_to_sow - 1) % (self.n_pits * 2)
            
            # change player
            current_player = (self.current_player + 1) % 2
            
            new_game = type(self)(
                pits,
                current_player,
                captures
            )
    
            return new_game, round_captures, new_game.game_finished







  
And some display functions




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        def show_state(self):
            if self.game_finished:
                print("Game finished")
            print("Current player: {} - Score: {}/{}\n{}".format(
                self.current_player,
                self.captures[self.current_player],
                self.captures[(self.current_player + 1) % 2],
                "-" * self.n_pits * 3
            ))
            
            pits = []
            for seeds in self.view_from_current_player:
                pits.append("{:3}".format(seeds))
            
            print("".join(reversed(pits[self.n_pits:])))
            print("".join(pits[:self.n_pits]))
    
        def __repr__(self):
            return "<Game current_player:{player} captures:{captures[0]}/{captures[1]}>".format(
                player=self.current_player,
                captures=self.captures
            )
        
        def _repr_svg_(self):
            board = np.array([
                list(reversed(self.pits[6:])),
                self.pits[:6]
            ])
            return board_to_svg(board, True)






  
Play a game




  


  .. code:: ipython3

    g = Game.new()
    g, captures, done = g.step(4)
    g








.. raw:: html
    :file: index_files/index_31_0.svg








  
As the rest of this work is always using trees as the base model for a game,
we also use it here in the implementation.




  


  .. code:: ipython3

    from typing import Optional, List
    from dataclasses import field
    
    @dataclass
    class TreeGame(Game):
        parent: Optional[Game] = None
        children: List[Optional[Game]] = field(default_factory=lambda: [None] * 6)






  


  .. code:: ipython3

    class TreeGame(TreeGame):
        ...
        
        def step(self, action):
            # If we already did compute the children node, juste return it
            if self.children[action] is not None:
                new_game = self.children[action]
                captures = new_game.captures[self.current_player] - self.captures[self.current_player]
                return new_game, captures, new_game.game_finished
            else:
                new_game, captures, finished = super().step(action)
                new_game.parent = self
                return new_game, captures, finished






  

-  ``is_fully_expanded`` tells you if all actions of this state have
   been computed
-  …




  


  .. code:: ipython3

    class TreeGame(TreeGame):
        ...
    
        @property
        def successors(self):
            children = [x for x in self.children if x is not None]
            successors = children + list(itertools.chain(*[x.successors for x in children]))
            return successors
        
        @property
        def unvisited_actions(self):
            return [i for i, x in enumerate(self.children) if x is None]
    
        @property
        def legal_unvisited_actions(self):
            return list(set(self.unvisited_actions).intersection(set(self.legal_actions)))
        
        @property
        def expanded_children(self):
            return [x for x in self.children if x is not None]
        
        @property
        def is_fully_expanded(self):
            legal_actions = set(self.legal_actions)
            unvisited_actions = set(self.unvisited_actions)
            return len(legal_actions.intersection(unvisited_actions)) == 0
        
        @property
        def is_leaf_game(self):
            return self.children == [None] * 6
        
        @property
        def depth(self):
            if self.parent is None:
                return 0
            return 1 + self.parent.depth






  


  .. code:: ipython3

    class TreeGame(TreeGame):
        ...
        
        def update_stats(self, winner):
            assert winner in [0, 1]
            self.wins[winner] += 1
            self.n_playouts += 1
            if self.parent:
                self.parent.update_stats(winner)






  
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

Citation:

> Surprisingly,
> increasing the bias in the random play-outs can
> occasionally weaken the strength of a program using the
> UCT algorithm even when the bias is correlated with Go
> playing strength. One instance of this was reported by Gelly
> and Silver [#GS07]_, and our group observed a drop in strength
> when the random play-outs were encouraged to form patterns
> commonly occurring in computer Go games [#Fly08]_.




  
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




  
Footnotes
---------

.. [#source_kalah] Picture by Adam Cohn under Creative Commonds license https://www.flickr.com/photos/adamcohn/3076571304/

.. [#Fly08] Jennifer Flynn. Independent study quarterly reports.
 http://users.soe.ucsc.edu/~charlie/projects/SlugGo/, 2008
 
.. [#GS07] Sylvain Gelly and David Silver. Combining online and offline
 knowledge in uct. In ICML ’07: Proceedings of the 24th
 Internatinoal Conference on Machine Learning, pages 273–280.
 ACM, 2007.




  


..
.. Although captured stones
.. contribute to a position’s final outcome, the best
.. move from a position does not depend on them.
.. We therefore consider the distribution of only
.. uncaptured stones [romein2003] -> false : need proof



