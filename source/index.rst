  
.. raw:: html

    <section class="first-page">
        <h1>Playing Awale with MCTS</h1>
        <h2>Master thesis submitted in partial fulfilment of the requirements
        for the degree of Master of Science in Applied Sciences and Engineering:&nbsp;Computer Science
        </h2>

        2020-2021
    </section>




  
.. contents:: Table of Contents
   :depth: 3

.. sectnum::




  
============
Introduction
============

Awale is a popular board game played mainly in Africa. The board has two rows of six pits, each containing four seeds in the initial state.

At each turn, the players move some seeds and can potentially capture some of them, according to deterministic rules. The goal of the game is to capture more seeds than one's opponent.

.. _board:

.. figure:: /_static/initial.jpg

   A typical Awalé board in the initial state with both players on their side of the board.
   

.. todo:: Explain here what i'm going to do in my thesis, why it is interesting and why it is new.

In Section 2, we present Awale in detail. We then introduce Game Theory frameworks in Section 3.
Section 4 reviews various approaches to solve Awale: retrograde analysis, :math:`\alpha-\beta`-pruning Minimax, and basic Monte Carlo Tree Search.
In Section 5, we describe more advanced versions of MCTS and in particular UCT.
Section 6 presents some empirical results (simulations) allowing to compare several MCTS algorithms and Section 7 concludes.




  
=====
Awale
=====

The subject of our study, Awale is an ancient, two player board game originating from Ghana.
This game is also sometimes called Awele, Oware, Owari or Ayo in the neighboring countries, languages and cultures :cite:`crane1982`.

Originally, the game is played on the ground, by digging two rows of six small pits, each containing
stones, seeds or shells. In the present document, we will name them seeds. The game is also often played on a wooden board symbolizing the original dirt pits.
The board can be schematized as in :numref:`Figure %s <fig:initial_board>`, every big circle representing a pit and every small disc representing a seed.
Numbers at the bottom right of each pit are the counts of seeds in each pit for better readability.
Each row of pits is owned by a player that sits in front of it (:numref:`see Fig. %s <board>`).
For the sake of convenience, the players are named North and South.
The 6 pits from the top row belong to North and the 6 from the bottom to South.

The players take turns, a player removing all the seeds from a pit and placing them in other pits following the rules. This is called sowing the seeds. This can result in a configuration in which the player is allowed to capture some seeds according to the rules.
The goal for each player is to capture more seeds than his opponent.
The rules vary slightly across countries and will be detailed in :ref:`sec:rules`. 







  











    

    
.. _fig:initial_board:
    


.. figure:: index_files/index_5_0.svg








  
  A schematized view of the initial state of the board.




  
Mancala
-------

The Mancala games are an ancient family of game that are played on many continents :cite:`deVoogt2008`, Awale being one of them.
The word mancala comes from the Arabic word "نقلة", transliterated as "naqala" and literally meaning "to move".

Like Awale, Mancala games can consist of rows of pits, some of them having more than two rows (:numref:`see Fig. %s <bao>`) and sometimes extra pits with a special role. Mancala games can sometimes be played by more than two players.
 
.. _bao:

.. figure:: _static/bao.jpg

  A wooden Bao game [#source_bao]_

There are too many variants of the Mancala games to list them all here, but a
few notable ones are Awale, Wari, Bao, Congkak and Kalah.

Mancala games in general, while less known than Chess or Go, are quite popular and
are played in tournaments around the world, both in offline and online competitions :cite:`owaresociety,fandom_tournaments`.



In particular, Kalah is a commercial, modern variant of Mancala, introduced in the 1950s by William Julius Champion Jr., that is widespread in the United States. :cite:`irving2000solving`. This variant has been studied in Artifical Intelligence as early as 1964 by :cite:`russel1964`.
Nowadays, Kalah is often used as an example game in computer-science courses.
Other Mancala games have been studied in Computer Science and Artificial Intelligence :cite:`deVoogt2008`. Tournaments opposing computers on both sides have been organised multiple times, notably in the Computer Olympiad organized by the International Computer Games Association :cite:`icga_olympiad`.





  
.. _sec:rules:

Rules of the game
-----------------

The basic rules of Awale are the same everywhere but there are some minor differences around the globe and in the literature.
The rules presented here and implemented later in this thesis are inspired from :cite:`goot2001` and adapted by us.

The goal for each player is to capture more seeds than his opponent. Because the
game has 48 seeds, capturing 25 is enough for a player to win and ends the game.

Each player plays alternatively, without the right to pass his turn. A
player's turn consists in choosing one of his non-empty pits, picking all seeds
contained in the pit and sowing them one by one in every consecutive pits on the right
(rotating counter-clockwise). The player thus has at most 6 possible moves at
each turn (one per non-empty pit owned by him).

Usually, the player that starts the game is the oldest player. In this work, South will always play first.

In this work, the pits of a player are numbered left to right from his point of view as shown in :numref:`Figure %s <fig:pit_numbering>`. :math:`1` being the leftmost pit of South, until :math:`6` at the far right. The same holds for North: :math:`1'` to :math:`6'`.




  











    

    
.. _fig:pit_numbering:
    


.. figure:: index_files/index_9_0.svg








  
  Pit numbering convention: the pits of a player are numbered left to right from his point of view.




  

As an example, in the initial state (:numref:`See Fig. %s <fig:initial_board>`), the first player to move is South (on the bottom) and he plays :math:`4` (highlighted in the figure in red), the board will then be in the  state shown in :numref:`Figure %s <fig:first_move>`.




  











    

    
.. _fig:first_move:
    


.. figure:: index_files/index_12_0.svg








  
  The board after the forst move, where South played pit 4.




  
When the last sowed seed is placed in a pit owned by the opponent and, after sowing,
the pit contains two or three seeds, the content of the pit is captured by
the player and removed from the game. If the pit preceding the captured pit also
contains two or three seeds, it is also captured. The capture continues until a
pit without two or three seeds is encountered. When the capture is ended the
next player's turn starts.

Otherwise, when the last sowed seed is placed in a pit that, after sowing, contains one seed, more
than 3 seeds or in the current player's own pits, the turn of the player is ended without
any capture.
For example, if South plays :math:`4` in the configuration shown in :numref:`Figure %s <fig:pre_capture>`, he will
be able to capture the seeds in pits 2' and 3' (highlighted in red in :numref:`Figure %s <fig:post_capture>`).




  











    

    
.. _fig:pre_capture:
    


.. figure:: index_files/index_15_0.svg








  
  An example of a board configuration where South is to play pit 4.




  











    

    
.. _fig:post_capture:
    


.. figure:: index_files/index_17_0.svg








  
  The resulting board after South played 4 in :numref:`Fig %s <fig:pre_capture>`. Pits 2' and 3' will be captured.




  
If the pit chosen by the player contains more than 12 seeds, the sowing makes
more than a full revolution of the board and the starting hole is skipped during the second
and subsequent passes.

If the current player's opponent has no seed left in his half board, the
current player has to play a move that gives him seeds if such a move exists.
This rule is called the "feed your opponent".
In :numref:`Figure %s <fig:feed>`, South has to play pit 5 because playing pit 1 would leave the opponent without any move to play.




  











    

    
.. _fig:feed:
    


.. figure:: index_files/index_20_0.svg








  
  South is forced to play pit 5 because playing pit 1 would leave North without any seed to play.




  
When a player has captured more than 25 seeds the game ends and he wins. If both
players have captured 24 seeds, the game ends by a draw. If the current player's
pits are all empty, the game ends and the player with the most captures wins.

The last way to stop the game is when a position is encountered twice in the
same game (there is a cycle): the game ends and the player with most captures
wins.




  
Implementation of the rules
---------------------------

You might be reading this document in the form of a web page or a pdf file but its original form is a Jupyter Notebook :cite:`jupyter`. Jupyter Notebooks are documents mixing computer code (in this case Python code), the result of the execution of the code and text. These can be used to document experiments in the same place they are run.

Sections containing code are prefixed by :code:`In[]:` and the output of the code is showed immediately under it, prefixed by :code:`Out[]:`. An example is shown below.




  


  .. code:: ipython3

    # This is Python code
    print("This was executed by Python")




.. parsed-literal::

    This was executed by Python





  
In this subsection, we use the use th power of Jupyter Notebooks to define in multiple steps a Python :code:`Game()` class holding the state of the game and its rules. We will then successively inherit from it to add the rules and some convenience methods.

We set the following encoding conventions:
 - :code:`0` is South, :code:`1` is North,
 - player's actions are numbered from :code:`0` to :code:`5`, :code:`0` being the leftmost pit in front of him, :code:`5` being the rightmost.

First, we define a dataclass with the minimal attributes needed to store a state of the game.







  


  .. code:: ipython3

    from dataclasses import dataclass, field
    
    
    @dataclass
    class Game:
        # a 2x6 matrix containing the number of seeds in each pits
        pits: np.array = field(default_factory=lambda: np.ones(6 * 2, dtype=int) * 4)
        # 0 for South and 1 for North
        current_player: int = 0
        # the number of seeds captured by each player
        captures: np.array = field(default_factory=lambda: np.zeros(2, dtype=int))






  
Next, we add some convenient methods that will be useful later.




  


  .. code:: ipython3

    class Game(Game):
        ...
    
        @property
        def view_from_current_player(self) -> List[int]:
            """Returns the board as seen by a player"""
            if self.current_player == 0:
                return self.pits
            else:
                return np.roll(self.pits, 6)
    
        @property
        def current_player_pits(self) -> List[int]:
            """Returns a 6-vector containing the pits owned by the current player"""
            if self.current_player == 0:
                return self.pits[:6]
            else:
                return self.pits[6:]
    
        @property
        def current_opponent(self) -> int:
            return (self.current_player + 1) % 2
    
        @property
        def adverse_pits_idx(self) -> List[int]:
            """Returns the indices in the `self.pits` array owned by the opposing player"""
            if self.current_player == 1:
                return list(range(6))
            else:
                return list(range(6, 6 * 2))






  
Now that the base is set, we start implementing the rules,
some of them being deliberately excluded from this implementation:

-  Loops in the game state are not checked (this considerably speeds up the computations and we did not encounter a loop in our preliminary work);
-  The "feed your opponent" rule is removed; This makes the
   rules simpler and we expect it does not tremendously change the complexity of the game.

.. todo We did later encounter loops after running way more simulations. But this only happened yet using basic algorithms (greedy vs greedy for example). For now, we simulate 500 turns, if we hit this threshold, we declare a tie. This should be detailed in the experimental setup




  


  .. code:: ipython3

    class Game(Game):
        ...
    
        @property
        def legal_actions(self) -> List[int]:
            """Returns a list of indices for each legal action for the current player"""
            our_pits = self.current_player_pits
            # Return every pit of the player that contains some seeds
            return [x for x in range(6) if our_pits[x] != 0]
    
        @property
        def game_finished(self) -> bool:
            # Does the current player has an available move ?
            no_moves_left = np.sum(self.current_player_pits) == 0
    
            # Has one player captured more than half the total seeds ?
            HALF_SEEDS = 24  # (there are 2*6*4=48 seeds in total)
            enough_captures = self.captures[0] > HALF_SEEDS or self.captures[1] > HALF_SEEDS
    
            # Is it a draw ? Does both player have 24 seeds ?
            draw = self.captures[0] == HALF_SEEDS and self.captures[1] == HALF_SEEDS
    
            # If one of the above three are True, the game is finished
            return no_moves_left or enough_captures or draw
    
        @property
        def winner(self) -> Optional[int]:
            """Returns the winner of the game or None if the game is not finished or in a draw"""
            if not self.game_finished:
                return None
            # The game is finished but both player have the same amount of seeds: it's a draw
            elif self.captures[0] == self.captures[1]:
                return None
            # Else, there is a winner: the player with the most seeds
            else:
                return 0 if self.captures[0] > self.captures[1] else 1






  
We can now define the :code:`Game.step(i)` method that is called for every step of the game.
It takes a single parameter, :code:`i`, and plays the i-th pit in the current sate.
This method returns the new state, the amount of seeds captured and a boolean informing whether the game is finished.




  


  .. code:: ipython3

    class Game(Game):
        ...
    
        def step(self, action: int) -> Tuple[Game, int, bool]:
            """Plays the action given as parameter and returns:
                - a the new state as a new Game object,
                - the amount of captured stones in the transition
                - a bool indicating if the new state is the end of the game
            """
            assert 0 <= action < 6, "Illegal action"
    
            # Translate the action index to a pit index
            target_pit = action if self.current_player == 0 else action - 6
    
            seeds = self.pits[target_pit]
            assert seeds != 0, "Illegal action: pit % is empty" % target_pit
    
            # Copy the attributes of `Game` so that the original
            # stays immutable
            pits = np.copy(self.pits)
            captures = np.copy(self.captures)
    
            # Empty the pit targeted by the player
            pits[target_pit] = 0
    
            # Fill the next pits while there are still seeds
            pit_to_sow = target_pit
            while seeds > 0:
                pit_to_sow = (pit_to_sow + 1) % (6 * 2)
                # Do not fill the target pit ever
                if pit_to_sow != target_pit:
                    pits[pit_to_sow] += 1
                    seeds -= 1
    
            # Count the captures of the play
            round_captures = 0
    
            # If the last seed was in a adverse pit we can try to collect seeds
            if pit_to_sow in self.adverse_pits_idx:
                # If the pit contains 2 or 3 seeds, we capture them
                while pits[pit_to_sow] in (2, 3):
                    captures[self.current_player] += pits[pit_to_sow]
                    round_captures += pits[pit_to_sow]
                    pits[pit_to_sow] = 0
    
                    # Select backwards the next pit to check
                    pit_to_sow = (pit_to_sow - 1) % 12
    
            # Change the current player
            current_player = (self.current_player + 1) % 2
    
            # Create the new `Game` instance
            new_game = type(self)(pits, current_player, captures)
    
            return new_game, round_captures, new_game.game_finished






  
As the game rules are now implemented, we can add some methods to display the current state of the board to the user, either in textual mode or as an SVG file that can be displayed inline in this document.




  


  .. code:: ipython3

    class Game(Game):
        ...
    
        def show_state(self):
            """Print a textual representation of the game to the standard output"""
            if self.game_finished:
                print("Game finished")
            print(
                "Current player: {} - Score: {}/{}\n{}".format(
                    self.current_player,
                    self.captures[self.current_player],
                    self.captures[(self.current_player + 1) % 2],
                    "-" * 6 * 3,
                )
            )
    
            pits = []
            for seeds in self.view_from_current_player:
                pits.append("{:3}".format(seeds))
    
            print("".join(reversed(pits[6:])))
            print("".join(pits[:6]))
    
        def _repr_svg_(self):
            """Return a SVG file representing the current state to be displayed in a notebook"""
            board = np.array([list(reversed(self.pits[6:])), self.pits[:6]])
            return board_to_svg(board, True) 






  
To show a minimal example of the implementation, we can now play a move and have its results displayed here.




  


  .. code:: ipython3

    g = Game()  # Create a new game
    g, captures, done = g.step(4)  # play the 5th pit (our implementation starts at 0)
    g  # Display the resulting board inline











.. figure:: index_files/index_37_0.svg








  
=====================
Awale and Game Theory
=====================

.. warning::
  Previouosly, this section contained text about perfect information games, strongly solved games, then represented perfect information games as finite state machines and trees. After reading more litterature, i decided to remove an rewrite it.
  I plan to rewrite it with the following: Set the basics of Game Theory and the concept of a "solution" to a game, talk about the minimax, define perfect information/combinatorial games then their tree representation.





  
Tree representation
~~~~~~~~~~~~~~~~~~~

A combinatorial game like Awale can be represented as a tree in a straightforward way where every node is a state of the game.
The root of the tree represents the initial state.
If in a state :math:`s` the current player plays action :math:`i` resulting in state :math:`s'` then :math:`s'` will be the i-th child of the node representing :math:`s`.

This results in the following properties:
    - As the current player at the root node is South and that players alternate after each turn,
      the tree contains alternating layers of current players:
      the current player for nodes with an even depth is South and for odd depths is North;
    - The leaf nodes of the tree correspond to final states;
    - The path from the root to a leaf thus represents an instance of a full game.

.. todo:: Insert a figure of an tree here


We can now define the branching factor: the maximum number of children of a node in the game.
In Awale the player can choose to sow his seeds from one of his non-empty pits.
As the player has 6 pits, the branching factor is 6.

We now implement this tree representation in Python by inheriting from :code:`Game()` class previously defined so that a state can hold references to its parent and children.




  


  .. code:: ipython3

    from __future__ import annotations
    from typing import Optional, List
    from weakref import ref, ReferenceType
    
    
    @dataclass
    class TreeGame(Game):
        # Hold an optional reference to the parent state
        parent: Optional[ReferenceType[Game]] = None
        # Hold a list of 6 optional references to the children
        children: List[Optional[Game]] = field(default_factory=lambda: [None] * 6)
        depth: int = 0






  
Next, we overload the ``Game.step(i)`` method so that we do not compute twice state if it was already in the tree. If a new node was generated, we keep a reference to the parent when we create a new child.




  


  .. code:: ipython3

    class TreeGame(TreeGame):
        ...
    
        def step(self, action):
            # If we already did compute the children node, just return it
            if self.children[action] is not None:
                new_game = self.children[action]
                captures = (
                    new_game.captures[self.current_player]
                    - self.captures[self.current_player]
                )
                return new_game, captures, new_game.game_finished
            # If not, call the original `step()` method and keep references in both directions
            else:
                new_game, captures, finished = super().step(action)
                new_game.parent = ref(self)
                new_game.depth = self.depth + 1
                self.children[action] = new_game
                return new_game, captures, finished






  


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






  
================================================
Artificial Intelligence approaches to play Awale
================================================

Many algorithms have been proposed and studied to play sequential perfect information games.
A few examples detailed here are retrograde analysis, Minimax, :math:`\alpha-\beta` pruning,
Monte Carlo tree search (MCTS) and the new approach from Deepmind: Alpha Zero :cite:`AlphaGoZero`.

We will quickly present those and then focus on MCTS and its variants as they are computationally feasible and do not require expert knowledge about the given game to make reasonable decisions.






  
First we implement a player class. A player keeps track of the game state internaly.
At each turn of the game, a player is called with the method `play()` to get the action played by the opponent
(and thus update it's internal state) and then chooses an action with `get_action()`,
updates once more it's internal state and then outputs it's action for the other player.




  


  .. code:: ipython3

    class Player:
        def play(self, their_action):
            # If we are the first player, there is no previous action
            if their_action != -1:
                # Play the opponent's move
                self.root, _, _ = self.root.step(their_action)
            else:
                assert self.player_id == 0, "Only the first player can have their_action=-1"
            
            action = self.get_action()
            self.root, _, _ = self.root.step(action)
            
            return action






  
Alpha-Beta pruning Minimax
--------------------------

.. todo:: Describe the algorithm and implement an agent for Awale



Retrograde analysis
-------------------


For both divergent and convergent games search algorithms can prove the game result for positions near
the end of a game. However, for divergent games the number of endgame
positions is so big that enumerating all of them is computationally impossible (except for trivial
games like Tic-Tac-Toe). However, for convergent games, the number of positions
near the end of the game is small. Usually small enough to traverse them all, and collect
their game values in a database, a so called endgame database.

Retrograde Analysis computes endgame databases by going backward from values of final
positions towards the initial position :cite:`goot2001`.
First, Retrograde Analysis identifies all final positions in which the game value is known.
By making reverse moves from these final positions the game value of some non-final positions can be deduced. And by making reverse moves from these newly proven non-final positions, the game value of other non-final positions can be deduced. This can continue either by running of available memory or by having enumerated all the legal positions in the game.

Ströhlein is the first researcher who came up with the idea to create endgame databases and applied his idea to chess :cite:`endgame1970`.
The first endgame database for Awale has been created by :cite:`allis1995` and was followed by many others, while the quest was ended by :cite:`romein2003solving` publishing a database for all legal positions.


The above-mentioned results for Kalah and Awale both use an almost brute-force
method to solve the game and use a database of all possible states. The database
used by :cite:`romein2003solving` has 204 billion entries and weighs 178GiB.
Such a huge database is of course not practical and  we thus think  there is still room for
improvement if we can create an agent with a policy that does not need a
exhaustive database, even if the agent is not capable of a perfect play.


Monte Carlo Tree Search
-----------------------

.. todo:: This section and the next should be more detailed

In this subsection, we define Markov Decision Processes (MDP) and model Awale with this framework. We then describe and detail Monte Carlo Tree Search, a policy-optimization algorithm for finite-horizon, finite-size MDPs.


As Awale can be represented as an MDP, we could be tempted to use the usual framework of Q-Learning [Cite XXX] to find the best policy to maximise our reward. But since the state space is huge, this is computationally difficult or even impossible in memory and time constrained cases.
To overcome this computational problem, the MCTS method constructs only a part of game the tree by sampling and tries to estimate the chance of winning based on this information.

Algorithm
~~~~~~~~~

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
* Back-propagation: update the counters described below of each ancestor
  of :math:`T`.


Each node holds 3 counters : (:math:`W_S`), the number of simulations using this node ended that
with a win for South;  and North (:math:`W_N`). From this
counters, a probability of North winning conditional on a given action can be computed
immediately: :math:`\frac{W_N}{N}`.

This sampling can be ran as many times as allowed (most of the
time, the agent is time constrained). One can also stop the sampling earlier if

each time refining the probability of
winning when choosing a child of the root node. When we are done sampling, the
agent chooses the child with the highest probability of winning and plays the
corresponding action in the game.

the total number of times a node has been played during a
sampling iteration (:math:`N`)




  
Implementation
~~~~~~~~~~~~~~




  


  .. code:: ipython3

    @dataclass
    class TreeStatsGame(TreeGame):
        wins: np.array = field(default_factory=lambda: np.zeros(2, dtype=int))
        n_playouts: int = 0
    
        def update_stats(self, winner):
            if winner in [0, 1]:
                self.wins[winner] += 1
            self.n_playouts += 1
            if self.parent and self.parent():
                self.parent().update_stats(winner)






  
The MCTS first chooses a node to expand with the `tree_policy()` when the node is found, it is expanded with the `default_policy()`. When reaching a terminal node, the counters are updated. This is repeated `BUDGET` times and then the final action is chosen as the action that has the highest amount of wins.

Both policies in this implementation are random walks.




  


  .. code:: ipython3

    class MCTSPlayer(Player):
        def __init__(self, player_id, budget: Union[int, timedelta]):
            self.root = TreeStatsGame()
            self.player_id = player_id
            self.budget = budget
    
        def tree_policy(self, node):
            while not node.is_leaf_game:
                if node.is_fully_expanded:
                    node = random.choice(node.expanded_children)
                else:
                    action = random.choice(node.legal_unvisited_actions)
                    node, _, _ = node.step(action)
            return node
        
        def explore_tree(self):
            # Choose a starting node
            node = self.tree_policy(self.root)
    
            # Run a simulation on that node
            finished = node.game_finished
            while not finished:
                action = self.default_policy(node)
                node, _, finished = node.step(action)
    
            # Backtrack stats
            node.update_stats(node.winner)
        
        def default_policy(self, node):
            # Random walk
            return random.choice(node.legal_actions)
        
        def action_score(self, x):
            node = self.root.children[x]
            if node is None:
                return -random.random()
    
            assert self.root.current_player == self.player_id
            assert node.current_player != self.player_id
    
            return node.wins[self.player_id]
            
        
        def get_action(self):
            if isinstance(self.budget, int):
                for _ in range(self.budget):
                    self.explore_tree()
            elif isinstance(self.budget, timedelta):
                start = datetime.now()
                end = start + self.budget
                while datetime.now() < end:
                    self.explore_tree()
            else:
                raise TypeError("budget should be Union[int, timedelta], not %s" % type(budget))
            
            possible_actions = self.root.legal_actions
            return max(possible_actions, key=self.action_score)






  
Naive agents
------------

To be able to benchmark our agents, we also implement two naive agents.
The first is a random player thatchooses an action at random between all the legal actions




  


  .. code:: ipython3

    class RandomPlayer(Player):
        def __init__(self, player_id):
            self.root = Game()
            self.player_id = player_id
        
        def get_action(self):
            return random.choice(self.root.legal_actions)






  
The second is :math:`\varepsilon`-Greedy: an agent that tries to maximise an immediate reward at each turn: the number of seeds captured during that turn. The :math:`\varepsilon` parameter introduces randomness: at each turn, the agent draws an number between 0 and 1, if it is geater than :math:`\varepsilon`, the agent plays at random.




  


  .. code:: ipython3

    class GreedyPlayer(Player):
        def __init__(self, player_id, eps=0):
            self.root = Game()
            self.player_id = player_id
            self.eps = eps
        
        def get_action(self):
            # Choose a move
            children = []
            
            for legal_action in self.root.legal_actions:
                new_state, captures, finished = self.root.step(legal_action)
                if new_state.winner is None:
                    win = 0
                elif new_state.winner == self.player_id:
                    win = 1
                else:
                    win = -1
                children.append((legal_action, captures, win))
            
            # order wins first, then by captures, then random
            sorted_children = sorted(children, key=lambda a_c_w: (-a_c_w[2], -a_c_w[1], random.random()))
            if random.random() < self.eps:
                action = random.choice(self.root.legal_actions)
            else:
                action = sorted_children[0][0]
                
            return action






  
================================
Monte Carlo tree search variants
================================

Node Selection
--------------

In step 1 and 3 of the algorithm, we have to choose nodes.
There are multiples ways to choose those.

In the original MCTS we take a child at random each time.
This is easy to implement but it is not effective since it explores every part of the tree even if a part has no chance of leading to a win for the player.




  
Upper Confidence Bounds for Trees
---------------------------------

A better method would be asymmetric and explore more often the interesting parts of the
tree. Kocsis and Szepervari :cite:`kocsis2006bandit` defined Upper Confidence
Bounds for Trees (UCT), a method mixing vanilla MCTS and Upper Confidence Bounds
(UCB).

Indeed, in step 1, selecting the node during the tree descent that maximizes the
probability of winning is analogous to the multi-armed bandit problem in which a
player has to choose the slot machine that maximizes the estimated reward.

The UCB is

.. math::

    \frac{W_1}{N} + c \times \sqrt{\frac{ln N'}{N}},

where :math:`N'` is the number of times the
parent node has been visited and :math:`c` is a parameter that can be tuned to balance exploitation of known wins and exploration of
less visited nodes. Kocsis et al. has shown that :math:`\frac{\sqrt{2}}{2}`
:cite:`kocsis2006bandit` is a good value when rewards are in :math:`[0, 1]`.

In step 3, the playouts are played by choosing an action from an uniform distribution since it is the first time these nodes
are seen and we do not have a generic evaluation function do direct the playout
towards 'better' states.




  

`UCTPlayer` reuses the MCTS agent but subclasses the `tree_policy` and uses UCT




  


  .. code:: ipython3

    class UCTPlayer(MCTSPlayer):
        def __init__(self, player_id, budget: Union[int, timedelta], c: float):
            super().__init__(player_id, budget)
            self.c = c
            
        def node_score(self, node):
            exporation = node.wins[node.current_opponent] / (node.n_playouts + 1)
            exploitation = math.sqrt(math.log(node.parent().n_playouts) / (node.n_playouts + 1))
            return exporation + self.c * exploitation
    
        def tree_policy(self, node):
            while not node.is_leaf_game:
                if node.is_fully_expanded:
                    node = max_rand(node.expanded_children, key=self.node_score)
                else:
                    action = random.choice(node.legal_unvisited_actions)
                    node, _, _ = node.step(action)
            return node






  
Informed UCT
------------

 `GreedyUCTPlayer` subclasses `UCTPlayer` and changes the `default_policy` to weigh more the actions that will give more immediate rewards.





  


  .. code:: ipython3

    class GreedyUCTPlayer(UCTPlayer):    
        def default_policy(self, node):
            # Greedy walk
            assert len(node.legal_actions) != 0
            captures = [node.step(action)[1] + 1 for action in node.legal_actions]
            return random.choices(node.legal_actions, weights=captures)[0]






  
Alpha Zero
----------

To replace the random play in step 3, D. Silver et al. propose
:cite:`AlphaGoZero` to use a neural network to estimate the value of a
game state without having to play it. This can greatly enhances the performance
of the algorithm as much less playouts are required.




  
=================
Empirical results
=================


This section first describes the experimental setup in wich the games between agents are played as well as the method used to run the experiments in a massively parallel setup to be able to record enough game to have statistically strong results. Next, we individually tune variables of the different agents to create a champion agent for each algorithm. Those champions are then opposed against each other in a final round of matches used to rank them.

Experimental setup
------------------

A match between two agents is played with the following code, where the variables `player` and `opponent` contain an instance of an agent (a class derived from `Player`).
Because most games finish in less than 200 moves, we limit games to 500 moves to avoid agents playing infinite games. A game that goes over the threshold of 500 moves is considered a draw, regardles of the score of both players.







  


  .. code:: ipython3

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






  
Relevant data from the match can then be recorded in a dictionary like this:




  


  .. code:: ipython3

    {
        "duration": duration,
        "depth": depth,
        "score": game.captures.tolist(),
        "winner": game.winner,
    }








.. parsed-literal::

    {'duration': 0.0265, 'depth': 81, 'score': [30, 14], 'winner': 0}








  
Because the number of matches we expect to play is quite high and a match between two agents might take a few minutes, we have to be able to run matches in a massively parralel setup.

To this effect, we placed the code to run a match in a standalone Python script that accepts the match parameters via environment variables and packaged it in a Docker container. The dictionary showed above is then outputed to the standard output.

This Docker container is then used to launch hundreds of AWS Batch tasks in parallel, their standard output being sent to AWS Cloudwatch to be analyzed later.
Each AWS Batch tasks are allowed 1 vCPU each with 500MB of RAM and are running on C5 compute optimized EC2 instances [#aws_c5]_. 

AWS Batch tasks can be launched with the following function:




  


  .. code:: ipython3

    import boto3
    client = boto3.client('batch')
    
    def submit_match(a, b, pool, side, timeout=600):
        return client.submit_job(
            jobDefinition='run-match',
            jobName=pool,
            jobQueue='match-queue',
            containerOverrides={
                'command': ["python", "simulate.py"],
                'environment': [
                    {'name': 'PLAYER_A', 'value': a % 0},
                    {'name': 'PLAYER_B', 'value': b % 1},
                    {'name': 'POOL','value': pool},
                    {'name': 'SIDE', 'value': str(side)},
                ]
            },
            timeout={'attemptDurationSeconds': timeout},
        )






  
Because we can not be sure an agent has the same strength if it is allowed to be the first player as if it is the second to play, each time we play a match between two agents (A and B), we play the match A vs B and B vs A.




  


  .. code:: ipython3

    def sumbit_symmetric_match(a, b, pool, timeout=600):
        submit_job(a, b, pool, side=0, timeout=timeout)
        submit_job(b, a, pool, side=1, timeout=timeout)






  
Algorithm tuning
----------------

Now that we have a way to run a match between two agents of our choice and record the result, we can start tuning each algorithm individually to create be best agent possible for a given algorithm.

.. todo:: Insert here a paragraph about the (non-)transitivity of the relation "A wins against B". The best way to avoid this problem would be to play a full tournament for each possible value of a variable. But this is not feasible. However, we think that the relation is fairly transitive inside a single algorithm family. This enables us to play a much smaller amount of matches.



:math:`\varepsilon`-Greedy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The first agent we have to tune is :math:`\varepsilon`-Greedy and it has one parameter, :math:`\varepsilon` that can very in the interval :math:`[0, 1]`. As running a match between two :math:`\varepsilon`-Greedy agents takes less than 100ms, playing thousands of matches is computaionaly feasible.

We thus pick evenly spaced values of :math:`\varepsilon` in the interval :math:`[0, 1]` and play 50 matches for each pair of values of :math:`\varepsilon`.




  


  .. code:: ipython3

    search_space = np.linspace(0, 1, 21)
    
    for i in range(25):
        for eps1 in search_space:
            for eps2 in search_space:
                player = f"GreedyPlayer(%s, {eps1})"
                opponent = f"GreedyPlayer(%s, {eps2})"
                sumbit_symmetric_match(player, opponent, "epsilon-greedy-tuning")






  
The results of these matches is shown in :numref:`Figure %s <eps-matrix>` below in wich we can see despite the noise that a higher value of :math:`\varepsilon` (meaning the agent choses most often the greedy approach) is stronger than a lower value. Due to the noise in the data despite the high number of games played it is hard to know for sure if :math:`\varepsilon = 1` is the optimium or if it is a bit lower. We will keep a value of :math:`\varepsilon = 0.95` for the rest of this work.

.. _eps-matrix:

.. figure:: /notebooks/plot-eps.png

  Heatmap of the win ratio of the row player against the column player.





  
MCTS
~~~~

The MCTS agent has a parameter :math:`t` that states how much time the agent may spend on simulation during its turn.
As :cite:`kocsis2006bandit` have shown that given enough time MCTS converges to the minimax tree and thus is optimal, we know that the higher is :math:`t`, the better the agent will be. However, since we are constrained by the capacity of our computation resources, we have to choose a reasonable value of :math:`t`.

Given our objective of producing an agent capable of playing against a human, choosing a value of :math:`t` higher than 1 minute is unrealistic as the human will not want to wait more than that at each turn of the game. While 1 minute is an upper bound, having a much smaller waiting time at each turn would be valuable. We think that  :math:`t = 5s` is a reasonable value.

As stated earlier, we know that the strength of the agent is an increasing function of :math:`t`. However, we don't know the shape of this function. We compare the strength of MCTS(t=5) against a range of values of :math:`t' \in \{0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40\}` by playing 10 matches for each value of :math:`t'`.




  


  .. code:: ipython3

    search_space = [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40]
    
    for i in range(5):
        for time in search_space:
                player = "MCTSPlayer(%s, td(seconds=5))"
                opponent = f"MCTSPlayer(%s, td(seconds={time}))"
    
                sumbit_symmetric_match(player, opponent, "mcts-5s-time-compare", timeout=60*100)






  
While the results showin in :numref:`Figure %s <mcts-time_5s>` are also noisy, we indeed see that the strength of MCTS increases with :math:`t` but the slope of the curve is not very important after :math:`t=5s` so we decide that :math:`t=5s` is a good compromise between strength and waiting time.


.. _mcts-time_5s:

..  figure:: notebooks/mcts-time.png

  Strength of MCTS related to the allowed simulation time budget




  
UCT
~~~

The UCT agent has 2 variables that we can tune, :math:`t` as in MCTS and :math:`c` the balance between exploration and exploitation. We will fix :math:`t=5s` so that we can fairly compare MCTS and UTC later.
:cite:`kocsis2006bandit` has shown that :math:`c=\frac{\sqrt{2}}{2}` is a good starting value. We thus play matches of UCT(:math:`c=\frac{\sqrt{2}}{2}`) against a range of 11 values equaly spaced between 0.2 and 2.2




  


  .. code:: ipython3

    search_space = np.linspace(0, 2, 11) + 0.2
    
    for i in range(25):
        for c in search_space:
                player = "UCTPlayer(%s, td(seconds=5), c=math.sqrt(2)/2)"
                opponent = f"UCTPlayer(%s, td(seconds=5), c={c:.2f})"
    
                sumbit_symmetric_match(player, opponent, "uct-tuning-c")






  
What we see in :numref:`Figure %s <utc-tuning-c>` is a bell curve with some noise and a plateau around :math:`c = \sqrt(2) / 2`. The noise is louder on the right than on on the left of its maximum. An explanation for this could be that on the left, as :math:`c` is lower, there is not much exploration so the algorithm is more deterministic while it's the opposite on the right and each simulation could be either really good or really bad depending on luck.

As the maximum of the bell curve is around :math:`c = \sqrt(2) / 2` it seems to confirm that it is the optimum value for UCT.

.. _utc-tuning-c:

.. figure:: notebooks/uct-value.png

  Strength of UCT(:math:`c=\frac{\sqrt{2}}{2}`) against other values of :math:`c`.




  
Under the assumption that the curve is smooth, we know that :math:`c = \sqrt(2) / 2` is will win against any value of :math:`c \in [0.2, 2.2]`. While this result might be convenient, we don't know if the relation of one agent winning against another is transitive, so while :math:`c = \sqrt(2) / 2` beats every value, we might have another value of :math:`c = \sqrt(2) / 2` that beats every :math:`c \neq \sqrt(2) / 2` by a bigger margin. To have a better intuition it is the case or not, we can also run the same experiment as above but with :math:`c = 1.5` to see if we were not lucky by using :math:`c = \sqrt(2) / 2` the first time. 




  


  .. code:: ipython3

    search_space = np.linspace(0, 2, 11) + 0.2
    
    for i in range(25):
        for c in space:
                player = "UCTPlayer(%s, td(seconds=5), c=1.5)"
                opponent = f"UCTPlayer(%s, td(seconds=5), c={c:.2f})"
    
                sumbit_symmetric_match(player, opponent, "uct-tuning-c-15")






  
While the curve in :numref:`Figure %s <uct-tuning-c-15>` is not as smooth as in the first experiment, the result of the matches against :math:`c = 1.5` seem to show the same curve with a maximum at :math:`c = \sqrt(2) / 2`.

.. _uct-tuning-c-15:

.. figure:: notebooks/uct-c-15.png

  Strength of UCT(:math:`c=1.5`) against other values of :math:`c`.





  
Comparing algorithms
--------------------

Now that we have found the best values of each variable for each algorithm, we have a small (:math:`N = 5`) set of agents to compare to each other. As the assumptions of smoothness and transitivity we placed in the previous section might not hold when comparing agents using different algorithms, we need to define a stronger framework to find the best agent.


How to compare A and B
~~~~~~~~~~~~~~~~~~~~~~


The first step is to define a way to compare agent A and B. The probability that A wins is denoted by :math:`p` and is unknown (the probability that B wins is :math:`1-p`).
Our null hypothesis is that both agents are equaly strong (:math:`p=0.50`) and the alternative hypothesis is that they are of different strength (:math:`p \neq 0.50`).
To compare agents A and B, we run :math:`N` matches and A wins :math:`n` times (thus B wins :math:`N-n` times).

Using the SciPy function :code:`scipy.stats.binom_test`, we then compute the p-value.
If it is lower than :math:`5\%`, we traditionally reject the null hypothesis.
This guarantees that, conditional on H0 being true, the probability of making an incorrect decision is :math:`5\%`.
But if H1 is true, the probability of an incorrect decision is not necessarily :math:`5\%`: it depends on the number :math:`N` of matches and on the true value of :math:`p`.
To ensure that the probability of an incorrect decision, conditional on H1, be acceptable, we resort to the concept of statistical power.

Suppose the true probability :math:`p` is :math:`0.75`. This is very far from the null hypothesis. In that case, we want the probability of choosing H1 (not making an incorrect decision) to be high (for instance :math:`95\%`). This probability is the power and can be computed by means of the R function powerBinom implemented in the R package exactci:




  


  .. code:: ipython3

    import scipy.stats
    
    #powerBinom(power = 0.95, p0 = 0.5, p1 = 0.75, sig.level = 0.05, alternative = "two.sided")






  
The output of this command is the number :math:`N` of matches needed to achieve the desired power and it is 49. As we always play a even number of matches between two agents (A vs. B and B vs. A), we decide that we need :math:`N=50` matches.

Now that we know the amount of matches we need to play to be able to assertain that H1 is probable enough, we still need to know how many matches of the 50 an agent needs to win so we may declare H1 true. This can be done with the :code:`scipy.stats.binom_test` function.




  


  .. code:: ipython3

    for wins in range(50):
        pvalue = scipy.stats.binom_test(wins, 50, p=0.5, alternative="greater")
        if pvalue < 0.05:
            print("If a agent wins", wins, "matches, we can reject H0 with a p-value of", round(pvalue, 4))
            break




.. parsed-literal::

    If a agent wins 32 matches, we can reject H0 with a p-value of 0.0325





  
Proof of non-transitivity
~~~~~~~~~~~~~~~~~~~~~~~~~

We now have a way to determine if an agent is stronger than another but we don't have a way to order all our agents regarding to their strength. In the following, we prove that a total order between all agents does not exist by showing that the relation of strength between two agents is not transitive.

Lets define 3 theoretical algorithms: each of them play the first move at random and the next moves of the match depending on the first move in three different ways: always playing the best move (noted :math:`+`), never playing the best move (noted :math:`-`) or playing at random (noted :math:`r`).

.. table:: Moves of the theoretical algorithms depending on the first move of the game.

    +------------+-----------+-----------+-----------+
    | First move | A         | B         | C         |
    +------------+-----------+-----------+-----------+
    | 1, 2       | :math:`+` | :math:`r` | :math:`-` |
    +------------+-----------+-----------+-----------+
    | 3, 4       | :math:`r` | :math:`-` | :math:`+` |
    +------------+-----------+-----------+-----------+
    | 5, 6       | :math:`-` | :math:`+` | :math:`r` |
    +------------+-----------+-----------+-----------+


If A and B are playing matches, if the match starts with move:
 - 1 or 2: A wins all the time,
 - 3 or 4: A wins more than half the matches,
 - 5 or 6: B wins all the matches.
 
So A wins more matches than B and we can say :math:`A > B`. By doing the same with B vs. C and C vs. A we have :math:`B > C` and :math:`C > A`. Thus the relation between these 3 theoretical algorithms is not transitive.

How to compare more than two agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described above, transitivity can not be proved in all cases.
Because we want to compare agents using different algorithms, we think that we can not make the assumption
(made previously inside an algorithm family) that strength is transitive.
We thus resort to playing a full tournament between the agents we want to compare.

We can play a trounament of 50 matches between every pair of agents. This is a values tournament as each pair has a score contained in :math:`[0, 50]`. 
We then transfrom this valued tournament in a binary tournament by deciding that :math:`A > B`, :math:`A < B` or :math:`A = B` if respectively :math:`A` wins more than 31 matches , A loses more than 31 matches or neither wins more than 31 matches.

This in turn enables us to use the framework of tournament solutions :cite:`laslier` to analyze the results and eventualy find a total order.

.. todo:: We might still want to rank our algorithms on a scale with total ordering. There are a lot of algorithms to do this (Elo ranking and others). Research is still developing on this subject and there is no consensus on the right method to use. This is beyond the topic, i won't go further.

https://www.researchgate.net/publication/287630111_A_Comparison_between_Different_Chess_Rating_Systems_for_Ranking_Evolutionary_Algorithms





  
Tournament results
------------------

We select the best agent for every algorithm and make each of them play 50 match against each other.




  


  .. code:: ipython3

    algos = [
        "RandomPlayer(%i)",
        "GreedyPlayer(%i, 0.95)",
        "MCTSPlayer(%i, td(seconds=5))",
        "UCTPlayer(%i, td(seconds=5), c=math.sqrt(2)/2)",
        "GreedyUCTPlayer(%i, td(seconds=5), c=math.sqrt(2)/2)",
    ]
    
    for i in range(25):
        for a in algos:
            for b in algos:
                sumbit_symmetric_match(a, b, "tournament")






  
The results, displayed in a matrix in :numref:`Figure %s <matrix>`, show that UCT and GreedyUCT beat every other agent. There is no clear winner between those 2 champions though.

.. _matrix:

.. figure:: notebooks/matrix.png

  Matrix representation of the valued tournament between every algorithm
  
.. todo:: We still have to transform the values tournament in a binary one and then analyze it with the framework of tournament solutions.

Limitations
-----------

.. todo:: As we only compare the champions of each algorithm, we might have a non-champion that would still won against another algo. Then we would not have a complete pre-order. We can not do this due to compute limitation.





  
==========
Conclusion
==========




  
========
Appendix
========

Bibliography
------------

.. warning::
   Some papers are currently wrongly cited. 

.. bibliography:: refs.bib
   :style: custom




  
Footnotes
---------

.. [#source_bao] Picture by Yintan under Creative Commons SA license https://commons.wikimedia.org/wiki/File:Bao_europe.jpg

.. [#Fly08] Jennifer Flynn. Independent study quarterly reports.
 http://users.soe.ucsc.edu/~charlie/projects/SlugGo/, 2008

.. [#GS07] Sylvain Gelly and David Silver. Combining online and offline
 knowledge in uct. In ICML ’07: Proceedings of the 24th
 Internatinoal Conference on Machine Learning, pages 273–280.
 ACM, 2007. 
 
 .. [#aws_c5] C5 instances contain a 2nd generation Intel Xeon Scalable Processor (Cascade Lake) with a sustained all core Turbo frequency of 3.6GHz.



