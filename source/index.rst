  
.. contents:: Table of Contents
   :depth: 3

.. sectnum::




  
============
Introduction
============

Awale is a popular board game played mainly in Africa. The board has two rows of six pits, each containing four seeds in the initial state.


.. figure:: /_static/awale.jpg

   A typical Awalé board in the initial state.
	
At each turn, the players move some seeds and can potentially capture some of them, according to deterministic rules. The goal of the game is to capture more seeds than one's opponent. 

.. todo:: Explain here what i'm going to do in my thesis, why it is interesting and why it is new.

In Section 2, we present Awale in detail.
Section 3 reviews various approaches to solve Awale: retrograde analysis, Minimax, and basic Monte Carlo Tree Search.
In Section 4, we described more advanced versions of MCTS and in particular UCT.
Section 5 presents some empirical results (simulations) allowing to compare several MCTS algorithms and Section 6 concludes.




  
=====
Awale
=====

The subject of our study, Awale is an ancient, two player board game originating from Ghana.
This game is also sometimes called Awele, Oware, Owari or Ayo in the neighouring countries, languages and cultures :cite:`crane1982`.

Originally, the game is played on the ground, by digging two rows of six small pits, each containing
stones, seeds or shells. In the present document, we will name them seeds. The game is also often played on a wooden board symbolizing the original dirt pits.
The board can be schematized as in Fig. YYY, every big circle representing a pit and every small disc representing a seed.
Numbers at the bottom right of each pit are the counts of seeds in each pit for better readability.
Each row of pits is owned by a player that sits in front of it (:numref:`see Fig. %s <intro-kalah>`).
For the sake of convenience, the players are named North and South.   
The 6 pits from the top row belong to North and the 6 from the bottom to South.

The players take turns, a player removing all the seeds from a pit and placing them in other pits following the rules. This is called sowing the seeds. This can result in a configuration in which the player is allowed to capture some seeds according to the rules.
The goal for each player is to capture more seeds than his opponent.
The rules vary slightly across countries and will be detailed in Section XXX.







  









.. raw:: html
    :file: index_files/index_4_0.svg








  
Mancala
-------

The Mancala games are an ancient family of game that are played on many continents :cite:`deVoogt2008`, Awale being one of them.
The word mancala comes from the Arabic word "نقلة", transliterated as "naqala" and literally meaning "to move". 

Like Awale, Mancala games can consist of rows of pits, some of them having more than two rows (see Fig. XXX, a Bao board) and sometimes extra pits with a special role. Mancala games can sometimes be played by more than two players.

.. _intro-kalah:

.. figure:: _static/intro-kalah.jpg

  A wooden Mancala game [#source_kalah]_

There are too many variants of the Mancala games to list them all here, but a
few notable ones are Awale, Wari, Bao, Congkak and Kalah.

In particular, Kalah is a commercial, modern variant of Mancala, introduced in the 1950s by William Julius Champion Jr., that is widespread in the United States. :cite:`irving2000solving`. This variant has been studied in Artifical Intelligence as early as 1964 by :cite:`russel1964`.
Nowadays, Kalah is often used as an example game in computer-science courses.

Mancala games in general, while less known than Chess or Go, are quite popular and
are played in tournaments around the world, both in offline and online competitions :cite:`owaresociety,fandom_tournaments`.

Mancala games have also been studied in Computer Science and Artificial Intelligence :cite:`deVoogt2008`. Tournaments opposing computers on both sides have been organised multiple times, notably in the Computer Olympiad organized by the International Computer Games Association :cite:`icga_olympiad`.





  
Rules of the game
-----------------

The basic rules of Awale are the same everywhere but there are some minor differences around the globe and in the literature.
The rules presented here and implemented later in this thesis are inspired from :cite:`goot2001` and adapted by us.

The goal for earch player is to capture more seeds than his opponent. Because the
game has 48 seeds, capturing 25 is enough for a player to win and ends the game.

Each player plays alternatively, without the right to pass his turn. A
player's turn consists in choosing one of his non-empty pits, picking all seeds
contained in the pit and sowing them one by one in every consecutive pits on the right
(rotating counter-clockwise). The player thus has at most 6 possible moves at
each turn (one per non-empty pit owned by him).

Usually, the player that starts the game is the oldest player. In this work, South will always play first.

In this work, the pits of a player are numbered left to right from his point of view as shown in Fig. YYY. :math:`1` being the leftmost pit of South, until :math:`6` at the far right. The same holds for North: :math:`1'` to :math:`6'`.

.. todo:: Insert figure with the pit numbering

As an example, in the initial state (showed inf Fig. `initial_board` YYY), the first player to move is South (on the bottom) and he plays :math:`4` (highlighted in the figure in red), the board will then be in the  state shown in Fig. `first_move` YYY.




  









.. raw:: html
    :file: index_files/index_7_0.svg








  
When the last sowed seed is placed in a pit owned by the opponent and, after sowing,
the pit contains two or three seeds, the content of the pit is captured by
the player and removed from the game. If the pit preceding the captured pit also
contains two or three seeds, it is also captured. The capture continues until a
pit without two or three seeds is encountered. When the capture is ended the
next player's turn starts.

Otherwise, when the last sowed seed is placed in a pit that, after sowing, contains one seed, more
than 3 seeds or in the current player's own pits, the turn of the player is ended without
any capture.
For example, if South plays :math:`4` in the configuration shown in Fig. `pre_capture` YYY, he will
be able to capture the seeds in pits 2' and 3' (highlighted in red in Fig. post_capture YYYY).




  









.. raw:: html
    :file: index_files/index_9_0.svg








  









.. raw:: html
    :file: index_files/index_10_0.svg








  
If the pit chosen by the player contains more than 12 seeds, the sowing makes
more than a full revolution of the board and the starting hole is skipped during the second
and subsequent passes.

If the current player's opponent has no seed left in his half board, the
current player has to play a move that gives him seeds if such a move exists.
This rule is called the "feed your opponent".
In Fig. `feed` YYY, South has to play pit 5 because playing pit 1 would leave the opponent without any move to play.




  









.. raw:: html
    :file: index_files/index_12_0.svg








  
When a player has captured more than 25 seeds the game ends and he wins. If both
players have captured 24 seeds, the game ends by a draw. If the current player's
pits are all empty, the game ends and the player with the most captures wins.

The last way to stop the game is when a position is encountered twice in the
same game (there is a cycle): the game ends and the player with most captures
wins.




  
Implementation of the rules
---------------------------

.. todo:: Insert here some text explaining that we are showing inline code as this document is in fact a big jupyter notebook that is executed.

In this subsection, we define in multiple steps a Python :code:`Game()` class holding the state of the game and its rules. We will then succesively inherit from it to add the rules and some sonvenience methods.

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
            '''Returns the board as seen by a player'''
            if self.current_player == 0:
                return self.pits
            else:
                return np.roll(self.pits, 6)
        
        @property
        def current_player_pits(self) -> List[int]:
            '''Returns a 6-vector containing the pits owned by the current player'''
            if self.current_player == 0:
                return self.pits[:6]
            else:
                return self.pits[6:]
    
        @property
        def current_opponent(self) -> int:
            return (self.current_player + 1) % 2
        
        @property
        def adverse_pits_idx(self) -> List[int]:
            '''Returns the indices in the `self.pits` array owned by the opposing player'''
            if self.current_player == 1:
                return list(range(6))
            else:
                return list(range(6, 6 * 2))






  
Now that the base is set, we start implementing the rules,
some of them being deliberately excluded from this implementation:

-  Loops in the game state are not checked (this considerably speeds up the computations and we did not encounter a loop in our preliminary work);
-  The "feed your opponent" rule is removed; This makes the
   rules simpler and we expect it does not tremendously change the complexity of the game.
   
.. todo We did later encounter loops after running way more simulations. But this only happend yet using basic algorithms (greedy vs greedy for example). For now, we simulate 500 truns, if we hit this threshold, we declare a tie. This should be detailed in the experimental setup




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        @property
        def legal_actions(self) -> List[int]:
            '''Returns a list of indices for each legal action for the current player'''
            our_pits = self.current_player_pits
            # Return every pit of the player that contains some seeds
            return [x for x in range(6) if our_pits[x] != 0]
        
        @property
        def game_finished(self) -> bool:
            # Does the current player has an available move ?
            no_moves_left = np.sum(self.current_player_pits) == 0
            
            # Has one player cpatured more than half the total seeds ?
            HALF_SEEDS = 24 # (there are 2*6*4=48 seeds in total)
            enough_captures = self.captures[0] > HALF_SEEDS or self.captures[1] > HALF_SEEDS
            
            # Is it a draw ? Does both player have 24 seeds ?
            draw = self.captures[0] == HALF_SEEDS and self.captures[1] == HALF_SEEDS
            
            # If one of the above three are True, the game is finished
            return no_moves_left or enough_captures or draw
        
        @property
        def winner(self) -> Optional[int]:
            '''Returns the winner of the game or None if the game is not finished or in a draw'''
            if not self.game_finished:
                return None
            # The game is finished but both player have the same amount of seeds: it's a draw
            elif self.captures[0] == self.captures[1]:
                return None
            # Else, there is a winner: the player with the most seeds
            else:
                return 0 if self.captures[0] > self.captures[1] else 1






  
We can now define the :code:`Game.step(i)` method that is called for every step of the game.
It takes a single paramter, :code:`i`, and plays the i-th pit in the current sate.
This method returns the new state, the amount of seeds captured and a boolean informing whether the game is finished.




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        def step(self, action: int) -> Tuple[Game, int, bool]:
            '''Plays the action given as parameter and returns:
                - a the new state as a new Game object,
                - the amount of captured stones in the transition
                - a bool indicating if the new state is the end of the game
            '''
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
            new_game = type(self)(
                pits,
                current_player,
                captures
            )
    
            return new_game, round_captures, new_game.game_finished







  
As the game rules are now implemented, we can add some methods to display the current state of the board to the user, either in textual mode or as an SVG file that can be displayed inline in this document.




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        def show_state(self):
            '''Print a textual representation of the game to the stdandard output'''
            if self.game_finished:
                print("Game finished")
            print("Current player: {} - Score: {}/{}\n{}".format(
                self.current_player,
                self.captures[self.current_player],
                self.captures[(self.current_player + 1) % 2],
                "-" * 6 * 3
            ))
            
            pits = []
            for seeds in self.view_from_current_player:
                pits.append("{:3}".format(seeds))
            
            print("".join(reversed(pits[6:])))
            print("".join(pits[:6]))
        
        def _repr_svg_(self):
            '''Return a SVG file representing the current state to be displayed in a notebook'''
            board = np.array([
                list(reversed(self.pits[6:])),
                self.pits[:6]
            ])
            return board_to_svg(board, True)






  
To show a minimal example of the implementation, we can now play a move and have its results displayed here.




  


  .. code:: ipython3

    g = Game() # Create a new game
    g, captures, done = g.step(4) # play the 5th pit (our implementation starts at 0)
    g # Display the resulting board inline








.. raw:: html
    :file: index_files/index_26_0.svg








  
=====================
Awale and Game Theory
=====================



Solving games
-------------

**Theorem** :cite:`neumann1928` In every two-player game (with perfect information) in which the set of outcomes is :math:`0 = \{I \, wins, II \, wins, Draw\}`, one and only one of the following three alternatives holds:
 1. Player :math:`I` has a winning strategy
 2. Player :math:`II` has a winning strategy
 3. Each of the two players has a strategy guaranteeing at least a draw.
 
Solve a position.

A game where all positions are solved is a solved game

Define:
 - agent policy
 
As stated in Section XXX, the branching factor of Awale is 6. This is very small compared to the branching factor of 19 for the game of Go and makes Awale much easier to explore and play.

If we build the complete tree, we compute every possible state in the game and every
leaf of the tree is a final state (end of a game). As said, previously, computing the complete tree is not
ideal for Awale (it has :math:`\approx 8 \times 10^{11}` nodes) and
computationally impossible for games with a high branching factor (unless very shallow).



A strongly solved game is defined by Allis :cite:`Allis94searchingfor` as:

    For all legal positions, a strategy has been determined to
    obtain the game-theoretic value of the position, for both players, under
    reasonable resources.

A solved game is, of course, much less interesting to study than an
unsolved one as we could just create an agent that has the knowledge of each
game-theoretic position values and can thus perfectly play.

(:math:`m,n`)-Kalah is a game in the Mancala family with :math:`m` pits per
side and :math:`n` seeds in each pit plus two extra pits with a special role.
It has been solved in 2000 for :math:`m \leq 6`  and :math:`n
\leq 6` except (:math:`6,6`) by :cite:`irving2000solving` and in
2011 for :math:`n = 6, m=6` by :cite:`kalah66`.





Now that we know the rules, we can see that Awale

* is sequential: the opponents play one after the other;
* hold no secret information: each player has the same information about
  the game;
* do not rely on randomness: the state of the game depends only on the actions
  taken sequentially by each player and an action has a deterministic result.

This type of game is called a sequential perfect information game
:cite:`osborne1994course`.

We can also see that the game is a two player zero-sum game.


.. todo:: This section is not done and will be heavily reworked. The following block of text is copied from "A Survey of Monte Carlo Tree Search Methods" and should not be in the finished document.

> 1) Combinatorial Games: Games are classified by the fol-
lowing properties:
• zero sum: whether the reward to all players sums to zero
(in the two-player case, whether players are in strict com-
petition with each other);
information: whether the state of the game is fully or par-
tially observable to the players;
• determinism: whether chance factors play a part (also
known as completeness, i.e., uncertainty over rewards);
• sequential: whether actions are applied sequentially or si-
multaneously;
• discrete: whether actions are discrete or applied in real
time.
Games with two players that are zero sum, perfect informa-
tion, deterministic, discrete, and sequential are described as
combinatorial games.

^ "A Survey of Monte Carlo Tree Search Methods"

Convergence.
We consider a game to be convergent when the size of the state space decreases as the game progresses. If the size of the state space increases, the game is said to be divergent.
In some games games like Chess, Checkers and Awari the players may capture pieces in the course of the game and may never add them back these are called convergent games :cite:`vandenherik2002`.
On the contrary, in some others the number of pieces on the board increases over time as a player’s move consists of putting a piece on the board. Examples of these games are Tic-Tac-Toe, Connect Four and Go. Those are divergent.


Other games in this category are for example Chess, Go, Checkers or even
Tic-tac-toe and Connect Four. Sequential perfect information games are particularly interesting
in computer science and artificial intelligence because they are easy to simulate.




  
Perfect information games as finite state machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When viewed from an external point of view, these types of games can be
modelized as finite states machines with boards being states (the initial board
is the initial state), each player's action being a transition and wins and draws
being terminal states.

It might be tempting to try to enumerate every possible play of those games by
starting a game and recursively trying each legal action until the end of the game
to find the best move for each state.

Unfortunately, most of the time, this is not a feasible approach due to the size
of the state space. As an example, Romein et al. claims that Awale has
889,063,398,406 legal positions :cite:`romein2003solving` and the exact number
(:math:`\approx 2.08 \times 10^{170}`) of legal positions in Go (another popular perfect information game)
is so big that it has only recently been determined :cite:`tromp2016`. Such state space are too
big to be quickly enumerated.




  
Tree representation
~~~~~~~~~~~~~~~~~~~

A [combinatorial game XXX] like Awale can be represented as a tree in a straightforward way where every node is a state of the game.
The root of the tree represents the initial state.
If in a state :math:`s` the current player plays action :math:`i` resulting in state :math:`s'` then :math:`s'` will be the i-th child of the node representing :math:`s`.

This results in the following properties:
    - As the current player at the root node is South and that players alternate after each turn,
      the tree containsalternating layers of current players:
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
                captures = new_game.captures[self.current_player] - self.captures[self.current_player]
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

Many algorithms have been proposed and studied to play [sequential perfect information XXX] games.
A few examples detailed here are retrograde analysis, Minimax, :math:`\alpha-\beta` pruning,
Monte Carlo tree search (MCTS) and the new approch from Deepmind: Alpha Zero :cite:`AlphaGoZero`.

We will quickly present those and then focus on MCTS and its variants as they are computationaly feasible and do not require expert knowledge about the given game to make reasonable decisions.



Alpha-Beta pruning Minimax
--------------------------




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

In this subsection, we define Markov Decision Processes (MDP) and modelize Awale with this framework. We then describe and detail Monte Carlo Tree Search, a policy-optimization algorithm for finite-horizon, finite-size MDPs. 

Markov descision processes
~~~~~~~~~~~~~~~~~~~~~~~~~~

In decision theory a Markov decision process (MDP) models sequential decision problems in fully observable environments.
In this model, an agent iteratively observes the
current state, selects an action, observes a consequential probabilistic state transition, and receives a reward
according to the outcome.
Importantly, the agent decides each action based on the current state alone and not the full history of past states, providing a Markov independence property :cite:`markov1954`.

Mathematically, an MDP consists of the following components:
 - a state space, :math:`X` ;
 - an action space, :math:`A`;
 - a transition probability function, :math:`P : X × A × X \rightarrow [0, 1]`; and
 - a reward function, :math:`R : X × A \rightarrow [0, 1]`.

If all transitions from a state have zero probability, the state is called a terminal state. By analogy, states that are not terminal are called nonterminal.

Markov games
~~~~~~~~~~~~

A Markov game can be thought of an extension of MDP environments
where a player may take an action from a state, but the reward and state transitions are uncertain as they depend on the adversary’s strategy [2].

[2] Michael Littman. Markov games as a framework for multi-agent reinforcement learning, 1994

For most common games like Go and Chess the transition and reward functions are deterministic given the actions of the player and the opponent, but we consider them non-deterministic values sine the player and opponent may use randomized strategies.
Finding an optimal policy in this scenario seems impossible since it depends critically on which adversary is used. The way this is resolved is by evaluating a policy with respect to the worst opponent for that policy.
The goal now is to find a policy that will maximize the reward knowing that this worst case opponent will then minimize the reward after the action is played (the fact that this is a zero-sum game makes it so the opponent will maximizes your negative reward); this idea is used widely in practice in what is known as
the minimax principle. This optimal policy is a bit pessimistic since you won’t always be playing against a worst-case opponent for that policy, but it does allow to construct a policy for a game that can be used against any adversary.

^ Lecture 19: Monte Carlo Tree Search: : Kevin Jamieson
^ https://pdfs.semanticscholar.org/574e/6872df3fe9b89afa98a7bdeef710a931da34.pdf

Monte Carlo Tree Search
~~~~~~~~~~~~~~~

As Awale can be represented as an MDP, we could be tempted to use the usual framework of Q-Learning [Cite XXX] to find the best policy to maximise our reward. But since the state space is huge, this is computationally difficult or even impossible in memory and time constrained cases.
To overcome this computational problem, the MCTS method constructs only a part of game the tree by sampling and tries to estimate the chance of winning based on this information.

Algorithm
~~~~~~~~

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
less visited nodes. Kocsis et al. [has shown XXX faux] that :math:`\frac{\sqrt{2}}{2}`
:cite:`kocsis2006bandit` is a good value when rewards are in :math:`[0, 1]`.

In step 3, the playouts are played by chosing an action from an uniform distribution since it is the first time these nodes
are seen and we do not have a generic evaluation function do direct the playout
towards 'better' states.




  
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




  
=================
Empirical results
=================


Experimental setup
------------------

.. todo:: As an algorithm might have an advantage we will always play A vs B and then B vs A. We thus have an even number or matches so we pick 50 and not 49 matches. Here we should also explain where we run the simulations (hardware setup) some explanation of the software distribution of the computation and then describe the simulation in itself and the limit to 500 steps.



Algorithm tuning
----------------



:math:`\varepsilon`-Greedy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The first agent we have to tune is :math:`\varepsilon`-Greedy and it has one parameter, :math:`\varepsilon` that can very in the interval :math:`[0, 1]`. As running a match between two :math:`\varepsilon`-Greedy agents takes less than 100ms, playing thousands of matches is computaionaly feasible.

We thus pick 21 evenly spaced values of :math:`\varepsilon` in the interval :math:`[0, 1]` and play 50 matches for each pair of values of :math:`\varepsilon`. The results of these matches is shown in Fig. XXX below.


.. figure:: /notebooks/plot-eps.png



MCTS
~~~~

The MCTS agent has a parameter :math:`t` that states how much time the agent may spend on simulation during its turn.
As (Kocsis and Szepesvári) XXX have shown that given enough time MCTS converges to the minimax tree and thus is optimal, we know that the higher is :math:`t`, the better the agent will be. However, since we are constrained by the capacity of our computation ressources, we have to choose a reasonable value of :math:`t`.

Given our objective of producing an agent capable of playing against a human, choosing a value of :math:`t` higher than 1 minute is unrealistic as the human will not want to wait more than that at each turn of the game. While 1 minute is an upper bound, having a much smaller waiting time at each turn would be valuable. We think that  :math:`t = 5s` is a reasonable value.

As stated earlier, we know that the strength of the agent is an increasing function of :math:`t`. However, we don't know the shape of this function. We play compare the strength of MCTS(t=5) against a range of values of :math:`t' \in \{0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40\}` by playing 10 matches for each value of :math:`t'`.

..  figure:: notebooks/mcts-time.png

While the results are noisy, we still see that the strength of MCTS does not increase quickly after :math:`t=5s` so we decide that :math:`t=5s` is a good compromise between strength and waiting time.


UCT
~~~

The UCT agent has 2 varibles that we can tune, :math:`t` as in MCTS and :math:`c` the balance between exploration and exploitation. We will fix :math:`t=5s` so that we can fairly compare MCTS and UTC later.
Kocsis et al.:cite:`kocsis2006bandit` has shown that :math:`c=\frac{\sqrt{2}}{2}` is a good starting value. We thus play matches of MCTS(:math:`c=\frac{\sqrt{2}}{2}`) against a range of 11 values equaly spaced between 0.2 and 2.2


.. todo:: :math:`c = \sqrt(2) / 2` is a good theoritical starting point (see aglo description) so we run matches with :math:`c = \sqrt(2) / 2` against a range of values, from 0.1 to 2. What we see is a bell curve with some noise. :math:`c = \sqrt(2) / 2` seems indeed the best value.



.. figure:: /notebooks/plot-c.png

.. todo:: Interpretation of the curve: The curve has a lot of noise on the right, not much on the left. An explanation for this could be that on the left, there is not much exploration so the algorithm is more deterministic while it's the opposite on the right and each simulation could be really good or really bad depending on luck.




Comparing algorithms
--------------------

How to compare A and B
~~~~~~~~~~~~~~~~~~~~~~


We wish to compare algorithms A and B. The probability that A wins is denoted by :math:`p` and is unknown (the probability that B wins is :math:`1-p`). Our nulhypothesis is that :math:`p=0.50` and the alternative hypothesis is that :math:`p \neq 0.50`. To compare algorithms A and B, we run :math:`N` simulations and A wins :math:`n` times (thus B wins :math:`N-n` times). Using the Python function xxx, we then compute the p-value. If it is lower than :math:`5\%`, we traditionally reject the nulhypothesis. This guarantees that, conditional on H0 being true, the probability of making an incorrect decision is :math:`5\%`. But if H1 is true, the probability of an incorrect decision is not necessarily :math:`5\%`: it depends on the number :math:`N` of simulations and on the true value of :math:`p`. To ensure that the probability of an incorrect decision, conditional on H1, be acceptable, we resort to the concept of statistical power.

Suppose the true proability p is :math:`0.75`. This is very far from the nulhypothesis. In that case, we want the probability of choosing H1 (not making an incorrect decision) to be high (for instance :math:`95\%`). This probability is the power and can be computed by means of the R function powerBinom implemented in the R package exactci:
powerBinom(power = 0.95, p0 = 0.5, p1 = 0.75, sig.level = 0.05, alternative = "two.sided")
The output of this command is the number :math:`N` of simulations needed to achieve the desired power and it is 49.

.. todo:: Compute critical value to decide an algorithm is better than an other. If we have 50 matches, and one algorithm wins strictly more than 31


How to compare more than 2
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo:: Now that we can compare 2 algorithms, we can ask ourselves if this relation is transitive.
In fact, we can prove that this relation is not always transitive in a mind experiment with 3 algorithms and a cycle. (I will write the proof later)

.. todo:: Transitivity can not be assumed in all cases. However, we can hope that if we optimize a parameter in an algorithm, transitivity exists. (We can show experimental clues that it is the case). If we want to compare different algorithms, we will have to use a full tournament.

.. todo:: We transform the valued tournament in a binary tournament. The check if the tournament a complete pre-order.


Run + result
------------

.. todo:: Here we run the big tournament with all the algorithms against the others

Limitations
-----------

.. todo:: As we only compare the champions of each algorithm, wa might have a non-champion that would still won against another algo. Then we would not have a complete pre-order. Wa can not do this due to compute limitation.


Ranking
--------

.. todo:: We might still want to rank our algorithms on a scale with total ordering. There are a lot of algorithms to do this (Elo ranking and others). Research is still developing on this subject and there is no consensus on the right method to use. This is beyond the topic, i won't go further. 
https://www.researchgate.net/publication/287630111_A_Comparison_between_Different_Chess_Rating_Systems_for_Ranking_Evolutionary_Algorithms





  
==========
Conclusion
==========




  
========
Appendix
========

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


