  
.. contents:: Table of Contents
   :depth: 3

.. sectnum::

.. topic:: About this work

    This work is a thesis submitted to the Faculty of Sciences in partial
    fulfillment of the requirements for the degree of Master’s in Computer Science.

    This work was originally published on https://mancala.ml as a web page consisting in text
    and notebook cells mixed together.

    The source of this document along with the source code of all experiments
    are available on GitHub_.

    .. _GitHub: https://github.com/C4ptainCrunch/thesis







  
============
Introduction
============

Awale is a popular board game played mainly in Africa. The board has two rows of six pits, each containing four seeds in the initial state.


.. figure:: /_static/awale.jpg

   A typical Awalé board in the initial state.
	
At each turn, the players move some seeds and can potentially capture some of them, according to deterministic rules. The goal of the game is to capture more seeds than one's opponent. 

..todo:: Texte expliquant ce que tu vas faire dans ton memoire et pourquoi c'est interessant, pourquoi c'est nouveau

In Section 2, we present Awale in detail.
Section 3 reviews various approaches to solve Awale: retrograde analysis, Minimax, and basic Monte Carlo Tree Search.
In Section 4, we described more advanced versions of MCTS and in particular UCT.
Section 5 presents some empirical results (simulations) allowing to compare several MCTS algorithms and Section 6 concludes.




  
=====
Awale
=====

The subject of our study, Awale is an ancient, two player board game originating from Ghana.
This game is also sometimes called Oware, Owari or Ayo in the neighouring countries, languages and cultures.

Originally, the game is played on the ground, by digging two rows of six small pits, each containing
stones, seeds or shells. In the present document, we will name them seeds. The game is also often played on a wooden board symbolizing the original dirt pits.
The board can be schematized as in Fig. XXX, every big circle representing a pit and every small disc representing a seed.
Numbers at the bottom right of each pit are the counts of seeds in each pit for better readability.
Each row of pits is owned by a player that sits in front of it (:numref:`see Fig. %s <intro-kalah>`).
For the sake of convenience, the players are named North and South.   
The 6 pits from the top row belong to North and the 6 from the bottom to South.

The game is played turn by turn, a player taking all the seeds from a pit and placing them in others following the rules. This is called sowing the seeds. The resulting configuration of the board might result in the player being allowed by the rules to capture some seeds.
The goal for each player is to capture more seeds than his opponent.







  









.. raw:: html
    :file: index_files/index_5_0.svg








  
Mancala
-------

The Mancala games are an ancient family of game that are played on many continents :cite:`deVoogt2008`, Awale being one of them.
The word mancala comes from the Arabic word "نقلة", transliterated as "naqala", literally meaning "to move". 

Like Awale, Mancala games can consist of rows of pits, some of them having more than two rows and sometimes extra pits with a special role. Mancala games can sometimes be played by more than two players.

.. _intro-kalah:

.. figure:: _static/intro-kalah.jpg

  A wooden Mancala game [#source_kalah]_

There are too many variants of the Mancala games to list them all here, but a
few notable ones are Awale, Wari, Bao, Congkak and Kalah.

In particular, Kalah is a commercial, modern variant of Mancala, introduced in the 1950s by William Julius Champion Jr. that is widespread in the United States. :cite:`irving2000solving`. This variant has been studied in Artifical Intelligence as early as 1964 by :cite:`russel1964`.
Nowadays Kalah is often used as an example game in computer-science courses.

Mancala games, while less known than Chess or Go, are played in tournaments around the world, both in offline and online competitions :cite:`owaresociety,fandom_tournaments`.
Tournaments opposing computers on both sides also have been organised multiple times, notably in the Computer Olympiad organized by the International Computer Games Association :cite:`icga_olympiad`.





  
Rules of the game
-----------------

The goal for both players is to capture more seeds than its opponent. As the
game has 48 seeds, capturing 25 is enough for a player to win and ends the game.

Each player plays alternatively, without the right to pass his turn. A
player's turn consists in choosing one of his non-empty pits, picking all seeds
contained in the pit and sowing them one by one in every consecutive pits on the right
(rotating counter-clockwise). The player thus has at most 6 possible moves at
each turn (one per non-empty pit owned by him).

Usually, the player that starts the game is the oldest player. In this work, we will start at random.

As an example, if we are in the initial state (showed inf Fig. `initial_board`) and the first player to move is South (on the bottom) and he chooses the 4th pit (highlighted in the figure in red), the board will then be in the  state shown in Fig. `first_move`.




  









.. raw:: html
    :file: index_files/index_8_0.svg








  
When the last sowed seed is placed in a pit owned by the opponent and after sowing
the pit contains two or three seeds, the content of the pit is captured by
the player and removed from the game. If the pit preceding the captured pit also
contains two or three seeds, it is also captured. The capture continues until a
pit without two or three seeds is encountered. When the capture is ended the
next player's turn starts.

Otherwise, when the last sowed seed is placed in a pit that, after sowing, contains one seed, more
than 3 seeds or in the current player's own pits, the turn of the player is ended without
any capture.

For example, if South plays the 4th pit in the configuration shown in Fig. `pre_capture` he will
be able to capture the opponent's 4th and 5th pits (highlighted in red in Fig. `post_capture`) 

.. todo::  "Second figure" -> utiliser numéro




  









.. raw:: html
    :file: index_files/index_10_0.svg








  









.. raw:: html
    :file: index_files/index_11_0.svg








  
If the pit chosen by the player contains more than 12 seeds, the sowing makes
more than a full revolution of the board and the starting hole is skipped during the second
and subsequent passes.

If the current player's opponent has no seed left in his half board, the
current player has to play a move that gives him seeds if such a move exists.
This rule is called the "feed your opponent".
In Fig. `feed`, South has to play the fifth pit because playing the first would leave the opponent without any move to play.




  









.. raw:: html
    :file: index_files/index_13_0.svg








  
When a player has captured more than 25 seeds the game ends and he wins. If both
players have captured 24 seeds, the game ends by a draw. If the current player
pits are all empty, the game ends and the player with the most captures wins.

The last way to stop the game is when a position is encountered twice in the
same game (there is a cycle): the game ends and the player with most captures
wins.




  
Implementation of the rules
---------------------------

We define a dataclass with the minimal attributes needed to store a state of the game.







  


  .. code:: ipython3

    from dataclasses import dataclass
    
    @dataclass
    class Game:
        pits: np.array # a 2x6 matrix containing the number of seeds in each pits
        current_player: int # 0 for North and 1 for South
        captures: np.array # the number of seeds captured by each player






  
Now that we have defined the fields our dataclass can hold to represent the state of the game,
we can inherit from it to add new methods.

The first is a static method to instantiate a game state in the initial position, with 4 seeds in each pit.




  


  .. code:: ipython3

    class Game(Game):
        ...
        
        @classmethod
        def new(klass):
            return klass(
                # A 6x2 matrix filled with 4 seeds
                pits=np.ones(6 * 2, dtype=int) * 4,
                current_player=0,
                captures=np.zeros(2, dtype=int),
            )






  
Next, we add some convenient methods that will be usefull later:




  


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






  
Now we start implementing the rules,
some of them being deliberately excluded from this implementation:

-  Loops in the game state are not checked (this considerably speeds up the computations and we never encountered a loop in practice);
-  The "feed your opponent" rule is removed; This makes the
   rules slightly simpler and we expect it does not change the complexity of the game.




  


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
            elif self.captures[0] == self.captures[1]:
                return None
            else:
                return 0 if self.captures[0] > self.captures[1] else 1






  
We can now define the ``Game.step(i)`` method that plays the
i-th pit in the current sate. This method returns the new state, the amount
of seeds captured and a boolean informing whether the game is finished.




  


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







  
We then add some display functions.




  


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
                "-" * 6 * 3
            ))
            
            pits = []
            for seeds in self.view_from_current_player:
                pits.append("{:3}".format(seeds))
            
            print("".join(reversed(pits[6:])))
            print("".join(pits[:6]))
        
        def _repr_svg_(self):
            board = np.array([
                list(reversed(self.pits[6:])),
                self.pits[:6]
            ])
            return board_to_svg(board, True)






  
We can now play a move and have its results displayed here.




  


  .. code:: ipython3

    g = Game.new()
    g, captures, done = g.step(4)
    g








.. raw:: html
    :file: index_files/index_29_0.svg








  
Formalism
---------

Now that we know the rules, we can see that Awale

* is sequential: the opponents play one after the other;
* hold no secret information: each player has the same information about
  the game;
* do not rely on randomness: the state of the game depends only on the actions
  taken sequentially by each player and an action has a deterministic result.

This type of game is called a sequential perfect information game
:cite:`osborne1994course`.

We can also see that the game is a two player zero-sum game.

Other games in this category are for example Chess, Go, Checkers or even
Tic-tac-toe and Connect Four. Sequential perfect information games are particularly interesting
in computer science and artificial intelligence as they are easy to simulate.




  
Perfect information games as finite state machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO:: formal definition of FSM ?

When viewed from an external point of view, these types of games can be
modelized as finite states machines with boards being states (the initial board
is the initial state), each player's action being a transition and wins and draws
being terminal states.

.. TODO:: formal description of the game as a FSM ?

It might be tempting to try to enumerate every possible play of those games by
starting a game and recursively trying each legal action until the end of the game
to find the best move for each state.

Unfortunately, most of the time, this is not a feasible approach due to the size
of the state space. As an example, Romein et al. claims that Awale has
889,063,398,406 legal positions :cite:`romein2003solving` and the exact number
(:math:`\approx 2.08 \times 10^{170}`) of legal positions in Go (another popular perfect information game)
is so big that it has only recently been determined :cite:`tromp2016`. Such state space are too
big to be quickly enumerated.




  
Perfect information games as Markov decision processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of being viewed from an external point of view, these types of games can
also be seen from the point of view of a single player. He only knows the state
of the board and his own moves and is not aware of the moves of his opponent,
neither in advance or after the move has been played.

When viewed under this angle, a game looks like this:
 * the game is in state :math:`&`,
 * the player plays his turn and the board changes deterministically,
 * the game is in state :math:`A'`,
 * his opponent plays and the board has multiple ways of changing,
 * the game is in state :math:`B` (one of the 6 possible successors
   of :math:`A'`).

We can model this as a Markov decision process (MDP).

.. TODO:: More on MDP and why it is a MDP.
        Vu qu'on a un MDP, on serait tenté du'iliser le framework classique de Q learning
        mais vu la taille de l'espace, on ne peut pas -> MCTS résoud bien ça
        S'inspirer de https://pdfs.semanticscholar.org/574e/6872df3fe9b89afa98a7bdeef710a931da34.pdf




  
Solved games
------------

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

:cite:`romein2003solving` claim to have solved
Awale by almost brute-force retrograde analysis. They have also published a database
of XXX. Their claim has since been challenged by Víktor Bautista i Roca in a paper published in XXX.
Bautista i Roca claims that several end states in the database are incorrect and that the proof is thus invalid.
As both the database made by Romein and the paper by Bautista i Roca are not anymore available
publicly, we cannot know who is right.

.. todo:: cite vandergoot2001 "Awari retrograde analysis" and "Games solved: Now and in the future"

The above-mentioned results for Kalah and Awale both use an almost brute-force
method to solve the game and use a database of all possible states. The database
used by :cite:`romein2003solving` has 204 billion entries and weighs 178GiB.
Such a huge database is of course not practical and  we thus think  there is still room for
improvement if we can create an agent with a policy that does not need a
exhaustive database, even if the agent is not capable of a perfect play.





  
Tree representation
-------------------

We now build a tree representation of the game where the root of the tree is the initial state and each child of a node represents one of the states created by playing one of the pits.

First, we start by adding new fields to the ``Game`` dataclass we defined earlier so that a state can hold links to its parent and children.




  


  .. code:: ipython3

    from typing import Optional, List
    from dataclasses import field
    
    @dataclass
    class TreeGame(Game):
        parent: Optional[Game] = None
        children: List[Optional[Game]] = field(default_factory=lambda: [None] * 6)






  
Next, we overload the ``Game.step(i)`` method so that we do not compute twice the same state and we keep a reference to the parent when we create a new child.




  


  .. code:: ipython3

    class TreeGame(TreeGame):
        ...
        
        def step(self, action):
            # If we already did compute the children node, just return it
            if self.children[action] is not None:
                new_game = self.children[action]
                captures = new_game.captures[self.current_player] - self.captures[self.current_player]
                return new_game, captures, new_game.game_finished
            else:
                new_game, captures, finished = super().step(action)
                new_game.parent = self
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
        
        @property
        def depth(self):
            if self.parent is None:
                return 0
            return 1 + self.parent.depth






  
Monte Carlo tree search
-----------------------

Many algorithms have been proposed and studied to play sequential
perfect information games.
A few examples are :math:`\alpha-\beta` pruning, Minimax,
Monte Carlo tree search (MCTS) and Alpha (Go) Zero :cite:`AlphaGoZero`.

We will focus on MCTS as it does not require any expert knowledge
about the given game to make reasonable decisions.

The principle of MCTS is simple : we represent the initial state of a game by
the root node of a tree. This node then has a child for each possible action
the current player can make. The n-th child of the node represents the state in
which the game would be if the player played the n-th possible action.

The maximum number of children of a node in the game is called the branching
factor. In a classical Awale game the player can choose to sow his seeds from
one of his non-empty pits. As the player has 6 pits, the branching factor is 6
(this is very small compared to the branching factor of 19 for the game of Go and
makes Awale much easier to play with MCTS).

If we build the complete tree, we compute every possible state in the game and every
leaf of the tree is a final state (end of a game). As said, previously, computing the complete tree is not
ideal for Awale (it has :math:`\approx 8 \times 10^{11}` nodes) and
computationally impossible for games with a high branching factor (unless very shallow).

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
        wins: np.array = field(default_factory=np.zeros(2, dtype=int))
        n_playouts: int = 0
        
        
        def update_stats(self, winner):
            assert winner in [0, 1]
            self.wins[winner] += 1
            self.n_playouts += 1
            if self.parent:
                self.parent.update_stats(winner)






  
Node Selection
--------------

In step 1 and 3 of the algorithm, we have to choose nodes.
There are multiples ways to choose those.

In the original MCTS we take a child at random each time.
This is easy to implement but it is not effective as it explores every part of the tree even if a part has no chance of leading to a win for the player.




  
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

In step 3, the playouts are played at random as it is the first time these nodes
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


