
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
modeled as finite states machines with boards being states (the initial board
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


Markov decision processes
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

If all transitions from a state have zero probability, the state is called a terminal state. By analogy, states that are not terminal are called non-terminal.

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







Citation:

> Surprisingly,
> increasing the bias in the random play-outs can
> occasionally weaken the strength of a program using the
> UCT algorithm even when the bias is correlated with Go
> playing strength. One instance of this was reported by Gelly
> and Silver [#GS07]_, and our group observed a drop in strength
> when the random play-outs were encouraged to form patterns
> commonly occurring in computer Go games [#Fly08]_.
