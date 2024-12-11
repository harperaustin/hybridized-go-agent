[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/3dwd8lN3)


Experimental Design:
_____________________________________________________________
My experimentation can be broke up into 3 parts:
1. I will be running the Value Agent, which is just a Greedy agent using the learned
heuristic, against all of the task 1 agents using the simple heuristic. These will just
be head to head games, 40 iterations, and I will record score and the score with black.
2. I will be running Policy Agent against all of the task 1 agents using the simple heuristic.
These will just be head to head games, 40 iterations, and I will record score and the score with black.
3. Finally, just for fun, I will run the Policy Agent against the Value Agent to see if there is any
clear advantage to either agent.

Value Agent (Greedy Agent using learned hearistic) Perfomance vs other agents (30 games):
______________________________________________________________
 - vs Random Agent: +28 score, +13 with Black
 - vs Greedy Agent: +30 score, +15 with Black
 - vs Minimax Agent (depth 2): +30 score, +15 with Black
 - vs Minimax Agent (depth 3): +30 score, +15 with Black
 - vs AlphaBeta Agent (depth 2): +30 score, +15 with Black
 - vs AlphaBeta Agent (depth 3): +30 score, +15 with Black
 - vs IDS Agent: +30 Score, +15 with Black
 - vs MCTS Agent: +30 Score, +15 with Black 


 Findings: These results show that a Greedy Agent using the learned heuristic is far
 better than any of adversarial search agents using the simple heuristic from task 1.
 The Value Agent consistently beats all of these other agents, for pretty much every game.
 I do find it interesting that the one agent that the Value Agent loses a few times against
 is the random agent. I suspect that the random agent produces scenarios the training data
 never prepared the Value Agent for, so the Value Agent doesn't really know what to do.

 Policy Agent Performance vs other agents (30 games):
 _____________________________________________________________
 - vs Random Agent: +12 score, -1 with Black
 - vs Greedy Agent: +30 score, +15 with Black
 - vs Minimax (depth 2): 0 score, -15 with Black
 - vs Mimimax (depth 3): 0 score, -15 with Black
 - vs AlphaBeta Agent (depth 2): 0 score, -15 with Black
 - vs AlphaBeta Agent (depth 3): 0 score, -15 with Black
 - vs IDS Agent: 0 score, -15 with Black
 - vs MCTS Agent: 0 score, -1 with Black


Findings: These results show that the Policy Agent is better than the random and greedy
agents from part 1, and about even with the other adversarial search agents. I am interested
to know if these results would be the same if playing on a bigger sized board. As the handout suggests,
adversarial search agents may perform better in end games, and with the board being so small, the 'end game'
is a large portion of the entire game. 


Value Agent (Greedy Agent using Learned Heuristic) vs Policy Agent (100):
______________________________________________________________
Across 100 games, both agents have a score of 0 and a score of -50 with Black.

Findings: This indicates that the Value Agent and the Policy Agent are about even in terms of
'skill level.'



