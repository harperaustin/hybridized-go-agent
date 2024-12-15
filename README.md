[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/3dwd8lN3)


My approach:
My 5x5 Go bot is a hybrid agent consisting of an opening book for the start of the game, a value agent with improved 
encodings and NN architecture for playing the mid-game/majority of the game, and an alphabeta agent with a new, more informative
heuristic for the end game.

Opening Book:
I hardcoded moves for up to the first 3 moves, as there are commonly known 'good' openings for the 5x5
game of Go. If going first, always play in the middle. For the second/third move, I used a dictionary
to map game_states to actions.

Value Agent:
Moving on from part 2, I wanted to create an improved value agent by providing better encodings for each
game state and changing the architecture of my neural network that learns the values. My encodings now add 
features with information for each slot on the board, stating if their are open neighboring slots, or occupied
by a friendly piece, or occupied by an enemy piece. By playing around with the architecture and parameters for 
the NN, I was able to find a set up that performed better than my previous value agent from part 2.

AlphaBeta Agent with new heuristic:


Putting it all together:






