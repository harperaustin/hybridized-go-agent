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
- Since Alpha-Beta is guarenteed to find the best moves possible according to the provided heuristic, it would do 
well at the end of the game, when there are not many available actions, so time isn't a worry. So, my approach
is to see how many pieces are on the board, and if there are more than 17, then to switch from the value agent
to the AB agent. 
- It was important that I developed a heuristic that was a more accurate representation of the game state than the
simple heuristic, which merely counted the number of pieces. In the game of Go, the positioning of these pieces and relative
positioning according to other pieces is very important. So, for this heuristic, I used a flood-fill algorithm, to calculate a
heuristic value that takes into account the number of connected pieces, so the player's territory, and the number of liberties,
or free spaces, next to each player's territory. This produces a value that is more accurate to the state of the game, and therefore
allows the alpha-beta to perform much better.

My Process:
- My process of developing this bot consisted of a lot of testing different values/parameters/approaches, comparing them to previous
attemmpts, and building upon the most promising one. With the value agent, I had to test and play around with the feature encoding and
NN architecture many times to get it to performm as good, and eventually better, than the original value agent. Now, from this point,
it was about finding when the best point to switch to the alpha-beta agent. This just took experimenting with different values until
I settled on 17, which seemed to have the best results. I took a similar approach to finding when to stop using the opening book. I found that
only doing the first 2 moves was the best for my case. 

- I had set up my hybrid agent so that I could easily switch between what agent was playing mid-game and late-game. Because of this, I was able to 
test out MCTS as the mid-game agent, or the policy agent as the mid-game agent, etc. I would simply experiment with these configurations by putting them
against a value agent or alpha beta as a general guideline, but my final implementation seemed to prove most effective. My considerations were primarily
about the score that the agent would recieve against the opponent. 







