# Go Bot: Hybrid Agent for 5x5 Go

## Task Overview

Final project for CSCI 0410. The goal of this project was to develop an intelligent agent capable of playing the game of Go on a 5x5 board. The challenge was to design a competitive bot that can effectively handle all phases game. This involved integrating multiple agent strategies (value-based learning, search-based planning, and rule-based heuristics), tuning transitions between them, and optimizing overall performance against known baselines. The final bot was evaluated based on its ability to consistently outperform existing agents.

---

## My Approach

My 5x5 Go bot is a hybrid agent composed of:

- **Opening Book**: Handles the first few moves using a handcrafted strategy.
- **Value Agent**: A neural network-based agent with improved state encodings and architecture for the mid-game.
- **AlphaBeta Agent**: Uses a new heuristic that evaluates territory and liberties to make endgame decisions.

---

### Opening Book

I hardcoded moves for the first three turns, based on known good openings in 5x5 Go:

- If going first: always play in the center.
- For the second and third moves, I used a dictionary to map `game_state → action` for optimal early play.

---

### Value Agent

To improve upon the original value agent:

- I enhanced the **state encoding** to include features for each slot:
  - Whether neighboring slots are open
  - Whether they are occupied by friendly or enemy pieces
- I experimented with various **neural network architectures** and hyperparameters until I found a configuration that consistently outperformed the earlier implementation.

---

### AlphaBeta Agent with New Heuristic

To strengthen endgame performance:

- I switch from the value agent to AlphaBeta when **more than 17 pieces** are on the board.
- I developed a new heuristic that uses a **flood-fill algorithm** to assess:
  - Connected territories
  - Liberties (open adjacent spaces)
- This heuristic better represents game dynamics than a simple piece count, improving AlphaBeta’s effectiveness.

---

## My Process

- I followed an **iterative testing process**: tweaking values, changing features, and comparing results to previous versions.
- I evaluated hybrid configurations (e.g., MCTS mid-game with AlphaBeta endgame), but my final combination showed the most consistent performance.
- I systematically determined transition points between agents (e.g., switching to AlphaBeta at 17+ pieces) by empirical testing.
- The hybrid architecture allowed me to **easily swap in and out different agents**, enabling rapid experimentation and refinement.
