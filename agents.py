from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time
from game_runner import run_many
import pickle
import torch
from torch import nn


MAXIMIZER = 0
MIMIZER = 1

class GameAgent():
    # Interface for Game agents
    @abstractmethod
    def get_move(self, game_state: GameState, time_limit: float) -> Action:
        # Given a state and time limit, return an action
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)

        # Compare heuristic of every reachable next state
        for action in actions:
            new_state = search_problem.transition(game_state, action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)


class MinimaxAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move method of MinimaxAgent
        pass

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)


class AlphaBetaAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move algorithm of AlphaBeta Agent
        pass

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)


class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.

        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move algorithm of IterativeDeepeningAgent
        pass

    def __str__(self):
        return f"IterativeDeepneing + " + str(self.search_problem)


class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (number of times simulations from children results in black win)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)


class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c

        # Initialize Search problem
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm
        
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: Implement MCTS
        pass

    def __str__(self):
        return "MCTS"

def load_model(path: str, model):
    """
    Load model from file

    Note: you still need to provide a model (with the same architecture as the saved model))

    Input:
        path: path to load model from
        model: Pytorch model to load
    Output:
        model: Pytorch model loaded from file
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
      super(ValueNetwork, self).__init__()

      # TODO: What should the output size of a Value function be?
      output_size = 0

      # TODO: Add more layers, non-linear functions, etc.
      self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      # TODO: Update as more layers are added
      return self.linear(x)

class GoProblemLearnedHeuristic(GoProblem):
    def __init__(self, model=None, state=None):
        super().__init__(state=state)
        self.model = model
        
    def __call__(self, model=None):
        """
        Use the model to compute a heuristic value for a given state.
        """
        return self

    def encoding(self, state):
        """
        Get encoding of state (convert state to features)
        Note, this may call get_features() from Task 1. 

        Input:
            state: GoState to encode into a fixed size list of features
        Output:
            features: list of features
        """
        # TODO: get encoding of state (convert state to features)
        features = []

        return features

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state

        Input:
            state: GoState to encode into a fixed size list of features
            player_index: index of player to evaluate heuristic for
        Output:
            value: heuristic (value) of current state
        """
        # TODO: Compute heuristic (value) of current state
        value = 0


        # Note, your agent may perform better if you force it not to pass
        # (i.e., don't select action #25 on a 5x5 board unless necessary)
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, board_size=5):
      super(PolicyNetwork, self).__init__()

      # TODO: What should the output size of the Policy be?
      output_size = 0

      # TODO: Add more layers, non-linear functions, etc.
      self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
      # TODO: Update as more layers are added
      return self.linear(x)
  
def get_features(game_state: GoState):
    """
    Map a game state to a list of features.

    Some useful functions from game_state include:
        game_state.size: size of the board
        get_pieces_coordinates(player_index): get coordinates of all pieces of a player (0 or 1)
        get_pieces_array(player_index): get a 2D array of pieces of a player (0 or 1)
        
        get_board(): get a 2D array of the board with 4 channels (player 0, player 1, empty, and player to move). 4 channels means the array will be of size 4 x n x n
    
        Descriptions of these methods can be found in the GoState

    Input:
        game_state: GoState to encode into a fixed size list of features
    Output:
        features: list of features
    """
    board_size = game_state.size
    
    # TODO: Encode game_state into a list of features
    features = []

    return features

class PolicyAgent(GameAgent):
    def __init__(self, search_problem, model_path, board_size=5):
        super().__init__()
        self.search_problem = search_problem
        self.model = load_model(model_path, PolicyNetwork)
        self.board_size = board_size

    def encoding(self, state):
        # TODO: get encoding of state (convert state to features)
        features = []

        return features

    def get_move(self, game_state, time_limit=1):
      """
      Get best action for current state using self.model

      Input:
        game_state: current state of the game
        time_limit: time limit for search (This won't be used in this agent)
      Output:
        action: best action to take
      """

      # TODO: Select LEGAL Best Action predicted by model
      # The top prediction of your model may not be a legal move!
      action = random.choice(self.search_problem.get_available_actions(game_state))

      # Note, you may want to force your policy not to pass their turn unless necessary
      assert action in self.search_problem.get_available_actions(game_state)
      
      return action

    def __str__(self) -> str:
        return "Policy Agent"

def main():
    agent1 = GreedyAgent()
    agent2 = GreedyAgent()
    # Play 10 games
    run_many(agent1, agent2, 10)


if __name__ == "__main__":
    main()
