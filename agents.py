from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time
from adversarial_search import minimax, alpha_beta
import pickle
import torch
from torch import nn
from typing import Dict, Tuple


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
        start_time = time.time()
        best_action, _ = minimax(self.search_problem, game_state, start_time, time_limit, cutoff_depth=self.depth)
        return best_action

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
        start_time = time.time()
        best_action, _ = alpha_beta(self.search_problem, game_state, start_time, time_limit, cutoff_depth=self.depth)
        return best_action

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)
    
class AlphaBetaAgentDiff(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemMerpHeuristic()):
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
        

        def alpha_beta_diff(asp: self.search_problem, game_state: GoState, start_time, time_limit, cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
            """
            Implement the alpha-beta pruning algorithm on ASPs,
            assuming that the given game is both 2-player and constant-sum.

            Input:
                asp - an AdversarialSearchProblem
                cutoff_depth - the maximum search depth, where 0 is the start state,
                            Depth 1 is all the states reached after a single action from the start state (1 ply).
                            cutoff_depth will always be greater than 0.
            Output:
                an action (an element of asp.get_available_actions(asp.get_start_state()))
                a dictionary of statistics for visualization
                    states_expanded: stores the number of states expanded during current search
                                    A state is expanded when get_available_actions(state) is called.
            """
            def val_ab(state: GameState, problem: self.search_problem, depth, alpha, beta):
                # Base cases:
                # if the problem is terminal, return the reward
                if problem.is_terminal_state(state):
                    return (problem.evaluate_terminal(state), None)
                # if reached cutoff depth, return the heuristic value
                elif depth == cutoff_depth:
                    if problem.heuristic(state, state.player_to_move()) == None:
                        print("heuristic is returning none.\n")
                    return (problem.heuristic(state, state.player_to_move()), None)
                
                else:
                    # MAX TURN
                    stats['states_expanded'] += 1
                    if state.player_to_move() == 0:
                        return max_val_ab(state, problem, depth, alpha, beta)
                    else:
                        # MIN TURN
                        return min_val_ab(state, problem, depth, alpha, beta)

            def max_val_ab(state: GameState, problem: self.search_problem, depth, alpha, beta):
                # start with a really low value
                max_v = float('-inf')
                best_action = None
                # look at every possible action from the current state
                if time.time() - start_time >= time_limit:
                    return (max_v, best_action)
                for action in problem.get_available_actions(state):
                    next_state = problem.transition(state, action)
                    # recursively call val on every next state
                    curr_v = val_ab(next_state, problem, depth + 1, alpha, beta)[0]
                    # if this val is greater than the max val, then set the new value and action
                    # as the current best
                    if curr_v >= max_v:
                        max_v = curr_v
                        best_action = action

                    alpha = max(alpha, max_v)
                    # checking if in this branch the minimizer already has an option
                    # that is lower than the current alpha. If so, we can prune, because
                    # the maximizer will only choose higher values
                    if beta <= alpha:
                        break
                # return the max v and the best action
                return (max_v, best_action)
            
            def min_val_ab(state: GameState, problem: self.search_problem, depth, alpha, beta):
                # start with a really high value
                min_v = float('inf')
                best_action = None
                if time.time() - start_time >= time_limit:
                    return (min_v, best_action)
                # look at every possible action from the current state
                for action in problem.get_available_actions(state):
                    next_state = problem.transition(state, action)
                    # recursively call val on every next state
                    curr_v = val_ab(next_state, problem, depth + 1, alpha, beta)[0]
                    if curr_v <= min_v:
                        # if this val is less than the min val, then set the new value and action
                        # as the current best
                        min_v = curr_v
                        best_action = action
                    beta = min(beta, min_v)
                        # checking if in this branch the maximizer already has an option
                        # that is higher than the current beta. If so, we can prune, because
                        # the minimizer will only choose lower values.
                    if beta <= alpha:
                        break
                # return the min v and the best action
                return (min_v, best_action)
            


            stats = {
                'states_expanded': 0  # Increase by 1 for every state transition
            }

            _,best_action = val_ab(game_state, asp, 0, float('-inf'), float('inf'))
            return best_action, stats
            

        start_time = time.time()
        best_action, _ = alpha_beta_diff(self.search_problem, game_state, start_time, time_limit, cutoff_depth=self.depth)
        return best_action


    def __str__(self):
        return f"AlphaBeta Diff w/ depth {self.depth} + " + str(self.search_problem)
    




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
        running = True
        cutoff_depth = 0
        start_time = time.time()
        best_action = None
        time_remaining = min(self.cutoff_time, time_limit)
        while running:
            # check to see if time limit has been reached
            elapsed_time = time.time() - start_time
            # if so, store the current best action
            if elapsed_time >= time_remaining:
                return best_action
            # otherwise increase depth
            cutoff_depth += 1
            self.search_problem.set_start_state(game_state)
            # run alpha beta with greater depth
            action = self.alpha_beta_ids(self.search_problem, start_time, self.cutoff_time - 0.05, cutoff_depth=cutoff_depth)
            # alpha beta will return none if the time limit is reached
            if action is not None:
                best_action = action

    def alpha_beta_ids(self, asp: GoProblemSimpleHeuristic, start_time, time_limit, cutoff_depth=float('inf')) -> Action:
        """
        Implement the alpha-beta pruning algorithm on ASPs,
        assuming that the given game is both 2-player and constant-sum.

        Input:
            asp - an AdversarialSearchProblem
            cutoff_depth - the maximum search depth, where 0 is the start state,
                        Depth 1 is all the states reached after a single action from the start state (1 ply).
                        cutoff_depth will always be greater than 0.
        Output:
            an action (an element of asp.get_available_actions(asp.get_start_state()))
            a dictionary of statistics for visualization
                states_expanded: stores the number of states expanded during current search
                                A state is expanded when get_available_actions(state) is called.
        """
        timeout = False
        def val_ab(state: GameState, problem: GoProblemSimpleHeuristic, depth, alpha, beta):
            # Base cases:
            # if the problem is terminal, return the reward
            nonlocal timeout
            if time.time() - start_time >= time_limit:
                timeout = True
                return None, None
            if problem.is_terminal_state(state):
                return (problem.evaluate_terminal(state), None)
            # if reached cutoff depth, return the heuristic value
            elif depth == cutoff_depth:
                return (problem.heuristic(state, state.player_to_move()), None)

            
            else:
                # MAX TURN
                stats['states_expanded'] += 1
                if state.player_to_move() == 0:
                    return max_val_ab(state, problem, depth, alpha, beta)
                else:
                    # MIN TURN
                    return min_val_ab(state, problem, depth, alpha, beta)

        def max_val_ab(state: GameState, problem: GoProblemSimpleHeuristic, depth, alpha, beta):
            # start with a really low value
            max_v = float('-inf')
            best_action = None
            # look at every possible action from the current state
            for action in problem.get_available_actions(state):
                next_state = problem.transition(state, action)
                # recursively call val on every next state
                curr_v = val_ab(next_state, problem, depth + 1, alpha, beta)[0]
                # if this val is greater than the max val, then set the new value and action
                # as the current best
                if timeout:
                    return (max_v, best_action)
                if  curr_v >= max_v:
                    max_v = curr_v
                    best_action = action

                alpha = max(alpha, max_v)
                # checking if in this branch the minimizer already has an option
                # that is lower than the current alpha. If so, we can prune, because
                # the maximizer will only choose higher values
                if beta <= alpha:
                    break
            # return the max v and the best action
            return (max_v, best_action)
        
        def min_val_ab(state: GameState, problem: GoProblemSimpleHeuristic, depth, alpha, beta):
            # start with a really high value
            min_v = float('inf')
            best_action = None
            # look at every possible action from the current state
            for action in problem.get_available_actions(state):
                next_state = problem.transition(state, action)
                # recursively call val on every next state
                curr_v = val_ab(next_state, problem, depth + 1, alpha, beta)[0]
                if timeout:
                    return (min_v, best_action)
                if  curr_v <= min_v:
                    # if this val is less than the min val, then set the new value and action
                    # as the current best
                    min_v = curr_v
                    best_action = action
                    beta = min(beta, min_v)
                    # checking if in this branch the maximizer already has an option
                    # that is higher than the current beta. If so, we can prune, because
                    # the minimizer will only choose lower values.
                    if beta <= alpha:
                        break
            # return the min v and the best action
            return (min_v, best_action)
        stats = {
            'states_expanded': 0  # Increase by 1 for every state transition
        }

        _,best_action = val_ab(asp.get_start_state(), asp, 0, float('-inf'), float('inf'))
        return best_action  

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
        start_time = time.time()
        node = MCTSNode(game_state)
        while time.time() - start_time  < time_limit - 0.1:
            # select a node using policy
            leaf = self.select(node)
            # expand that nodes and get children
            children = self.expand(leaf)
            # simulate results using uniform random rollout policy
            results = self.simulate(children)
            # backprop to update results
            self.backpropogate(results, children)
        # best_child = max(node.children, key=lambda child: child.visits)
        most_visits = 0
        best_child = None
        for child in node.children:
            if child.visits >= most_visits:
                most_visits = child.visits
                best_child = child
        return best_child.action

    def upper_confidence_bound(self, node: MCTSNode):
        """
        Calculates the upper confidence bound for a given node
        """
        # if the node hasn't been visited, give it high score so we do visit it
        if node.visits == 0 or node.parent is None:
            return float('inf')
        # compute components of ucb calculation
        average_reward = node.value / node.visits

        exploration_term = self.c * np.sqrt(np.log(node.parent.visits / node.visits))
        # exploration_term = 2 * np.sqrt(np.log(node.parent.visits / node.visits))
        return average_reward + exploration_term
        
    def select(self, node: MCTSNode):
        """
        Selects a node using a specified policy
        """
        # use upper confidence bound
        current_node = node
        # loop until we found a leaf node
        # try other check with terminal state
        while len(current_node.children) > 0:

            # use upper confidence bound as policy to find the child to visit
            max_uct = -float('inf')
            best_child = None
            # choose the child with the highest uct value
            for child in current_node.children:
                uct = self.upper_confidence_bound(child)
                if uct >= max_uct:
                    max_uct = uct
                    best_child = child
            current_node = best_child

        return current_node
    
    def expand(self, leaf : MCTSNode):
        """
        Expands a node and returns it a list of its children
        """
        children = []
        state = leaf.state
        
        # get all available actions
        actions = self.search_problem.get_available_actions(state)
        
        # transition through all actions
        for action in actions:
            new_child = MCTSNode(self.search_problem.transition(state, action), leaf, None, action)
            children.append(new_child)
            leaf.children.append(new_child)
        return children
    
    def rollout(self, node: MCTSNode):
        """
        Completes a single rollout for a node using uniform random policy
        """
        state = node.state
        while not self.search_problem.is_terminal_state(state):
            
            actions = self.search_problem.get_available_actions(state)
            # choose a random action
            random_action = np.random.choice(actions)
            # transition using that action
            state = self.search_problem.transition(state, random_action)
        # evaluate the result once we hit a terminal state
        result = self.search_problem.evaluate_terminal(state)
        return result

    
    def simulate(self, children):
        """
        Simulate a rollout on all children of a given node
        """
        results = []
        for child in children:
            result = self.rollout(child)
            results.append(result)
        return results
    
    def backpropogate(self, results, children):
        """
        Update values of nodes and visits
        """
        for child, result in zip(children, results):
            curr_node = child
            while curr_node is not None:
                
                curr_node.visits += 1
                # if black wins and the current player is black
                if result > 0 and curr_node.state.player_to_move() == 1:
                    curr_node.value += 1
                # if white wins and the current player is white
                elif result < 0 and curr_node.state.player_to_move() == 0:
                    curr_node.value += 1
                curr_node = curr_node.parent

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

      # output size should be 1, as we are predicting a value in between [-1,1]
      output_size = 1

      # add more layers
      self.layer1 = nn.Linear(input_size, 64)
      self.layer2 = nn.Linear(64, 32)
      self.layer3 = nn.Linear(32, 10)
      self.layer4 = nn.Linear(10, output_size)
      
      self.tanh = nn.Tanh()
      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      z1 = self.layer1(x)
      a1 = torch.relu(z1)
      z2 = self.layer2(a1)
      a2 = self.sigmoid(z2)
      z3 = self.layer3(a2)
      a3 = torch.relu(z3)
      z4 = self.layer4(a3)
      output = self.sigmoid(z4)
      return output
    

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
        return get_features(state)

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state

        Input:
            state: GoState to encode into a fixed size list of features
            player_index: index of player to evaluate heuristic for
        Output:
            value: heuristic (value) of current state
        """
        value = 0
        # get encoding for the state:
        state_encoding = self.encoding(state)
        features_tensor = torch.Tensor(state_encoding)
        value = self.model(features_tensor)
        # create heuristic value based on this state
        # use return value you get from value Network
        
        # Note, your agent may perform better if you force it not to pass
        # (i.e., don't select action #25 on a 5x5 board unless necessary)
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"
    
def create_value_agent_from_model():
    model_path = "value_model.pt"
    feature_size = 53
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)
    learned_agent = GreedyAgent(heuristic_search_problem)
    return learned_agent

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, board_size=5):
      super(PolicyNetwork, self).__init__()

      output_size = (board_size * board_size) + 1

      self.layer1 = nn.Linear(input_size, 64)
      self.layer2 = nn.Linear(64, 32)
      self.layer3 = nn.Linear(32, 32)
      self.layer4 = nn.Linear(32, output_size)
      self.tanh = nn.Tanh()
      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      z1 = self.layer1(x)
      a1 = self.tanh(z1)
      z2 = self.layer2(a1)
      a2 = torch.relu(z2)
      z3 = self.layer3(a2)
      a3 = torch.relu(z3)
      z4 = self.layer4(a3)
      output = self.sigmoid(z4)
      return output
  
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
    features = []
    # for first 25 features, use just a 1 or 0 to indicate if a black piece is in the slot
    black_player_pieces = game_state.get_pieces_array(0)
    black_total_pieces = sum(sum(black_player_pieces))
    for row in black_player_pieces:
        for piece in row:
            features.append(piece)
    # for second 25 features, use just a 1 or 0 to indicate if a white piece is in the slot
    white_player_pieces = game_state.get_pieces_array(1)
    white_total_pieces = sum(sum(white_player_pieces))
    for row in white_player_pieces:
        for piece in row:
            features.append(piece)
            
    # add total number of pieces
    features.append(black_total_pieces)
    features.append(white_total_pieces)
    # finally append the player to move
    features.append(game_state.player_to_move())


    # the solution might just be calling getboard() and flattening, look into this:

    return features

class PolicyAgent(GameAgent):
    def __init__(self, search_problem, model_path, board_size=5):
        super().__init__()
        self.search_problem = search_problem
        self.model = load_model(model_path, PolicyNetwork(53, 5))
        self.board_size = board_size

    def encoding(self, state):
        return get_features(state)

    def get_move(self, game_state, time_limit=1):
      """
      Get best action for current state using self.model

      Input:
        game_state: current state of the game
        time_limit: time limit for search (This won't be used in this agent)
      Output:
        action: best action to take
      """

      features_tensor = torch.tensor(self.encoding(game_state), dtype=torch.float32)
      model_output = self.model(features_tensor)
      legal_actions = self.search_problem.get_available_actions(game_state)
      #print(legal_actions)
      for i in range(len(model_output)):
          if i not in legal_actions:
              model_output[i] = float('-inf')
      # get best legal action
      action = torch.argmax(model_output).item()
      # if the selected action is to pass, but there are other options, choose other options
      if action == 25 and len(legal_actions) > 1:
          legal_actions.remove(25)
          action = legal_actions[torch.argmax(model_output[legal_actions])]
      #print(action)
      # Note, you may want to force your policy not to pass their turn unless necessary
      assert action in self.search_problem.get_available_actions(game_state)
      
      return action

    def __str__(self) -> str:
        return "Policy Agent"
    

def create_policy_agent_from_model():
    """
    Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.    
    """

    model_path = "policy_model.pt"
    agent = PolicyAgent(GoProblem(size=5), model_path)
    return agent

class CustomNetwork(nn.Module):
    def __init__(self, input_size):
      super(CustomNetwork, self).__init__()

      # output size should be 1, as we are predicting a value in between [-1,1]
      output_size = 1

      # add more layers
      self.layer1 = nn.Linear(input_size,64)
      self.layer2 = nn.Linear(64, 32)
      self.layer3 = nn.Linear(32, 10)
      self.layer4 = nn.Linear(10, output_size)
      # self.layer5 = nn.Linear(16,8)
      # self.layer6 = nn.Linear(8,output_size)
      
      # self.dropout = nn.Dropout(0.2)
      self.tanh = nn.Tanh()
      self.sigmoid = nn.Sigmoid()


    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      # relu, tanh, relu, sigmoid
      z1 = self.layer1(x)
      a1 = torch.relu(z1)


      z2 = self.layer2(a1)
      a2 = torch.tanh(z2)


      z3 = self.layer3(a2)
      a3 = torch.relu(z3)


      z4 = self.layer4(a3)



      output = self.sigmoid(z4)
      return output
    
class GoProblemCustomHeuristic(GoProblem):
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
        return get_features_advanced(state)

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state

        Input:
            state: GoState to encode into a fixed size list of features
            player_index: index of player to evaluate heuristic for
        Output:
            value: heuristic (value) of current state
        """
        # MAY NEED AN IF STATEMENT HERE, like if end game heuristic is alpha beta, 
        value = 0
        # get encoding for the state:
        state_encoding = self.encoding(state)
        features_tensor = torch.Tensor(state_encoding)
        value = self.model(features_tensor)
        # create heuristic value based on this state
        # use return value you get from value Network
        
        # Note, your agent may perform better if you force it not to pass
        # (i.e., don't select action #25 on a 5x5 board unless necessary)
        return value
    
    def get_move(self, game_state, time_limit=1):
      """
      Get best action for current state using self.model

      Input:
        game_state: current state of the game
        time_limit: time limit for search (This won't be used in this agent)
      Output:
        action: best action to take
      """

      # if we are at opening book place, then use opening book aciton
      # if mid game, use learned value function
      # if we are end game, use alpha beta with better heuristic
      return None
    


    def __str__(self) -> str:
        return "Merp Agent"

def create_custom_agent_from_model():
    model_path = "custom_model.pt"
    feature_size = 103
    model = load_model(model_path, CustomNetwork(feature_size))
    heuristic_search_problem = GoProblemCustomHeuristic(model)
    learned_agent = GreedyAgent(heuristic_search_problem)
    return learned_agent
    
def get_features_advanced(game_state: GoState):
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
    features = []
    # for first 25 features, use just a 1 or 0 to indicate if a black piece is in the slot
    black_player_pieces = game_state.get_pieces_array(0)
    black_total_pieces = sum(sum(black_player_pieces))
    for row in black_player_pieces:
        for piece in row:
            features.append(piece)
    # for second 25 features, use just a 1 or 0 to indicate if a white piece is in the slot
    white_player_pieces = game_state.get_pieces_array(1)
    white_total_pieces = sum(sum(white_player_pieces))
    for row in white_player_pieces:
        for piece in row:
            features.append(piece)
            
    # add total number of pieces
    features.append(black_total_pieces)
    features.append(white_total_pieces)
    # finally append the player to move
    features.append(game_state.player_to_move())

    # loop through all of the black pieces
    for i in range(len(black_player_pieces)):
        for j in range(len(black_player_pieces[0])):
            # initialize a value for the adjacent piece being either same color, opponent, or empty
            # need to check for every piece regardless 
            adjacent_black_count = 0
            adjacent_white_count = 0
            adjacent_empty_count = 0

            # check below current piece
            new_i = i + 1
            if new_i >= 0 and new_i < 5:
                adjacent_black_count += black_player_pieces[new_i][j]
                adjacent_white_count += white_player_pieces[new_i][j]
                if black_player_pieces[new_i][j] == 0 and white_player_pieces[new_i][j] == 0:
                    adjacent_empty_count += 1

            # check above the current piece
            new_i = i - 1
            if new_i >= 0 and new_i < 5:
                adjacent_black_count += black_player_pieces[new_i][j]
                adjacent_white_count += white_player_pieces[new_i][j]
                if black_player_pieces[new_i][j] == 0 and white_player_pieces[new_i][j] == 0:
                    adjacent_empty_count += 1

             # check to the right of the current piece
            new_j = j + 1
            if new_j >= 0 and new_j < 5:
                adjacent_black_count += black_player_pieces[i][new_j]
                adjacent_white_count += white_player_pieces[i][new_j]
                if black_player_pieces[i][new_j] == 0 and white_player_pieces[i][new_j] == 0:
                    adjacent_empty_count += 1

             # check to the left of current piece
            new_j = j - 1
            if new_j >= 0 and new_j < 5:
                adjacent_black_count += black_player_pieces[i][new_j]
                adjacent_white_count += white_player_pieces[i][new_j]
                if black_player_pieces[i][new_j] == 0 and white_player_pieces[i][new_j] == 0:
                    adjacent_empty_count += 1
            # for each tile, add a feature indicating how many adjacent pieces are white and another for how many are black and another for empty
            # (divide by 4 to normalize the value, as we are checking 4 total adjacent squares)
            features.append(adjacent_black_count/4)
            features.append(adjacent_white_count/4)
            #features.append(adjacent_empty_count/4)


    return features

    
    

def main():
    from game_runner import run_many
    agent1 = AlphaBetaAgentDiff(depth=2)
    agent2 = create_value_agent_from_model()
    
    # Play 10 games
    run_many(agent1, agent2, 40)


if __name__ == "__main__":
    main()
