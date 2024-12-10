import random
from go_search_problem import GoProblem, GoState, Action

from typing import Dict, Tuple
import time
from adversarial_search_problem import (
    Action,
    State as GameState,
)
from heuristic_go_problems import GoProblemSimpleHeuristic


def minimax(asp: GoProblemSimpleHeuristic, game_state: GoState, start_time, time_limit, cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and zero-sum.



    Input:
        asp - a HeuristicAdversarialSearchProblem
        cutoff_depth - the maximum search depth, where 0 is the start state. 
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
        a dictionary of statistics for visualization
            states_expanded: stores the number of states expanded during current search
                            A state is expanded when get_available_actions(state) is called.

    """


    def val(state: GameState, problem: GoProblemSimpleHeuristic, depth):
        # Base cases:
        # if the problem is terminal, return the reward
        if problem.is_terminal_state(state):
            return (problem.evaluate_terminal(state), None)
        # if reached cutoff depth, return the heuristic value
        elif depth == cutoff_depth:
            return (problem.heuristic(state, state.player_to_move()), None)
        
        else:
            # MAX TURN
            stats['states_expanded'] += 1
            if state.player_to_move() == 0:
                return max_val(state, problem, depth)
            else:
                # MIN TURN
                return min_val(state, problem, depth)

    def max_val(state: GameState, problem: GoProblemSimpleHeuristic, depth):
        # start with a really low value
        max_v = float('-inf')
        best_action = None
        # look at every possible action from the current state
        if time.time() - start_time >= time_limit:
            return (max_v, best_action)

        for action in problem.get_available_actions(state):
            next_state = problem.transition(state, action)
            # recursively call val on every next state
            curr_v = val(next_state, problem, depth + 1)[0]
            # if this val is greater than the max val, then set the new value and action
            # as the current best
            if  curr_v >= max_v:
                max_v = curr_v
                best_action = action
        # return the max v and the best action
        return (max_v, best_action)
    
    def min_val(state: GameState, problem: GoProblemSimpleHeuristic, depth):
        # start with a really high value
        min_v = float('inf')
        best_action = None
        # look at every possible action from the current state
        for action in problem.get_available_actions(state):
            next_state = problem.transition(state, action)
            # recursively call val on every next state
            curr_v = val(next_state, problem, depth + 1)[0]
            if  curr_v <= min_v:
                # if this val is less than the min val, then set the new value and action
                # as the current best
                min_v = curr_v
                best_action = action
        # return the min v and the best action
        return (min_v, best_action)


    best_action = None
    stats = {
        'states_expanded': 0
    }

    _,best_action = val(game_state, asp, 0)


    return best_action, stats






def alpha_beta(asp: GoProblemSimpleHeuristic, game_state: GoState, start_time, time_limit, cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
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
    def val_ab(state: GameState, problem: GoProblemSimpleHeuristic, depth, alpha, beta):
        # Base cases:
        # if the problem is terminal, return the reward
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
        if time.time() - start_time >= time_limit:
            return (max_v, best_action)
        for action in problem.get_available_actions(state):
            next_state = problem.transition(state, action)
            # recursively call val on every next state
            curr_v = val_ab(next_state, problem, depth + 1, alpha, beta)[0]
            # if this val is greater than the max val, then set the new value and action
            # as the current best
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
        if time.time() - start_time >= time_limit:
            return (min_v, best_action)
        # look at every possible action from the current state
        for action in problem.get_available_actions(state):
            next_state = problem.transition(state, action)
            # recursively call val on every next state
            curr_v = val_ab(next_state, problem, depth + 1, alpha, beta)[0]
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

    _,best_action = val_ab(game_state, asp, 0, float('-inf'), float('inf'))
    return best_action, stats