from go_search_problem import GoProblem, GoState
BLACK = 0
WHITE = 1

class GoProblemSimpleHeuristic(GoProblem):
    def __init__(self, state=None):
        super().__init__(state=state)

    def heuristic(self, state, player_index):
        """
        Very simple heuristic that just compares the number of pieces for each player
        
        Having more pieces (>1) than the opponent means that some were captured, capturing is generally good.
        """
        return len(state.get_pieces_coordinates(BLACK)) - len(state.get_pieces_coordinates(WHITE))

    def __str__(self) -> str:
        return "Simple Heuristic"


class GoProblemLearnedHeuristic(GoProblem):
    def __init__(self, model=None, state=None,):
        super().__init__(state=state)
        self.model = model

    def encoding(self, state):
        pass

    def heuristic(self, state, player_index):
        pass

    def __str__(self) -> str:
        return "Learned Heuristic"
    
class GoProblemMerpHeuristic(GoProblem):
    def __init__(self, model=None, state=None,):
        super().__init__(state=state)
        self.model = model

    def encoding(self, state):
        pass

    def heuristic(self, state, player_index):
        black_territory, white_territory = 0, 0
        visited = set()
        board = state.get_board()


        # This gets the size of connected regions of the board
        def flood_fill(x,y, state, color):
            stack = [(x,y)]
            region = []
            liberties = set()
            curr_player_pieces = state.get_pieces_array(color)
           
            empty_spaces = state.get_board()[2]

            while stack:
                curr_x, curr_y = stack.pop()
                if (curr_x, curr_y) in visited:
                    continue
                visited.add((curr_x,curr_y))
                region.append((curr_x, curr_y))

                for neighbor_x, neighbor_y in neighbors(curr_x, curr_y):
                    if curr_player_pieces[neighbor_x][neighbor_y] == 1:
                        stack.append((neighbor_x, neighbor_y))
                    elif empty_spaces[neighbor_x][neighbor_y] == 1:
                        liberties.add((neighbor_x, neighbor_y))
            
            return region, liberties

        
        def neighbors(x, y):
            # get all connecting game states that are in the board
            ns = [(x - 1, y), (x + 1, y), (x, y - 1), (x , y + 1)]
            return [(x, y) for x, y in ns if 0 <= x < 5 and 0 <= y < 5]
        

        # loop through every slot in the board
        for x in range(5):
            for y in range(5):
                # only visit each slot once
                if (x,y) not in visited:
                    if board[BLACK][x][y] == 1:
                        region, liberties = flood_fill(x,y,state, BLACK)
                        black_territory += len(liberties)
                    elif board[WHITE][x][y] == 1:
                        region, liberties = flood_fill(x,y,state, WHITE)
                        white_territory += len(liberties)

        return black_territory - white_territory


    def __str__(self) -> str:
        return "Merp Heuristic"
