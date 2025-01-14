from custom_dataclass import Move, MoveType, MoveAction, CardInfo, FoundationState, TableauState, CardLocation
from config import CARD_CONFIG
from screenshot import capture_new_screenshot
from typing import List, Optional, Tuple
from copy import deepcopy

class SolitaireSolver:
    """
    Class to represent the Solitaire game solver, which computes the optimal moves based on the current game state. 

    The solver utilizes the player's actions, image processing, and the current game state to determine the best strategies 
    for solving the Solitaire game. The solver keeps track of moves made, the cards known to be in the game, and the state 
    of the game (visible and hidden cards).
    """

    def __init__(self, game_state, image_processor, player, screen_region):
        """
        Initializes the SolitaireSolver with the provided game state, image processor, player, and screen region.

        Args:
            game_state (GameState): The initial state of the Solitaire game, containing information such as tableau,
                                     foundations, and visible cards.
            image_processor (ImageProcessor): An object that handles processing screenshots to detect the current game state.
            player (SolitairePlayer): An object representing the player, capable of simulating moves in the game.
            screen_region (Rectangle): A rectangle representing the region of the screen where the game is displayed.
        
        Initializes the `moves_history` to track the moves made, the `known_cards` to keep track of discovered cards, 
        and counts the number of `total_hidden_cards` based on the game state. Additionally, the solver initializes 
        `known_cards` from the current visible cards.
        """
        self.screen_region = screen_region  # Define the region on the screen where the game is played
        self.image_processor = image_processor  # Image processor used for processing screenshots of the game state
        self.player = player  # The player that will perform actions in the game
        self.game_state = deepcopy(game_state)  # Deep copy of the initial game state to track changes during gameplay
        self.moves_history = []  # Keeps a history of moves made during the game
        self.known_cards = set()  # Set of known cards that are visible or revealed
        self.total_hidden_cards = self._count_hidden_cards()  # Tracks the total number of hidden cards in the game
        
        # Initialize known cards from visible positions in the initial game state
        self._initialize_known_cards()

    def _count_hidden_cards(self) -> int:
        """
        Count the total number of hidden cards in the game.

        This method counts:
            - Hidden stock cards (cards with card_value == 0).
            - Face-down tableau cards (those that are not revealed yet).
        
        Returns:
            int: The total count of hidden cards (including stock and tableau).
        """
        hidden_count = 0
        
        # Count hidden stock cards (cards with card_value == 0)
        for card in self.game_state["stock"].stock_stack:
            if card.card_value == 0:
                hidden_count += 1
        
        # Count face-down tableau cards (unopened cards)
        for tableau in self.game_state["tableau"]:
            hidden_count += tableau.unopen_count
        
        return hidden_count

    def _initialize_known_cards(self) -> None:
        """
        Initialize the set of known cards from the visible positions in the game.

        This method updates the `known_cards` set with the visible cards from:
            - The tableau (cards that are visible and have card values and suits).
            - The foundation (cards that have been placed in the foundation).
            - The stock (cards that are visible and have card values).
        """
        # Add visible tableau cards to known cards
        for tableau in self.game_state["tableau"]:
            visible_cards = tableau.card_stack
            for card in visible_cards:
                if hasattr(card, 'card_value') and hasattr(card, 'card_suit'):
                    self.known_cards.add((card.card_value, card.card_suit))

        # Add foundation cards to known cards
        for foundation in self.game_state["foundation"]:
            if foundation.current_card_value > 0:
                # Add each card in the foundation to the known cards set
                for i in range(len(foundation.current_card_value)):
                    self.known_cards.add((i+1, foundation.card_suit))

        # Add visible stock cards to known cards (cards with card_value != 0)
        for card in self.game_state["stock"].stock_stack:
            if card.card_value != 0:
                self.known_cards.add((card.card_value, card.card_suit))

    def evaluate_position(self) -> float:
        """
        Evaluate the current position's progress and return a score between 0 and 1,
        where higher values indicate better progress in the game.

        The score is calculated based on the following factors:
            - Foundation cards: Heavily weighted (50% of total score).
            - Revealed cards: Moderately weighted (30% of total score).
            - Empty tableau columns: Lightly weighted (20% of total score).

        Returns:
            float: The calculated score for the current game position, ranging from 0 to 1.
        """
        score = 0.0
        max_score = 1.0
        
        # Points for cards in the foundation (heavily weighted)
        foundation_cards = sum(f.current_card_value for f in self.game_state["foundation"])
        score += (foundation_cards / 52) * 0.5  # Foundation cards worth 50% of total score

        # Points for revealed cards (moderate weight)
        revealed_ratio = 1 - (self.total_hidden_cards / 52)
        score += revealed_ratio * 0.3  # Revealed cards worth 30% of total score

        # Points for empty tableau columns (light weight)
        empty_tableaus = sum(1 for t in self.game_state["tableau"] if not t.card_stack)
        score += (empty_tableaus / 7) * 0.2  # Empty tableaus worth 20% of total score
        
        # Validate card count consistency
        hidden_cards_count = self.get_hidden_cards_count()
        visible_cards_count = self.get_visible_cards_count()
        if hidden_cards_count + visible_cards_count != 52:
            raise ValueError(f"Card count mismatch: Visible ({visible_cards_count}), Hidden ({hidden_cards_count})")

        return min(score, max_score)


    def solve(self) -> Tuple[List[Move], float]:
        """
        Attempt to make the best possible moves in the solitaire game.

        This method iteratively searches for the next best move, executes it, and tracks the progress. 
        The loop terminates either after a maximum number of moves is reached, if no valid moves are found, 
        if the game is won, or if the player gets stuck in an infinite loop of the same move.

        The result is a tuple containing:
            - A list of moves made during the game.
            - The final score of the game position.

        Returns:
            Tuple[List[Move], float]: A tuple containing the list of moves and the final score.
        """
        max_moves = 1000  # Prevent infinite loops by limiting the number of moves
        move_count = 0  # Track the number of moves made
        stuck_count = 0  # Track how many times the player is stuck on the same move
        
        while move_count < max_moves:
            # Find the next best move
            move = self.find_next_move()
            
            # If no valid move is found, stop the game
            if not move:
                break

            try:
                # Print the progress report for the current move
                self.print_progress_report()

                # Execute the move and update the move count
                self.execute_move(move)
                move_count += 1

                # If the move is Tableau to Tableau, check if the player is stuck
                if move.move_type == MoveType.TABLEAU_TO_TABLEAU:
                    stuck_count = len([m for m in self.moves_history if m == move])

                # Stop if the player is stuck on the same move for too long
                if stuck_count > 5:
                    print("Player is stuck too long.")
                    break

                # Check if the game is won
                if self.is_game_won():
                    print("Game won!")
                    break

            except ValueError as e:
                print(f"Invalid move encountered: {e}")
                break

        # Evaluate the final position score and return the results
        final_score = self.evaluate_position()
        return self.moves_history, final_score

    def get_visible_cards_count(self) -> int:
        """Return the number of currently visible cards."""
        return len(self.known_cards)

    def get_hidden_cards_count(self) -> int:
        """Return the number of currently hidden cards."""
        return self.total_hidden_cards

    def get_progress_report(self) -> dict:
        """
        Get a detailed progress report of the current game state.
        """
        last_move = self.moves_history[-1] if self.moves_history else None
        return {
            'visible_cards': self.get_visible_cards_count(),
            'hidden_cards': self.get_hidden_cards_count(),
            'foundation_progress': [f.current_card_value for f in self.game_state["foundation"]],
            'empty_tableaus': sum(1 for t in self.game_state["tableau"] if not t.card_stack),
            'score': self.evaluate_position(),
            'known_cards_count': len(self.known_cards),
            'moves_made': len(self.moves_history),
            'last_move': last_move
        }

    def print_progress_report(self) -> None:
        """Print a formatted progress report."""
        report = self.get_progress_report()
        print("\nGame Progress Report:")
        print("=" * 50)
        print(f"Last Move: {report['last_move']}")
        print(f"Position Score: {report['score']:.2f}")
        print(f"Visible Cards: {report['visible_cards']}")
        print(f"Hidden Cards: {report['hidden_cards']}")
        print(f"Foundation Progress: {report['foundation_progress']}")
        print(f"Empty Tableaus: {report['empty_tableaus']}")
        print(f"Total Moves Made: {report['moves_made']}")
        print("=" * 50)

    def find_next_move(self) -> Optional[Move]:
        """
        Find the next best move in the current game state.

        This method searches through the possible moves in a specific order: 
        1. Moves to the foundation (from waste and tableau)
        2. Moves to tableau (from waste and tableau)
        3. Stock moves (either moving a card from stock to waste or restocking)

        It sorts the possible moves by priority (highest first) and returns the best possible move.
        If no valid move is found, it returns None.

        Returns:
            Optional[Move]: The best possible move, or None if no move is available.
        """
        possible_moves = []
        
        # 1. Check foundation moves first (from waste and tableau)
        possible_moves.extend(self._find_waste_to_foundation_moves())
        possible_moves.extend(self._find_tableau_to_foundation_moves())
        
        # 2. Check tableau moves second (from waste and tableau)
        possible_moves.extend(self._find_waste_to_tableau_moves())
        possible_moves.extend(self._find_tableau_to_tableau_moves())
        
        # 3. Check stock moves last
        if self.game_state["stock"].stock_stack:
            if self.game_state["stock"].is_available:
                # Add stock to waste move with high priority if stock is available
                possible_moves.append(Move(
                    move_type=MoveType.STOCK_TO_WASTE,
                    move_action=MoveAction.CLICK,
                    source=-1,
                    destination=-1,
                    source_loc=self.game_state["stock"].card_loc,
                    priority=50  # Highest priority for available stock
                ))
            else:
                # Add restock move with slightly lower priority if stock is not available
                possible_moves.append(Move(
                    move_type=MoveType.RESTOCK,
                    move_action=MoveAction.CLICK,
                    source=-1,
                    destination=-1,
                    source_loc=self.game_state["stock"].card_loc,
                    priority=40  # Restocking has a lower priority
                ))

        # Sort possible moves by priority (highest priority first)
        possible_moves.sort(key=lambda x: x.priority, reverse=True)
        
        # Return the move with the highest priority, or None if no moves are available
        return possible_moves[0] if possible_moves else None

    def _find_waste_to_foundation_moves(self) -> List[Move]:
        """
        Find all possible moves from the waste pile to the foundation piles.

        This method checks if a move from the current card in the waste pile to each of 
        the foundation piles is valid. If a valid move is found, a corresponding Move 
        object is created and added to the list of possible moves.

        Returns:
            List[Move]: A list of valid moves from waste to foundation piles, 
            or an empty list if no valid moves are found.
        """
        moves = []
        
        # If no card is in the waste pile, return an empty list
        if self.game_state["waste"].current_card_waste_index == -1:
            return moves

        # Get the card currently in the waste pile
        waste_card = self.game_state["stock"].stock_stack[self.game_state["waste"].current_card_stock_index]
        
        # Check each foundation pile to see if a move to that foundation is valid
        for i, foundation in enumerate(self.game_state["foundation"]):
            if self._is_valid_foundation_move(waste_card, foundation):
                # If the move is valid, create and append a Move object for the move
                moves.append(Move(
                    move_type=MoveType.WASTE_TO_FOUNDATION,  # Move type is WASTE_TO_FOUNDATION
                    move_action=MoveAction.CLICK_AND_DRAG,   # The move involves clicking and dragging
                    source=-1,                               # No specific source since it's from the waste pile
                    destination=i,                          # Destination is the foundation pile index
                    source_loc=self.game_state["waste"].card_loc[self.game_state["waste"].current_card_waste_index],  # Location of the waste card
                    destination_loc=self.game_state["foundation"][i].card_loc,  # Location of the foundation pile
                    card_info=waste_card,                    # Card being moved from the waste pile
                    priority=100                             # High priority for foundation moves
                ))

        return moves

    def _find_tableau_to_foundation_moves(self) -> List[Move]:
        """Find all possible moves from tableau to foundation piles."""
        moves = []
        for i, tableau in enumerate(self.game_state["tableau"]):
            if not tableau.card_stack:
                continue
                
            tableau_card = tableau.card_stack[-1]
            for j, foundation in enumerate(self.game_state["foundation"]):
                if self._is_valid_foundation_move(tableau_card, foundation):
                    moves.append(Move(
                        move_type=MoveType.TABLEAU_TO_FOUNDATION,
                        move_action=MoveAction.CLICK_AND_DRAG,
                        source=i,
                        destination=j,
                        source_loc=tableau.card_loc[-1],
                        destination_loc=foundation.card_loc,
                        card_info=tableau_card,
                        priority=90  # High priority but slightly lower than waste to foundation
                    ))
        return moves

    def _find_waste_to_tableau_moves(self) -> List[Move]:
        """
        Find all possible moves from the waste pile to the tableau piles.

        This method checks if a move from the current card in the waste pile to each of 
        the tableau piles is valid. If a valid move is found, a corresponding Move 
        object is created and added to the list of possible moves.

        Returns:
            List[Move]: A list of valid moves from waste to tableau piles, 
            or an empty list if no valid moves are found.
        """
        moves = []

        # If no card is in the waste pile, return an empty list
        if self.game_state["waste"].current_card_waste_index == -1:
            return moves

        # Get the card currently in the waste pile
        waste_card = self.game_state["stock"].stock_stack[self.game_state["waste"].current_card_stock_index]

        # Check each tableau pile to see if a move to that tableau is valid
        for i, tableau in enumerate(self.game_state["tableau"]):
            if self._is_valid_tableau_move(waste_card, tableau):
                # Determine the destination location for the move
                if tableau.card_loc:
                    # If the tableau has cards, use the last card's location as the destination
                    destination_loc = tableau.card_loc[-1]
                else:
                    # If the tableau is empty, calculate the destination location based on the tableau's contour
                    x, y, w, h = tableau.contour
                    destination_loc = CardLocation(x=x + CARD_CONFIG.card_w // 2, y=y + CARD_CONFIG.card_h // 2)

                # Create and append the Move object for the valid move
                moves.append(Move(
                    move_type=MoveType.WASTE_TO_TABLEAU,  # Move type is WASTE_TO_TABLEAU
                    move_action=MoveAction.CLICK_AND_DRAG,   # The move involves clicking and dragging
                    source=-1,                               # No specific source since it's from the waste pile
                    destination=i,                          # Destination is the tableau pile index
                    source_loc=self.game_state["waste"].card_loc[self.game_state["waste"].current_card_waste_index],  # Location of the waste card
                    destination_loc=destination_loc,  # Destination location for the tableau pile
                    card_info=waste_card,                    # Card being moved from the waste pile
                    priority=50                             # Medium priority for tableau moves
                ))

        return moves

    def _find_tableau_to_tableau_moves(self) -> List[Move]:
        """
        Find all possible moves between tableau piles.

        This method checks all pairs of tableau piles and finds valid moves where 
        a card can be moved from one tableau to another. The priority of moves 
        is determined based on whether a face-down card is revealed or a tableau 
        becomes empty.

        Returns:
            List[Move]: A list of valid moves between tableau piles, 
            or an empty list if no valid moves are found.
        """
        moves = []

        # Loop through each tableau as the source
        for i, source_tableau in enumerate(self.game_state["tableau"]):
            # Skip empty tableau piles (no cards to move)
            if not source_tableau.card_stack:
                continue

            # Loop through each tableau as the destination
            for j, dest_tableau in enumerate(self.game_state["tableau"]):
                # Skip moving to the same tableau
                if i == j:
                    continue

                # Check each card in the source tableau for possible moves
                for k, source_card in enumerate(source_tableau.card_stack):
                    # Check if the move is valid for the given source and destination tableau
                    if self._is_valid_tableau_move(source_card, dest_tableau):
                        # Default move priority
                        priority = 10

                        # Increase priority if the move reveals a face-down card
                        if k == 0:
                            if source_card.card_value != 13:
                                priority = 60
                            if source_tableau.unopen_count > 0:
                                priority = 70

                        # Determine the destination location for the move
                        if dest_tableau.card_loc:
                            # If the tableau has cards, use the last card's location as the destination
                            destination_loc = dest_tableau.card_loc[-1]
                        else:
                            # If the tableau is empty, calculate the destination location based on the tableau's contour
                            x, y, w, h = dest_tableau.contour
                            destination_loc = CardLocation(x=x + CARD_CONFIG.card_w // 2, y=y + CARD_CONFIG.card_h // 2)

                        # Create and append the Move object for the valid tableau-to-tableau move
                        moves.append(Move(
                            move_type=MoveType.TABLEAU_TO_TABLEAU,  # Move type is TABLEAU_TO_TABLEAU
                            move_action=MoveAction.CLICK_AND_DRAG,   # The move involves clicking and dragging
                            source=i,                                # Source is the tableau pile index
                            destination=j,                           # Destination is the tableau pile index
                            source_loc=source_tableau.card_loc[source_tableau.card_stack.index(source_card)],  # Location of the source card
                            destination_loc=destination_loc,        # Destination location for the tableau pile
                            card_info=source_card,                   # Card being moved from the source tableau
                            priority=priority                        # Priority of the move (higher priority for face-down card reveals)
                        ))

        return moves


    def _is_valid_foundation_move(self, card: CardInfo, foundation: FoundationState) -> bool:
        """
        Check if a card can be moved to a foundation pile.

        A move is valid if the card:
        - Matches the suit of the foundation pile.
        - Has a value exactly one higher than the current top card in the foundation.

        Args:
            card (CardInfo): The card that is being considered for the move.
            foundation (FoundationState): The foundation pile where the card might be moved.

        Returns:
            bool: True if the card can be legally moved to the foundation, False otherwise.
        """
        
        # Check if the suit matches the foundation pile's suit
        if card.card_suit != foundation.card_suit:
            return False
        
        # Check if the card value is one higher than the foundation's current top card value
        return card.card_value == foundation.current_card_value + 1


    def _is_valid_tableau_move(self, card: CardInfo, tableau: TableauState) -> bool:
        """
        Check if a card can be moved to a tableau pile.

        A move is valid if the card:
        - Is moved to an empty tableau, only if the card is a King (value 13).
        - Is moved onto an existing tableau pile if:
            - The card is of the opposite color of the top card in the tableau.
            - The card has a value exactly one less than the top card in the tableau.

        Args:
            card (CardInfo): The card that is being considered for the move.
            tableau (TableauState): The tableau pile where the card might be moved.

        Returns:
            bool: True if the card can be legally moved to the tableau, False otherwise.
        """
        
        # For an empty tableau, only accept a King (value 13)
        if not tableau.card_stack:
            return card.card_value == 13
        
        # Get the top card of the destination pile
        dest_card = tableau.card_stack[-1]
        
        # Card must be opposite color and one value lower than the destination card
        return (card.card_color != dest_card.card_color and 
                card.card_value == dest_card.card_value - 1)

    def execute_move(self, move: Move) -> None:
        """
        Execute the given move and update the game state accordingly.

        The method updates the game state by executing the appropriate action based on 
        the move type. It also logs the move in the history.

        Args:
            move (Move): The move to be executed, containing details such as move type, 
                        source, destination, and card information.

        Raises:
            ValueError: If an invalid move type is encountered.
        """
        
        # Execute the move action for the player
        self.player.execute_move(move)

        # Process based on the move type
        if move.move_type == MoveType.STOCK_TO_WASTE:
            self._execute_stock_to_waste()
        elif move.move_type == MoveType.RESTOCK:
            self._execute_restock()
        elif move.move_type == MoveType.WASTE_TO_FOUNDATION:
            self._execute_waste_to_foundation(move)
        elif move.move_type == MoveType.WASTE_TO_TABLEAU:
            self._execute_waste_to_tableau(move)
        elif move.move_type == MoveType.TABLEAU_TO_FOUNDATION:
            self._execute_tableau_to_foundation(move)
        elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
            self._execute_tableau_to_tableau(move)
        else:
            # Raise an error if an invalid move type is encountered
            raise ValueError(f"Invalid move type: {move.move_type}")

        # Append the executed move to the move history
        self.moves_history.append(move)

    def _execute_stock_to_waste(self) -> None:
        """Move the top card from the stock pile to the waste pile."""
        # Add the card from stock to waste
        self._add_card_to_waste()

    def _execute_restock(self) -> None:
        """
        Restore the stock pile and reset the waste pile's card indices.
        """
        # Update game state: make stock available and reset waste indices
        self.game_state["stock"].is_available = True
        self.game_state["waste"].current_card_stock_index = -1
        self.game_state["waste"].current_card_waste_index = -1

    def _execute_waste_to_foundation(self, move: Move) -> None:
        """Move a card from the waste pile to the corresponding foundation pile."""           
        waste_card = self._remove_card_from_waste()
        self._add_card_to_foundation(move)

    def _execute_waste_to_tableau(self, move: Move) -> None:
        """Move a card from waste to the corresponding tableau pile."""           
        waste_card = self._remove_card_from_waste()
        self._add_card_to_tableau(move, [waste_card])

    def _execute_tableau_to_foundation(self, move: Move) -> None:
        """Move a card from tableau to foundation pile."""
        card_list = self._remove_card_from_tableau(move)      
        self._add_card_to_foundation(move)

    def _execute_tableau_to_tableau(self, move: Move) -> None:
        """Move a card from one tableau pile to another."""
        card_list = self._remove_card_from_tableau(move)
        self._add_card_to_tableau(move, card_list)

    def _remove_card_from_waste(self):
        """
        Removes a card from the waste stack and updates the game state accordingly.
        """
        # Step 1: Pop a card from the stock_stack using the current_card_stock_index
        card_info = self.game_state["stock"].stock_stack.pop(self.game_state["waste"].current_card_stock_index)
        
        # Step 2: Decrease the current_card_stock_index to reflect the card removal
        self.game_state["waste"].current_card_stock_index -= 1
        
        # Step 3: Check if the updated current_card_stock_index is less than 3
        if self.game_state["waste"].current_card_stock_index < 3:
            # Reset the card_loc at the current_card_waste_index if the condition is met
            self.game_state["waste"].card_loc[self.game_state["waste"].current_card_waste_index] = CardLocation()
            
            # Decrease the current_card_waste_index
            self.game_state["waste"].current_card_waste_index -= 1
        
        # Step 4: Return the card_info that was removed
        return card_info

    def _remove_card_from_tableau(self, move):
        """
        Removes a sequence of cards from a tableau stack and updates the tableau state.
        """
        source_tableau = self.game_state["tableau"][move.source]
        
        # Step 1: Retrieve the source tableau based on the move.source index
        index_to_move = source_tableau.card_stack.index(move.card_info)
        
        card_list = []
        
        # Step 2: Find the index of the card to move and start removing cards
        while len(source_tableau.card_stack) > index_to_move:
            # Step 3: Remove the card from the tableau's card_stack and card_loc
            card = source_tableau.card_stack.pop(index_to_move)
            source_tableau.card_loc.pop(index_to_move)
            card_list.append(card)  # Add the removed card to the card_list

        # Step 4: Update the open_count based on the remaining cards in the tableau
        source_tableau.open_count = len(source_tableau.card_stack)
        
        # Step 5: Flip the top card of the tableau if necessary
        self._flip_unopen_tableau_card(source_tableau)
        
        # Step 6: Correct the location of the remaining tableau cards
        self._fix_tableau_card_loc(source_tableau)
        
        return card_list

    def _add_card_to_foundation(self, move):
        """
        Adds a card to the foundation pile by updating the current card value.
        """
        self.game_state["foundation"][move.destination].current_card_value += 1

    def _add_card_to_tableau(self, move, card_list):
        """
        Adds a list of cards to a tableau stack and updates the tableau state.
        """
        # Step 1: Retrieve the destination tableau based on the move.destination index
        destination_tableau = self.game_state["tableau"][move.destination]         
    
        # Step 2: Iterate over the card_list and append each card to the tableau's card stack and card locations
        for card in card_list:
            destination_tableau.card_stack.append(card)  # Add card to the stack
            destination_tableau.card_loc.append(CardLocation())  # Add a new CardLocation for the card

        # Step 3: Update the open_card count based on the new total cards in the tableau
        destination_tableau.open_card = len(destination_tableau.card_stack)

        # Step 4: Call _fix_tableau_card_loc to adjust the locations of cards in the tableau
        self._fix_tableau_card_loc(destination_tableau)
    

    def _add_card_to_waste(self):
        """
        Adds a card from the stock to the waste pile and updates the game state.
        """
        # Step 1: Increment the current_card_waste_index if it's less than 2
        if self.game_state["waste"].current_card_waste_index < 2:
            self.game_state["waste"].current_card_waste_index += 1

        # Step 2: Increment the current_card_stock_index to move to the next card in the stock
        self.game_state["waste"].current_card_stock_index += 1

        # Step 3: Check if we have reached the last card in the stock and mark the stock as unavailable
        if self.game_state["waste"].current_card_stock_index == len(self.game_state["stock"].stock_stack) - 1:
            self.game_state["stock"].is_available = False

        # Step 4: Capture a new screenshot of the screen region
        self.screen_region = capture_new_screenshot(self.screen_region)

        # Step 5: Calculate the top-left corner of the waste area based on stock and foundation positions
        x, y, w, h = self.game_state["stock"].contour
        top_left = x + w, y
        x, y, w, h = self.game_state["foundation"][0].contour
        bottom_right = x, y + h

        # Step 6: Update the waste pile contour using the image processor
        self.game_state["waste"].contour = self.image_processor.get_waste_contour(self.screen_region.image, top_left, bottom_right)

        # Step 7: Calculate and update the card's location in the waste pile based on the index and configuration
        x, y, w, h = self.game_state["waste"].contour
        self.game_state["waste"].card_loc[self.game_state["waste"].current_card_waste_index] = CardLocation(
            x=x + CARD_CONFIG.overlap_open_waste_card_w * self.game_state["waste"].current_card_waste_index + CARD_CONFIG.card_w // 2, 
            y=y + h // 2
        )

        # Step 8: If the card value in the stock is 0, update the card's information using the image processor
        if self.game_state["stock"].stock_stack[self.game_state["waste"].current_card_stock_index].card_value == 0:
            self.game_state["stock"].stock_stack[self.game_state["waste"].current_card_stock_index] = self.image_processor.get_waste_card_info(
                self.screen_region.image, self.game_state["waste"].contour, self.game_state["waste"].current_card_waste_index
            )


    def _flip_unopen_tableau_card(self, tableau):
        """
        Flips a face-down card in the tableau to face-up, updates the tableau state,
        and processes the newly revealed card's information.
        """
        # Step 1: Check if a face-down card has been revealed
        if tableau.open_count == 0 and tableau.unopen_count > 0:
            # Step 2: Update the counts for open and unopen cards
            tableau.unopen_count -= 1
            tableau.open_count += 1
            
            # Step 3: Capture a new screenshot for the tableau region
            self.screen_region = capture_new_screenshot(self.screen_region)
            
            # Step 4: Extract the contour of the newly revealed card
            open_card_contour = self.image_processor.extract_open_card_from_tableau(self.screen_region.image, tableau.contour)
            
            if open_card_contour:
                # Step 5: If the contour is valid, get the card information from the image
                card_info = self.image_processor.get_card_info(self.screen_region.image, open_card_contour)
                
                # Step 6: Process the tableau's card stack and location
                tableau.card_stack.append(card_info)
                tableau.card_loc.append(CardLocation(x=open_card_contour[0] + CARD_CONFIG.card_w // 2,
                                                    y=open_card_contour[1] + CARD_CONFIG.card_h // 2))


    def _fix_tableau_card_loc(self, tableau):
        """
        Adjusts the locations of all cards in the tableau based on their stack and visibility.
        """
        if tableau.card_loc:
            # Calculate the height for unopen cards
            unopen_card_h = CARD_CONFIG.overlap_unopen_tableau_card_h * tableau.unopen_count
            top_open_card_edge = tableau.contour[1] + unopen_card_h
            
            # Determine the overlap height for open cards
            if len(tableau.card_loc) > 1:
                current_overlap_open_tableau_card_h = min(
                    (CARD_CONFIG.max_tableau_h - unopen_card_h - CARD_CONFIG.card_h) // (len(tableau.card_loc) - 1),
                    CARD_CONFIG.overlap_open_tableau_card_h
                )
            else:
                current_overlap_open_tableau_card_h = CARD_CONFIG.overlap_open_tableau_card_h
            
            # Update the positions for each card in the tableau
            for i, loc in enumerate(tableau.card_loc):
                loc.x = tableau.contour[0] + CARD_CONFIG.card_w // 2
                loc.y = top_open_card_edge + current_overlap_open_tableau_card_h * i + current_overlap_open_tableau_card_h // 2

            # Special case for the last card's position adjustment
            tableau.card_loc[-1].y = tableau.card_loc[-1].y - current_overlap_open_tableau_card_h // 2 + CARD_CONFIG.card_h // 2

    def is_game_won(self) -> bool:
        """
        Checks if the game is won by verifying if all foundation piles have 13 cards.
        """
        return all(f.current_card_value == 13 for f in self.game_state["foundation"])


