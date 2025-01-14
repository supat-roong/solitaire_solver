import sys
import os
from pathlib import Path
import logging

# Add the 'src' directory to the Python path to make the necessary modules accessible.
# This allows importing the custom modules in the 'src' directory without needing to specify their full path.
src_dir = os.path.join(str(Path(__file__).resolve().parent), "src")
sys.path.append(src_dir)

# Import the custom modules needed for the script.
from debugger import Debugger
from image_processor import SolitaireImageProcessor
from solver import SolitaireSolver
from player import SolitairePlayer
from screenshot import capture_screen_region
from config import DEBUG


def main():
    """
    Main function that initializes and runs the Solitaire game solver.
    """
    # Set up logging configuration: log messages will be printed at the INFO level.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Create a logger for this module.

    # Initialize the Debugger if debugging is enabled.
    debugger = None
    if DEBUG:
        debugger = Debugger()

    # Capture the screen region where the game is displayed.
    screen_region = capture_screen_region()

    # Initialize the image processor to process the captured screen image and extract game state information.
    image_processor = SolitaireImageProcessor(debugger)

    # Create a SolitairePlayer to manage gameplay.
    player = SolitairePlayer(screen_region)

    # Process the initial image to extract the game state (e.g., cards, tableau, etc.).
    initial_game_state = image_processor.process_initial_image(screen_region.image)

    # Log the initial game state for inspection.
    logger.info(f"Initial Game State: {initial_game_state}")

    # Create the SolitaireSolver to determine the best moves to solve the game based on the initial game state.
    solver = SolitaireSolver(initial_game_state, image_processor, player, screen_region)

    # Solve the game using the solver.
    solver.solve()

    # Log the final game state after solving.
    logger.info(f"Final Game State: {solver.game_state}")


if __name__ == "__main__":
    # Entry point of the script: calls the main function to start the game-solving process.
    main()
