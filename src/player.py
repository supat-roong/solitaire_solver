import time
import pyautogui
from config import PLAYER_CONFIG
from custom_dataclass import Move, MoveAction


class SolitairePlayer:
    """
    Class to simulate a Solitaire player that interacts with the game by 
    performing mouse movements and actions based on the given moves.

    Attributes:
        screen_region (Rectangle): The region of the screen where the game is located.
    """

    def __init__(self, screen_region) -> None:
        """
        Initializes the SolitairePlayer instance.

        Args:
            screen_region (Rectangle): The region of the screen where the Solitaire game is displayed.
        """
        self.screen_region = screen_region
        pyautogui.PAUSE = PLAYER_CONFIG.action_pause  # Set global action pause between pyautogui actions.

    def execute_move(self, move: Move) -> None:
        """
        Executes the specified move in the Solitaire game, simulating mouse actions.

        Args:
            move (Move): The move object containing the move's details, including source and destination coordinates,
                         and the type of action (click or click-and-drag).

        This method performs different actions based on the type of move:
            - MoveAction.CLICK: Moves the mouse to the source location and clicks.
            - MoveAction.CLICK_AND_DRAG: Moves the mouse to the source location, clicks and drags to the destination.
        """
        # Calculate the actual screen coordinates based on the screen region
        source_loc_x, source_loc_y = self.screen_region.x + move.source_loc.x, self.screen_region.y + move.source_loc.y

        # If single tap-to-move is enabled, just move and click at the source location
        if PLAYER_CONFIG.enable_single_tap_to_move:
            pyautogui.moveTo(source_loc_x, source_loc_y, duration=PLAYER_CONFIG.move_duration)
            pyautogui.click(source_loc_x, source_loc_y)

        else:
            # Perform the specified move action
            move_action = move.move_action
            if move_action == MoveAction.CLICK:
                # Click action: move to source and click
                pyautogui.moveTo(source_loc_x, source_loc_y, duration=PLAYER_CONFIG.move_duration)
                pyautogui.click(source_loc_x, source_loc_y)
            elif move_action == MoveAction.CLICK_AND_DRAG:
                # Click and drag action: move to source, hold, and drag to destination
                destination_loc_x, destination_loc_y = self.screen_region.x + move.destination_loc.x, self.screen_region.y + move.destination_loc.y

                pyautogui.mouseDown(button="left")  # Press and hold the mouse button
                pyautogui.moveTo(destination_loc_x, destination_loc_y, duration=PLAYER_CONFIG.drag_duration)  # Drag to the destination
                pyautogui.mouseUp()  # Release the mouse button

        # Wait for animation to finish (e.g., card reveal animation)
        time.sleep(PLAYER_CONFIG.animation_pause)
