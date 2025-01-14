from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
import numpy as np


@dataclass
class CardLocation:
    """
    Represents the location of a card on the screen.

    Attributes:
        x: X-coordinate of the card's position (in pixels).
        y: Y-coordinate of the card's position (in pixels).
    """
    x: int = -1
    y: int = -1


@dataclass
class CardInfo:
    """
    Stores information about a card in the game.

    Attributes:
        card_value: The value of the card (e.g., 2-10, Jack, Queen, King, Ace).
        card_suit: The suit of the card (e.g., Hearts, Diamonds, etc.).
        card_color: The color of the card (e.g., red, black).
    """
    card_value: int = 0
    card_suit: int = 0
    card_color: int = 0


class MoveType(Enum):
    """
    Enum representing different types of moves in the game.

    - STOCK_TO_WASTE: Move from the stock pile to the waste pile.
    - RESTOCK: Move cards from the waste pile back to the stock pile.
    - WASTE_TO_FOUNDATION: Move a card from the waste pile to the foundation.
    - WASTE_TO_TABLEAU: Move a card from the waste pile to the tableau.
    - TABLEAU_TO_FOUNDATION: Move a card from the tableau to the foundation.
    - TABLEAU_TO_TABLEAU: Move a card from one tableau pile to another.
    """
    STOCK_TO_WASTE = "stock_to_waste"
    RESTOCK = "restock"
    WASTE_TO_FOUNDATION = "waste_to_foundation"
    WASTE_TO_TABLEAU = "waste_to_tableau"
    TABLEAU_TO_FOUNDATION = "tableau_to_foundation"
    TABLEAU_TO_TABLEAU = "tableau_to_tableau"


class MoveAction(Enum):
    """
    Enum representing possible move actions.

    - CLICK: Move the card by clicking it.
    - CLICK_AND_DRAG: Move the card by clicking and dragging it.
    """
    CLICK = "click"
    CLICK_AND_DRAG = "click_and_drag"


@dataclass
class Move:
    """
    Represents a move action taken in the game.

    Attributes:
        move_type: Type of the move (e.g., stock_to_waste, tableau_to_foundation).
        move_action: The action taken for the move (e.g., click, click_and_drag).
        source: Index of the source pile (e.g., stock, tableau).
        destination: Index of the destination pile (e.g., foundation, tableau).
        source_loc: The source card location (optional).
        destination_loc: The destination card location (optional).
        card_info: Information about the card being moved (optional).
        priority: The priority of the move, with higher numbers indicating higher priority.
    """
    move_type: MoveType
    move_action: MoveAction
    source: int  # Index of source pile
    destination: int  # Index of destination pile
    source_loc: Optional[CardLocation] = None
    destination_loc: Optional[CardLocation] = None
    card_info: Optional[CardInfo] = None
    priority: int = 0  # Higher number means higher priority


@dataclass
class StockState:
    """
    Represents the state of the stock pile in the game.

    Attributes:
        stock_stack: List of cards in the stock pile.
        card_loc: The location of the stock pile.
        contour: Bounding box coordinates for the stock pile on the screen.
        is_available: Boolean indicating if the stock pile is available for use.
    """
    stock_stack: List[CardInfo]
    card_loc: CardLocation
    contour: Tuple[int, int, int, int] = (-1, -1, -1, -1)
    is_available: bool = True


@dataclass
class WasteState:
    """
    Represents the state of the waste pile in the game.

    Attributes:
        card_loc: List of card locations in the waste pile.
        current_card_stock_index: Index of the current card in the stock.
        current_card_waste_index: Index of the current card in the waste pile.
        contour: Bounding box coordinates for the waste pile on the screen.
    """
    card_loc: List[CardLocation]
    current_card_stock_index: int = -1
    current_card_waste_index: int = -1
    contour: Tuple[int, int, int, int] = (-1, -1, -1, -1)


@dataclass
class FoundationState:
    """
    Represents the state of the foundation piles in the game.

    Attributes:
        card_loc: The location of the foundation pile.
        current_card_value: The current value of the card in the foundation.
        card_suit: The suit of the current card in the foundation.
        contour: Bounding box coordinates for the foundation pile on the screen.
    """
    card_loc: CardLocation
    current_card_value: int = 0
    card_suit: int = 0
    contour: Tuple[int, int, int, int] = (-1, -1, -1, -1)


@dataclass
class TableauState:
    """
    Represents the state of the tableau piles in the game.

    Attributes:
        card_loc: List of card locations in the tableau.
        card_stack: List of cards in the tableau.
        unopen_count: The number of unopened cards in the tableau.
        open_count: The number of opened cards in the tableau.
        contour: Bounding box coordinates for the tableau pile on the screen.
    """
    card_loc: List[CardLocation]
    card_stack: List[CardInfo]
    unopen_count: int = 0
    open_count: int = 0
    contour: Tuple[int, int, int, int] = (-1, -1, -1, -1)


@dataclass
class ImageProcessorConfig:
    """
    Configuration for image processing in the game.

    Attributes:
        num_candidate_ocr: Number of candidate OCR results to consider.
        upscale_factor: Factor by which to upscale the images.
        contour_area_threshold: Minimum area of contours to consider.
        card_suits_dir: Directory containing images of card suits.
    """
    num_candidate_ocr: int
    upscale_factor: float
    contour_area_threshold: int  
    card_suits_dir: str


@dataclass
class PlayerConfig:
    """
    Configuration for player actions and animations.

    Attributes:
        enable_single_tap_to_move: Whether single tap to move is enabled.
        action_pause: Duration of the pause after an action.
        animation_pause: Duration of the pause between animations.
        drag_duration: Duration of the drag action.
        move_duration: Duration of the move action.
    """
    enable_single_tap_to_move: bool
    action_pause: float
    animation_pause: float
    drag_duration: float
    move_duration: float


@dataclass
class CardConfig:
    """
    Configuration for card layout and appearance.

    Attributes:
        overlap_open_waste_card_w_ratio: Width ratio for overlapping open waste cards.
        overlap_open_tableau_card_h_ratio: Height ratio for overlapping open tableau cards.
        overlap_unopen_tableau_card_h_ratio: Height ratio for overlapping unopened tableau cards.
        max_tableau_h_ratio: Maximum height ratio for tableau cards.
        corner_offset: Offset for card corners.
        corner_w_ratio: Width ratio for corner offset.
        corner_h_ratio: Height ratio for corner offset.
    """
    overlap_open_waste_card_w_ratio: float
    overlap_open_tableau_card_h_ratio: float
    overlap_unopen_tableau_card_h_ratio: float
    max_tableau_h_ratio: float
    corner_offset: int
    corner_w_ratio: float
    corner_h_ratio: float

    def set_card_dimension(self, card_w, card_h):
        """
        Set the dimensions for the cards and calculate layout ratios.

        Attributes:
            card_w: Width of the card (in pixels).
            card_h: Height of the card (in pixels).
        """
        print(f"Card width: {card_w} px")
        print(f"Card height: {card_h} px")
        self.card_w = card_w
        self.card_h = card_h
        self.overlap_open_waste_card_w = int(round(self.overlap_open_waste_card_w_ratio * card_w))
        self.overlap_open_tableau_card_h = int(round(self.overlap_open_tableau_card_h_ratio * card_h))
        self.overlap_unopen_tableau_card_h = int(round(self.overlap_unopen_tableau_card_h_ratio * card_h))
        self.max_tableau_h = int(round(self.max_tableau_h_ratio * card_h))
        self.corner_w = int(round(self.corner_w_ratio * card_w))
        self.corner_h = int(round(self.corner_h_ratio * card_h))

@dataclass
class ScreenRegion:
    """
    Represents a rectangular region on the screen with associated image data.

    Attributes:
        x: X-coordinate of the top-left corner.
        y: Y-coordinate of the top-left corner.
        width: Width of the region in pixels.
        height: Height of the region in pixels.
        image: Numpy array containing the image data (BGR format).
    """
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray

    def __post_init__(self):
        """Validate region parameters and image data."""
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if len(self.image.shape) != 3 or self.image.shape[2] != 3:
            raise ValueError("image must be a 3-dimensional array with 3 color channels (BGR)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.image.shape[0] != self.height or self.image.shape[1] != self.width:
            raise ValueError(
                f"Image dimensions ({self.image.shape[1]}x{self.image.shape[0]}) do not match specified dimensions ({self.width}x{self.height})"
            )
