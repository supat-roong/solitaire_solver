import cv2
import os
import numpy as np
import pytesseract
from custom_dataclass import CardLocation, CardInfo, StockState, WasteState, FoundationState, TableauState
from config import TESSERACT_PATH, IMAGE_PROCESSOR_CONFIG, CARD_COLORS, CARD_SUITS, CARD_CONFIG, CONTOUR_AREA_COLORS
from debugger import Debugger
from typing import Dict, Optional

class SolitaireImageProcessor:
    """
    A class for processing images to recognize card suits and states for a solitaire game.
    
    This class provides functionality to preprocess images, detect card suits, 
    and recognize relevant game states. It integrates Tesseract OCR for text recognition
    and uses predefined templates for detecting card suits. Debugging and intermediate
    image saving are supported through the Debugger class.
    """

    def __init__(self, debugger: Optional[Debugger] = None):
        """
        Initialize the SolitaireImageProcessor instance.

        This constructor sets up the Tesseract OCR path, loads card suit templates, 
        and optionally enables debugging through a Debugger instance.

        Args:
            debugger (Optional[Debugger]): An optional instance of the Debugger class 
                                           used for logging or saving intermediate 
                                           processing images. If None, debugging is disabled.

        Attributes:
            debugger (Optional[Debugger]): The Debugger instance for handling debugging tasks.
            card_suits_template (dict[str, cv2.Mat]): A dictionary where keys are card suit names
                                                      and values are OpenCV image matrices of the
                                                      corresponding card suit templates.

        Raises:
            FileNotFoundError: If any card suit template image cannot be loaded.
            EnvironmentError: If the Tesseract OCR path is invalid or not executable.
        """
        self.debugger: Optional[Debugger] = debugger

        # Configure Tesseract OCR command path
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

        # Validate the Tesseract path
        if not os.path.isfile(TESSERACT_PATH) or not os.access(TESSERACT_PATH, os.X_OK):
            raise EnvironmentError(
                f"Tesseract executable not found or not executable at path: {TESSERACT_PATH}. "
                "Please check the Tesseract installation and the configured path."
            )

        # Dictionary to store preloaded card suit templates
        self.card_suits_template: dict[str, cv2.Mat] = {}

        # Load card suit templates
        for card_suit in CARD_SUITS:
            # Construct the file path for the card suit template
            template_path = os.path.join(IMAGE_PROCESSOR_CONFIG.card_suits_dir, f"{card_suit}.png")

            # Load the template image
            template_image = cv2.imread(template_path)

            # Validate that the template image was loaded successfully
            if template_image is None:
                raise FileNotFoundError(f"Could not load template for '{card_suit}' from path: {template_path}")

            # Store the loaded template in the dictionary
            self.card_suits_template[card_suit] = template_image

    def debug_image(self, image: np.ndarray, title: str) -> None:
        """Display intermediate images using the debugger's show_debug method."""
        if not self.debugger:
            return
        
        # Call the show_debug method of the Debugger class to handle image saving and displaying
        self.debugger.show_debug(title, image)

    def debug_data(self, data: str, title: str) -> None:
        """Print debug data using the debugger's show_debug method."""
        if not self.debugger:
            return
        
        # Call the show_debug method of the Debugger class to handle data printing
        print("=" * 50)
        print(f"[DEBUG] {title}:", data)
        print("=" * 50)

    def process_initial_image(self, image: np.ndarray) -> Dict:
            """
            Process a solitaire game image and return the classified game state.

            This method processes the input image to identify potential card contours using image 
            processing techniques such as grayscale conversion, Gaussian blurring, Canny edge detection, 
            and contour finding. It then classifies the game state based on the identified contours.

            Args:
                image (np.ndarray): The input image representing the current state of the solitaire game.

            Returns:
                Dict: A dictionary representing the classified game state based on the identified card contours.
                    The keys of the dictionary represent different game states (e.g., stock, waste, tableau, etc.)
                    and the values are the corresponding details of those states.
            """
            # Debugging: Display the original image
            self.debug_image(image, "Original Image")

            # Step 1: Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.debug_image(gray, "Grayscale Image")

            # Step 2: Apply Gaussian Blur to reduce noise and detail
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            self.debug_image(blurred, "Blurred Image")

            # Step 3: Detect edges using the Canny edge detector
            edges = cv2.Canny(blurred, 50, 150)
            self.debug_image(edges, "Edges")

            # Step 4: Find contours in the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.debug_data(len(contours), "Number of Contours")

            # Step 5: Filter and store potential card contours as bounding rectangles
            card_contours = [cv2.boundingRect(contour) for contour in contours]
            self.debug_data(card_contours, "Card Contours")

            # Step 6: Draw bounding rectangles around detected cards (for visualization)
            card_image = image.copy()
            for x, y, w, h in card_contours:
                cv2.rectangle(card_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.debug_image(card_image, "Detected Cards")

            # Step 7: Classify the game state based on the detected card contours
            game_state = self._classify_klondike_contours(image, card_contours)

            return game_state

    def _classify_klondike_contours(self, image: np.ndarray, contours: list[tuple]) -> Dict:
        """
        Classify contours into a Klondike Solitaire game layout based on the detected contours.

        This method processes the contours from an image of a Klondike Solitaire game, classifies 
        them into different game regions such as stock, waste, foundation, and tableau, and 
        returns the game state in the form of a dictionary.

        Args:
            image (np.ndarray): The input image representing the current state of the solitaire game.
            contours (list[tuple]): A list of contours, each represented by a tuple of (x, y, w, h),
                                    where (x, y) are the coordinates of the top-left corner and 
                                    (w, h) are the width and height of the bounding box.

        Returns:
            Dict: A dictionary representing the classified game state, which includes keys for stock, 
                waste, foundation, and tableau. The corresponding values contain the associated 
                game region states (e.g., `StockState`, `WasteState`, etc.).
        
        Raises:
            ValueError: If the expected number of tableau piles or top row contours is not detected.
        """
        # Initialize the game state
        game_state = {
            "stock": StockState(stock_stack=[CardInfo() for _ in range(24)], card_loc=CardLocation()),
            "waste": WasteState(card_loc=[CardLocation() for _ in range(3)]),
            "foundation": [FoundationState(card_suit=i, card_loc=CardLocation()) for i in range(len(CARD_SUITS))],
            "tableau": [TableauState(unopen_count=i, open_count=1, card_stack=[], card_loc=[]) for i in range(7)]
        }

        # Separate contours into top and bottom rows
        top_row_threshold = min([y + h for (x, y, w, h) in contours])
        top_row_contours = [(x, y, w, h) for (x, y, w, h) in contours if y + h / 2 < top_row_threshold]
        bottom_row_contours = [(x, y, w, h) for (x, y, w, h) in contours if (x, y, w, h) not in top_row_contours]

        # Sort contours
        top_row_contours = sorted(top_row_contours, key=lambda c: c[0])
        bottom_row_contours = sorted(bottom_row_contours, key=lambda c: c[0])

        if len(bottom_row_contours) != 7:
            raise ValueError(f"Expected 7 tableau piles, but found {len(bottom_row_contours)}.")

        if len(top_row_contours) < 5:
            raise ValueError(f"Not enough top row contours detected, found only {len(top_row_contours)}")

        # Handle waste pile
        waste_contour = None
        if len(top_row_contours) == 6:
            waste_contour = top_row_contours.pop(1)

        # Set card dimensions
        card_w = int(round(sum([w for (x, y, w, h) in top_row_contours]) / len(top_row_contours)))
        card_h = int(round(sum([h for (x, y, w, h) in top_row_contours]) / len(top_row_contours)))
        CARD_CONFIG.set_card_dimension(card_w, card_h)

        # Process top row contours
        self._process_top_row_contours(image, top_row_contours, game_state)
        
        # Process bottom row contours
        self._process_bottom_row_contours(image, bottom_row_contours, game_state)

        # Handle waste pile details
        if waste_contour is not None:
            game_state["waste"].contour = waste_contour
            current_card_waste_index = round((waste_contour[2] - card_w) / CARD_CONFIG.overlap_open_waste_card_w)
            game_state["waste"].current_card_waste_index = current_card_waste_index
            for i, card in enumerate(game_state["stock"].stock_stack):
                if card.card_value == 0:
                    game_state["waste"].current_card_stock_index = i
                    game_state["stock"].stock_stack[i] = self.get_card_info(image, waste_contour)
                    break

        return game_state

    def _process_top_row_contours(self, image, top_row_contours, game_state):
        """
        Process the contours in the top row, which correspond to the stock pile and foundation piles.
        
        The top row typically consists of two types of areas:
        - Stock pile: the initial stack of cards that can be drawn from.
        - Foundation piles: the four stacks where cards are placed to complete the game.
        
        Args:
            image (np.ndarray): The image containing the game state.
            top_row_contours (list): A list of contours corresponding to the stock pile and foundation piles.
            game_state (dict): The current state of the game, including information about the stock pile and foundation piles.
        
        Updates the `game_state` dictionary with the following:
            - Stock pile availability and location.
            - Foundation pile card information (e.g., card value, suit, contour).
        """
        for i, contour in enumerate(top_row_contours):
            contour_state = self._check_contour_state(image, contour)
            
            if i == 0:
                # Process stock pile contour
                game_state["stock"].is_available = (contour_state == "unopen_card")
                game_state["stock"].contour = contour
                game_state["stock"].card_loc.x = contour[0] + contour[2] // 2
                game_state["stock"].card_loc.y = contour[1] + contour[3] // 2
            elif 1 <= i <= 4:
                # Process foundation pile contours (index 1 to 4 corresponds to 4 foundation piles)
                foundation_pile = game_state["foundation"][i - 1]
                foundation_pile.contour = contour
                foundation_pile.card_loc.x = contour[0] + contour[2] // 2
                foundation_pile.card_loc.y = contour[1] + contour[3] // 2
                
                if contour_state == "background":
                    foundation_pile.current_card_value = 0
                elif contour_state == "open_card":
                    # Retrieve card information for open card in foundation pile
                    card_info = self.get_card_info(image, contour)
                    foundation_pile.current_card_value = card_info.card_value
                    foundation_pile.card_suit = card_info.card_suit
                    # Additional processing for foundation piles can go here...

    def _process_bottom_row_contours(self, image, bottom_row_contours, game_state):
        """
        Process the contours in the bottom row, which correspond to the tableau piles.
        
        The bottom row typically consists of tableau piles where cards are stacked.
        
        Args:
            image (np.ndarray): The image containing the game state.
            bottom_row_contours (list): A list of contours corresponding to the tableau piles.
            game_state (dict): The current state of the game, including information about the tableau piles.
        
        Updates the `game_state` dictionary with the following:
            - Tableau pile contour and card stack.
            - Location of open cards within each tableau pile.
        """
        for i, contour in enumerate(bottom_row_contours):
            tableau_pile = game_state["tableau"][i]
            tableau_pile.contour = contour
            
            # Extract the contour of the open card in the tableau pile (if any)
            open_card_contour = self.extract_open_card_from_tableau(image, contour)
            
            if open_card_contour:
                card_info = self.get_card_info(image, open_card_contour)
                # Append the card information to the tableau pile's card stack
                tableau_pile.card_stack.append(card_info)
                
                # Store the location of the open card
                tableau_pile.card_loc.append(CardLocation(x=open_card_contour[0] + open_card_contour[2] // 2,
                                                        y=open_card_contour[1] + open_card_contour[3] // 2))

    def get_card_info(self, image, contour):
        """
        Extracts the card information (color, value, and suit) from the given image 
        using the specified contour for localization. The method also applies OCR to 
        recognize the card value and identifies the card suit based on color.

        Parameters:
        - image (np.ndarray): The input image containing the card.
        - contour (tuple): A tuple (x, y, w, h) representing the bounding box 
                            of the card detected in the image.

        Returns:
        - card_info (CardInfo): A CardInfo object containing the extracted card's 
                                color, value, and suit information.
        """
        # Initialize a CardInfo object to store the extracted data
        card_info = CardInfo()
        
        # Extract the bounding box coordinates and dimensions from the contour
        x, y, w, h = contour

        # Define region of interest (ROI) for the card corner, used for color detection
        corner_roi_x = x + CARD_CONFIG.corner_offset
        corner_roi_y = y + CARD_CONFIG.corner_offset
        corner_roi_w = CARD_CONFIG.corner_w
        corner_roi_h = CARD_CONFIG.corner_h
        corner_roi = image[corner_roi_y:corner_roi_y+corner_roi_h, corner_roi_x:corner_roi_x+corner_roi_w]

        # Classify the card's color (either 'red' or 'black') based on the corner ROI
        card_color_str = self._classify_card_color(corner_roi)
        card_info.card_color = CARD_COLORS.index(card_color_str)

        # Extract the card value and suit images from the corner ROI
        card_value_image, card_suit_image = self._extract_card_value_and_suit(corner_roi)

        # Determine the card's suit based on its color
        if card_color_str == "red":
            card_suit_str = self._classify_card_suit(card_suit_image, ["diamond", "heart"])
        elif card_color_str == "black":
            card_suit_str = self._classify_card_suit(card_suit_image, ["spade", "club"])
        else:
            raise ValueError(f"Unknown card color found: {card_color_str}")
        
        # Store the card suit index in the CardInfo object
        card_info.card_suit = CARD_SUITS.index(card_suit_str)

        # Attempt OCR on the card value image and retry with different techniques if necessary
        for _ in range(IMAGE_PROCESSOR_CONFIG.num_candidate_ocr):
            card_value = self._card_value_ocr(card_value_image)
            print(f"OCR result: {card_value}")
            
            # If OCR results in a valid card value (between 1 and 13), break the loop
            if str(card_value) in [str(i) for i in range(1, 14)]:
                break

            # If OCR failed, try adding padding to the image and reattempt OCR
            print(f"Retry OCR with padded image")
            new_height, new_width = card_value_image.shape[0] + 20, card_value_image.shape[1] + 20
            new_img = np.full((new_height, new_width, 3), 255, dtype=np.uint8)  # Create white padding

            # Center the original image within the new padded image
            value_height, value_width = card_value_image.shape[:2]
            start_y = (new_height - value_height) // 2
            start_x = (new_width - value_width) // 2
            new_img[start_y:start_y + value_height, start_x:start_x + value_width] = card_value_image

            # Attempt OCR on the padded image
            card_value = self._card_value_ocr(new_img)
            print(f"OCR result: {card_value}")
            
            if str(card_value) in [str(i) for i in range(1, 14)]:
                break

            # If OCR still fails, try upscaling the image and reattempt OCR
            print(f"Retry OCR with upscaled image")
            width = int(card_value_image.shape[1] * IMAGE_PROCESSOR_CONFIG.upscale_factor)
            height = int(card_value_image.shape[0] * IMAGE_PROCESSOR_CONFIG.upscale_factor)
            card_value_image = cv2.resize(card_value_image, (width, height), interpolation=cv2.INTER_CUBIC)

        # If OCR still does not produce a valid result, default the value to '12' (Queen)
        if str(card_value) not in [str(i) for i in range(1, 14)]:
            card_value = 12

        # Log the final OCR result for debugging
        print(f"OCR result: {card_value}")

        # Store the recognized card value in the CardInfo object
        card_info.card_value = int(card_value)

        # Map the OCR value to its respective card representation ('A', 'J', 'Q', 'K')
        card_value_mapping = {"1": 'A', "11": 'J', "12": "Q", "13": "K"}
        card_value = card_value_mapping.get(str(card_value), card_value)

        # Annotate and display the result on the card image
        annotated_image = corner_roi.copy()
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        label = f"{card_value} of {card_suit_str.capitalize()}"
        cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show the annotated image for debugging
        self.debug_image(annotated_image, label)

        return card_info

    def _classify_card_color(self, roi):
        """
        Classifies the card's color as either "red" or "black" based on the given region of interest (ROI).

        This method uses HSV color space to create masks for red and black colors, then compares the 
        number of non-zero pixels in each mask to determine the dominant color.

        Parameters:
        - roi (numpy.ndarray): The region of interest containing the card, typically part of the card's 
                                suit or value, to analyze its color.

        Returns:
        - str: Returns "red" if the card's color is predominantly red, "black" if the card's color is 
            predominantly black.
        """
        # Convert the ROI to HSV color space for better color segmentation
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for detecting red color
        red_lower1 = np.array([0, 50, 50])   # Lower bound for red (Hue range: 0-10)
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50]) # Upper bound for red (Hue range: 170-180)
        red_upper2 = np.array([180, 255, 255])

        # Define HSV ranges for detecting black color
        black_lower = np.array([0, 0, 0])    # Low saturation and brightness for black
        black_upper = np.array([180, 255, 50])

        # Create binary masks for red and black colors
        red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2  # Combine both red ranges
        black_mask = cv2.inRange(hsv_roi, black_lower, black_upper)

        # Count non-zero pixels in each mask to determine the color dominance
        red_pixels = cv2.countNonZero(red_mask)
        black_pixels = cv2.countNonZero(black_mask)

        # Return "red" if red pixels dominate, otherwise return "black"
        if red_pixels > black_pixels:
            return "red"
        else:
            return "black"

    def _card_value_ocr(self, image):
        """
        Perform Optical Character Recognition (OCR) on the card value image to extract the card's value.

        This method first upscales the input image, processes it for OCR by converting it to grayscale and 
        applying binary thresholding. It attempts OCR in multiple stages with restricted character sets 
        to accurately identify card values.

        Parameters:
        - image (numpy.ndarray): The image containing the card's value, typically a cropped part of the card.

        Returns:
        - str: The recognized card value, either as a number (1-13) or a mapped character ('A', 'J', 'Q', 'K').
        """
        # Upscale the image to enhance OCR accuracy
        width = int(image.shape[1] * IMAGE_PROCESSOR_CONFIG.upscale_factor)
        height = int(image.shape[0] * IMAGE_PROCESSOR_CONFIG.upscale_factor)
        upscaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Convert the upscaled image to grayscale for better contrast
        gray_upscaled = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary thresholding using Otsu's method to binarize the image
        _, binary = cv2.threshold(gray_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert the binary image back to BGR format for OCR
        roi = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Attempt OCR with a restricted alphabet (Ace, Jack, Queen, King)
        card_value_text = pytesseract.image_to_string(
            roi,
            config='--psm 10 --oem 3 -c tessedit_char_whitelist=AJQK'
        ).strip().upper()

        # Define a mapping for face card values (A = 1, J = 11, Q = 12, K = 13)
        card_value_mapping = {'A': "1", 'J': "11", 'Q': "12", 'K': "13"}

        # If the OCR result is empty or invalid, attempt OCR with numeric characters
        if not card_value_text or card_value_text not in card_value_mapping:
            card_value_text = pytesseract.image_to_string(
                roi,
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=1234567890'
            ).strip().upper()

        # If still invalid, attempt OCR with a combined character set (numeric + face cards)
        if not card_value_text:
            card_value_text = pytesseract.image_to_string(
                roi,
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=1234567890AJQK'
            ).strip().upper()

        # Return the mapped card value or the OCR result if not a face card
        return card_value_mapping.get(card_value_text, card_value_text)

    def _extract_card_value_and_suit(self, image):
        """
        Extract the card value and suit from the corner Region of Interest (ROI) of the card image.

        This method processes the upper and lower parts of the card image to extract the card value 
        and suit based on contour detection. It first isolates the regions containing the card value 
        (upper part) and suit (lower part), then uses contour detection to identify and extract the 
        relevant regions for OCR processing.

        Parameters:
        - image (numpy.ndarray): The input card image containing the value and suit in the corners.

        Returns:
        - tuple: A tuple containing two images:
            - value_img (numpy.ndarray): The cropped image of the card value.
            - suit_img (numpy.ndarray): The cropped image of the card suit.
        
        Raises:
        - ValueError: If no valid contours are found for either the card value or suit.
        """
        # Extract upper and lower regions of interest (ROIs) for value and suit
        upper_img = image[:25, :]  # Upper part for card value
        lower_img = image[22:, :]  # Lower part for card suit

        # Convert the upper ROI to grayscale and apply binary thresholding (invert)
        gray = cv2.cvtColor(upper_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the upper ROI (card value area)
        upper_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Filter contours based on area to identify valid card value contours
        card_value_contours = [
            cv2.boundingRect(contour) for contour in upper_contours
            if cv2.contourArea(contour) > IMAGE_PROCESSOR_CONFIG.contour_area_threshold
        ]

        # Extract the bounding box coordinates for the card value
        if card_value_contours:
            x_vals = [x for (x, y, w, h) in card_value_contours]
            y_vals = [y for (x, y, w, h) in card_value_contours]
            w_vals = [x + w for (x, y, w, h) in card_value_contours]
            h_vals = [y + h for (x, y, w, h) in card_value_contours]

            # Crop the value image based on the contours
            value_img = upper_img[min(y_vals):max(h_vals), min(x_vals):max(w_vals)]
        else:
            value_img = None

        # Convert the lower ROI to grayscale and apply binary thresholding (invert)
        gray = cv2.cvtColor(lower_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the lower ROI (card suit area)
        lower_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter contours based on area to identify valid card suit contours
        card_suit_contours = [
            cv2.boundingRect(contour) for contour in lower_contours
            if cv2.contourArea(contour) > IMAGE_PROCESSOR_CONFIG.contour_area_threshold
        ]

        # Extract the bounding box coordinates for the card suit
        if card_suit_contours:
            x_vals = [x for (x, y, w, h) in card_suit_contours]
            y_vals = [y for (x, y, w, h) in card_suit_contours]
            w_vals = [x + w for (x, y, w, h) in card_suit_contours]
            h_vals = [y + h for (x, y, w, h) in card_suit_contours]

            # Crop the suit image based on the contours
            suit_img = lower_img[min(y_vals):max(h_vals), min(x_vals):max(w_vals)]
        else:
            suit_img = None

        # Check if both value and suit contours were found
        if not (card_value_contours and card_suit_contours):
            raise ValueError("Not enough contours to determine card value and suit.")

        return value_img, suit_img

    def extract_open_card_from_tableau(self, image, tableau_contour):
        """
        Extract the open card from the tableau pile in a card game.

        This method scans the rows within a given region of interest (ROI) defined by the tableau contour
        to identify potential open card rows. It uses a color threshold to differentiate between open and 
        unopened cards, and then determines the best row corresponding to the open card.

        Parameters:
        - image (numpy.ndarray): The input image containing the tableau.
        - tableau_contour (tuple): A tuple (x, y, w, h) defining the bounding box of the tableau pile.

        Returns:
        - tuple: A tuple (x, y, w, h) representing the bounding box of the extracted open card region.

        Raises:
        - ValueError: If no open card rows are identified within the tableau ROI.
        """
        # Extract the region of interest (ROI) from the tableau using the bounding box
        x, y, w, h = tableau_contour
        roi = image[y:y + h, x:x + w]

        # List to store rows that could potentially contain open cards
        potential_open_card_rows = []

        # Loop through each row in the ROI to check for the open card
        for i in range(h):
            roi_row = roi[i:i+1, :]  # Extract a single row from the ROI

            # Create a blue mask to detect unopened cards based on the defined color range
            blue_mask = cv2.inRange(
                roi_row, 
                np.array(CONTOUR_AREA_COLORS["unopen_card"][0]), 
                np.array(CONTOUR_AREA_COLORS["unopen_card"][1])
            )

            # Calculate the ratio of blue pixels in the row
            blue_ratio = np.sum(blue_mask > 0) / w

            # If blue ratio is small, it indicates a potential open card row
            if blue_ratio < 0.05:
                potential_open_card_rows.append(i)

        # Check if any open card rows were identified
        if not potential_open_card_rows:
            raise ValueError("No open card rows found in the tableau ROI.")

        # Find and return the best region for the open card
        return self._find_best_open_card_region(x, y, w, potential_open_card_rows)

    def _find_best_open_card_region(self, x, y, w, potential_rows):
        """
        Find the best region for an open card based on consecutive rows.

        This function identifies the region with the most consecutive rows that 
        are likely to contain an open card. It returns the bounding box coordinates 
        of the best region based on the longest sequence of consecutive rows.

        Parameters:
        - x (int): The x-coordinate of the tableau bounding box.
        - y (int): The y-coordinate of the tableau bounding box.
        - w (int): The width of the tableau bounding box.
        - potential_rows (list of int): A list of row indices within the tableau 
                                        that potentially contain open cards.

        Returns:
        - tuple: A tuple (x, y, w, h) representing the bounding box of the best 
                region for the open card, where 'h' is the number of consecutive 
                rows with potential open cards.
        - None: If no suitable consecutive rows are found.
        """
        if not potential_rows:
            return None  # No potential rows found, return None

        # Initialize variables to track the best region based on consecutive rows
        max_consecutive = 0
        best_start = None
        current_start = potential_rows[0]
        current_consecutive = 1

        # Loop through the potential rows to find the longest sequence of consecutive rows
        for i in range(1, len(potential_rows)):
            if potential_rows[i] - potential_rows[i-1] == 1:
                # Consecutive row found, increase the count
                current_consecutive += 1
            else:
                # Sequence broken, check if current sequence is the longest
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
                    best_start = current_start
                # Start a new sequence
                current_start = potential_rows[i]
                current_consecutive = 1

        # Check the last sequence
        if current_consecutive > max_consecutive:
            max_consecutive = current_consecutive
            best_start = current_start

        # If a valid sequence is found, return the best region (x, y, width, height)
        if best_start is not None:
            return (x, y + best_start, w, max_consecutive)

        # No valid region found
        return None

    def _check_contour_state(self, image, contour):
        """
        Determine the state of the contour (background, open card, or unopen card) based on color distribution.

        This function checks the color distribution of a given contour in the image 
        and determines whether it represents a background, an open card, or an unopen card.
        The decision is made based on the area ratios of the color masks applied to the contour.

        Parameters:
        - image (numpy.ndarray): The image in which the contour is located.
        - contour (tuple): The bounding box of the contour in the form (x, y, w, h).

        Returns:
        - str: The state of the contour ("background", "open_card", "unopen_card").
        """
        x, y, w, h = contour
        roi = image[y:y + h, x:x + w]

        # Generate masks for different regions (background, open card, unopen card)
        masks = {
            "background": cv2.inRange(roi, np.array(CONTOUR_AREA_COLORS["background"][0]), 
                                    np.array(CONTOUR_AREA_COLORS["background"][1])),
            "open_card": cv2.inRange(roi, np.array(CONTOUR_AREA_COLORS["open_card"][0]), 
                                    np.array(CONTOUR_AREA_COLORS["open_card"][1])),
            "unopen_card": cv2.inRange(roi, np.array(CONTOUR_AREA_COLORS["unopen_card"][0]), 
                                        np.array(CONTOUR_AREA_COLORS["unopen_card"][1]))
        }

        # Calculate the ratio of the mask area for each state
        ratios = {k: np.sum(v > 0) / (w * h) for k, v in masks.items()}

        # Find the dominant state with the maximum ratio
        dominant_state = max(ratios, key=ratios.get)

        # Return the state with the highest ratio
        return dominant_state



    def _classify_card_suit(self, card_suit_image, card_suits):
        """
        Classifies the suit of a card from the given image using BGR templates, normalizing match scores for size differences.

        Args:
            card_suit_image (np.ndarray): Image of the card's suit to classify.
            card_suits (list): A list of suit names to classify among (e.g., ['hearts', 'diamonds', 'clubs', 'spades']).

        Returns:
            str: The classified suit ('hearts', 'diamonds', 'clubs', 'spades').
        """
        # Initialize variables
        templates = {key: value for key, value in self.card_suits_template.items() if key in card_suits}
        
        # Define upscale factors to try
        upscale_factors = [IMAGE_PROCESSOR_CONFIG.upscale_factor * (2 ** i) for i in range(IMAGE_PROCESSOR_CONFIG.num_candidate_ocr)]

        # Sharpening kernel
        sharpening_kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])

        # Dictionary to count votes for each suit
        votes = {suit: 0 for suit in card_suits}

        # Loop through all upscale factors
        for upscale_factor in upscale_factors:
            best_match = None
            best_score = -1

            # Loop through all templates
            for suit, template in templates.items():
                # Resize the card suit image and the template
                width = int(card_suit_image.shape[1] * upscale_factor)
                height = int(card_suit_image.shape[0] * upscale_factor)
                resized_card_image = cv2.resize(card_suit_image, (width, height), interpolation=cv2.INTER_CUBIC)
                resized_template_image = cv2.resize(template, (width, height), interpolation=cv2.INTER_CUBIC)

                # Apply sharpening to the resized images
                sharpened_card_image = cv2.filter2D(resized_card_image, -1, sharpening_kernel)
                sharpened_template_image = cv2.filter2D(resized_template_image, -1, sharpening_kernel)

                # Match the descriptors using BFMatcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(
                    self._compute_orb_features(self._preprocess_card_suit_image(sharpened_card_image)),
                    self._compute_orb_features(self._preprocess_card_suit_image(sharpened_template_image))
                )

                # Calculate match score based on the number of good matches
                match_score = len(matches)

                if match_score > best_score:
                    best_score = match_score
                    best_match = suit

            # Add a vote to the best matching suit for the current upscale factor
            if best_match is not None:
                votes[best_match] += 1

        # Determine the best suit by majority vote
        final_match = max(votes, key=votes.get)
        
        # print out the voting results
        self.debug_data(data=votes, title="Card suits classification")

        return final_match

    def _preprocess_card_suit_image(self, image):
        """
        Preprocesses the input image of a playing card suit by converting it to grayscale,
        sharpening it, and then thresholding it to create a binary image.

        Steps:
            1. Converts the input image to grayscale.
            2. Applies a sharpening filter to enhance edges.
            3. Thresholds the sharpened image to create a binary image.

        Args:
            image (numpy.ndarray): The input image, typically a BGR image of the playing card.

        Returns:
            numpy.ndarray: A binary image where the foreground is white and the background is black.
        """
        
        # Convert the input image to grayscale to simplify processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define a kernel for sharpening the image
        kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])  # Sharpen kernel

        # Apply the sharpening filter to the grayscale image
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Apply a binary threshold to the sharpened image to obtain a black-and-white image
        _, binary = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)

        return binary

    def _compute_orb_features(self, image):
        """
        Computes the ORB (Oriented FAST and Rotated BRIEF) feature descriptors for the input image.

        ORB is a fast and efficient feature detector and descriptor extractor. It combines the
        FAST keypoint detector and BRIEF descriptor with orientation for rotation invariance.

        Args:
            image (numpy.ndarray): The input image for which ORB features are to be computed. The
                                image should be in grayscale format.

        Returns:
            numpy.ndarray: A NumPy array containing the ORB feature descriptors extracted from the 
                        input image. If no keypoints are found, an empty array is returned.
        
        Notes:
            - The `fastThreshold=0` and `edgeThreshold=0` parameters are set to their minimum
            values to increase the sensitivity of the ORB detector for keypoint detection.
            - If keypoints are found, their descriptors are returned as a 2D NumPy array, where each
            row represents a descriptor for a keypoint.
        """
        orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0)  # Create ORB detector
        _, descriptors = orb.detectAndCompute(image, None)
        return descriptors

    
    def get_waste_contour(self, image, top_left, bottom_right):
        """
        Extracts the largest contour of a waste object from a region of interest (ROI) in the input image.

        The function processes the given image to find the largest contour within a defined rectangular
        area, adjusted by a fixed boundary around the ROI. The contour is determined based on edge 
        detection, and the bounding rectangle of the largest contour is returned.

        Args:
            image (numpy.ndarray): The input image (in BGR color format) from which the waste contour is 
                                to be detected.
            top_left (tuple): The (x, y) coordinates of the top-left corner of the region of interest (ROI).
            bottom_right (tuple): The (x, y) coordinates of the bottom-right corner of the ROI.

        Returns:
            tuple: A tuple representing the bounding rectangle of the largest contour, in the form
                (x, y, width, height). If no contours are found, returns None.

        Notes:
            - The ROI is adjusted by 20 pixels in all directions, ensuring the boundaries do not go 
            outside the image dimensions.
            - Image processing steps include converting to grayscale, applying Gaussian blur, and 
            detecting edges using the Canny edge detector.
            - The function assumes that the input image is in color (BGR format).
            - If multiple contours are found, the largest one (by area) is selected.
        """
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Fix boundary adjustments
        y1 = max(y1 - 20, 0)  # Ensure y1 is not less than 0
        y2 = min(y2 + 20, image.shape[0])  # Ensure y2 does not exceed image height
        x1 = max(x1 - 20, 0)  # Ensure x1 is not less than 0
        x2 = min(x2 + 20, image.shape[1])  # Ensure x2 does not exceed image width

        roi = image[y1:y2, x1:x2]
        
        # Image processing pipeline
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None  # Return None if no contours are found

        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Return the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x + x1, y + y1, w, h)

    
    def get_waste_card_info(self, image, contour, waste_index):
        """
        Extracts information from a waste card within a given contour region in the input image.

        The function modifies the contour to align with a specific waste card based on the provided 
        `waste_index`, and then retrieves the card's information using the `get_card_info` function.

        Args:
            image (numpy.ndarray): The input image (in BGR color format) containing the waste card.
            contour (tuple): A tuple (x, y, width, height) representing the bounding box of the contour
                            where the waste card is located.
            waste_index (int): The index of the waste card to be processed, used to adjust the contour's 
                            horizontal position.

        Returns:
            CardInfo: The information of the waste card retrieved from the image, as returned by the 
                `get_card_info` function.
        """
        x, y, w, h = contour
        # Adjust the contour based on waste_index
        new_contour = (x + CARD_CONFIG.overlap_open_waste_card_w * waste_index, y, CARD_CONFIG.card_w, h)
        x, y, w, h = new_contour
        # Get the card information using the adjusted contour
        return self.get_card_info(image, new_contour)



