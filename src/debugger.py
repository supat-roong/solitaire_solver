from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Debugger:
    """
    A visualization tool for debugging image processing pipelines.

    This class provides functionality to display intermediate results of image
    processing operations, allowing step-by-step visualization with interactive
    control. Each visualization is numbered sequentially and requires user
    interaction to proceed to the next step.

    Attributes:
        current_step (int): Counter keeping track of the current debugging step number.
        window_position (Tuple[int, int]): The position (x, y) for the debugging window.
    """

    def __init__(self, window_position: Tuple[int, int] = (300, 50)) -> None:
        """
        Initialize the Debugger with a step counter starting at 0 and set window position.

        Args:
            window_position (Tuple[int, int]): Tuple specifying the (x, y) position
                of the window on the screen. Defaults to (450, 160).
        """
        self.current_step: int = 0
        self.window_position: Tuple[int, int] = window_position

    def _set_window_position(self) -> None:
        """Set the position of the matplotlib window if supported by the backend."""
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "window"):
            try:
                manager.window.wm_geometry(f"+{self.window_position[0]}+{self.window_position[1]}")
            except Exception:
                pass  # Gracefully handle backends that don't support window positioning

    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        """
        Display a debug image and wait for user interaction before proceeding.

        Args:
            title (str): Descriptive title for the current debugging step.
            img (np.ndarray): Image array to display. Can be either grayscale
                (2D array) or color (3D array in BGR format).
            cmap (Optional[str]): Colormap to use for displaying the image.
                Particularly useful for grayscale images or heat maps.
                Defaults to None, which uses matplotlib's default colormap.

        Notes:
            - Increments the internal step counter automatically.
            - Automatically converts BGR to RGB for color images.
            - Displays the image in a numbered step with grid and axes.
            - Waits for a key press before closing the figure.
        """
        # Increment the step counter
        self.current_step += 1

        # Validate the input image
        if not isinstance(img, np.ndarray):
            raise ValueError("The provided image must be a numpy ndarray.")
        if img.ndim not in {2, 3}:
            raise ValueError("The provided image must be either a 2D (grayscale) or 3D (color) array.")

        # Create new figure with specified size
        plt.figure(figsize=(10, 8))
        plt.title(f"Step {self.current_step}: {title}")

        # Convert BGR to RGB if dealing with a color image
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image with specified parameters
        plt.imshow(img, cmap=cmap)
        plt.axis("on")
        plt.grid(True)

        # Add instruction text at the bottom
        plt.figtext(0.5, 0.01, "Press any key to continue...", ha="center", va="bottom", fontsize=10)

        # Set the window position
        self._set_window_position()

        # Show the plot and wait for user input
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
