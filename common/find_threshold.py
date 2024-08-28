import cv2
import numpy as np
import json
class find_best_threshold:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = None
        self.threshold_value = 0
        self.min_contour_area = 50
        self.cropped_original = None
        self.cropped_original = None
        self.original = None
        self.display_height = None
        self.load_img()
        self.init_trackbars()
    def load_img(self):
        self.original = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise FileNotFoundError("Image file not found.")
        self.cropped_original = self.original[160:1200, :]
        self.max_contour_area = 0.9 * self.cropped_original.size
        self.display_width = 600  # Increase width for better resolution
        scale_factor = self.display_width / self.cropped_original.shape[1]
        self.display_height = int(self.cropped_original.shape[0] * scale_factor)

    def update_threshold_value(self, val):
        self.threshold_value = val

    def update_min_contour_area(self,val):
        self.min_contour_area = val
        # settings.update_min_contour_area(val)

    def add_title(self, image, title, width):
        """

        :param image:
        :param title:
        :param width:
        :return:
        """
        # Create a blank image for the title
        title_image = np.zeros((50, width, 3), dtype=np.uint8)
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        # Get text size
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_x = (title_image.shape[1] - text_size[0]) // 2
        text_y = (title_image.shape[0] + text_size[1]) // 2
        # Put text on the title image
        cv2.putText(title_image, title, (text_x, text_y), font, font_scale, color, thickness)
        # Stack title and image vertically
        combined = np.vstack((title_image, image))
        return combined

    def init_trackbars(self):
        # Initialize the main window and trackbars for all parameters
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Controls', 500, 500)  # Adjust the size to fit all controls comfortably
        cv2.createTrackbar('Threshold Value', 'Controls', 0, 255, self.update_threshold_value)
        cv2.createTrackbar('Min Contour Area', 'Controls', 50, 500, self.update_min_contour_area)  # New parameter
        # Create a named window that can be resized for results
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Result', self.display_width * 2, self.display_height * 2 + 200)  # Adjusted for four panels and titles

    def show_result(self, thresholded, contour_image, filtered_contour_image):
        """

        :param thresholded:
        :param contour_image:
        :param filtered_contour_image:
        :return:
        """
        # Convert grayscale images to BGR for stacking
        cropped_original_bgr = cv2.cvtColor(self.cropped_original, cv2.COLOR_GRAY2BGR)
        thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        # Resize images to ensure consistent dimensions
        original_bgr_resized = cv2.resize(cropped_original_bgr, (self.display_width, self.display_height))
        thresholded_bgr_resized = cv2.resize(thresholded_bgr, (self.display_width, self.display_height))
        contour_image_resized = cv2.resize(contour_image, (self.display_width, self.display_height))
        filtered_contour_image_resized = cv2.resize(filtered_contour_image,
                                                    (self.display_width, self.display_height))

        # Add titles and stack images
        titled_original = self.add_title(original_bgr_resized, "Original", self.display_width)
        titled_thresholded = self.add_title(thresholded_bgr_resized, "Thresholded", self.display_width)
        titled_contour = self.add_title(contour_image_resized, "Contours", self.display_width)
        titled_filtered_contour = self.add_title(filtered_contour_image_resized, "Filtered Contours",
                                                 self.display_width)

        # Stack images in two rows and two columns
        top_row = np.hstack((titled_original, titled_thresholded))
        bottom_row = np.hstack((titled_contour, titled_filtered_contour))

        combined_image = np.vstack((top_row, bottom_row))

        cv2.imshow('Result', combined_image)

    def write_json(self):
        settings = {
            "threshold_value": self.threshold_value,
            "min_contour_menu_area": self.min_contour_area
        }
        # 写入到 JSON 文件
        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
    def cal_threshold_roi_percent(self, image, contours, pixel_threshold):
        """
        compute contour area percent
        :param image:
        :param contours:
        :param pixel_threshold:
        :return:
        """
        for contour in contours:
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour
            pixel_values = image[mask == 255]
            position = tuple(contour[0][0])
            area_percent = np.sum(pixel_values < pixel_threshold) / len(pixel_values)
            self.put_text_on_ore(image,str(area_percent), position)
    def put_text_on_ore(self, contour_image,title, text_position):
        """
        write text on picture
        :param contour_image:
        :param title:
        :param text_position:
        :return:
        """
        cv2.putText(contour_image, title, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    def process_img(self):
        while True:
            # Read trackbar positions
            th_val = cv2.getTrackbarPos('Threshold Value', 'Controls')
            min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'Controls')
            # Thresholding
            if th_val > 0:
                _, thresholded = cv2.threshold(self.cropped_original, th_val, 255, cv2.THRESH_BINARY)
            else:
                thresholded = self.cropped_original
            # Find contours using cv2.RETR_TREE and cv2.CHAIN_APPROX_SIMPLE
            contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = cv2.cvtColor(self.cropped_original, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)  # Draw contours in red
            # Remove too small and large contours
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area
                                 and cv2.contourArea(cnt) <= self.max_contour_area]
            filtered_contour_image = cv2.cvtColor(self.cropped_original, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(filtered_contour_image, filtered_contours, -1, (0, 255, 0), 2) # Draw filtered contours in green
            self.cal_threshold_roi_percent(filtered_contour_image, filtered_contours, th_val)
            self.show_result(thresholded, contour_image, filtered_contour_image)
            # Break loop on ESC key press
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        self.write_json()
processor = find_best_threshold(r'../../data/1/code/face/diff_folder/diff_0001.jpg')
processor.process_img()