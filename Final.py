import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import tempfile
import json
import tensorflow as tf
import imutils

def preprocess_image(input_path, output_path=None):
    """
    Preprocess a single image with skew correction, line removal, 
    noise reduction, and adaptive thresholding.
    
    :param input_path: Path to the input image
    :param output_path: Path to save the preprocessed image (optional)
    :return: Preprocessed image
    """
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Error reading image from {input_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove horizontal & vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    remove_horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    remove_vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Find contours of horizontal and vertical lines
    cnts_h = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts_v = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Draw white over the line contours to remove them
    for c in cnts_h + cnts_v:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 5)

    # Apply noise reduction and adaptive thresholding
    blurred = cv2.medianBlur(gray, 3)  # Noise reduction
    filtered = cv2.fastNlMeansDenoising(blurred, None, 30, 7, 21)
    final_img = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    # Save the preprocessed image if output path is provided
    if output_path:
        cv2.imwrite(output_path, final_img)
        print(f"Preprocessed image saved to: {output_path}")

    return final_img

def clean_image(input_path, output_path):
    """
    Clean the image by removing small contours
    """
    # Read the input image
    image = cv2.imread(input_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold and dilate
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh1, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]  # if imutils.is_cv2() else cnts[1]

    # Make a copy of the original image for processing
    processed_image = image.copy()

    # Fix the contours format if needed
    fixed_cnts = []
    for cnt in cnts:
        if cnt.shape[1] == 4:  # If shape is (n, 4) instead of (n, 1, 2)
            # Take only the first two columns (x,y coordinates)
            fixed_cnt = cnt[:, :2].reshape(-1, 1, 2)
            fixed_cnts.append(fixed_cnt)
        else:
            fixed_cnts.append(cnt)

    # Sort the contours 
    sorted_ctrs = sorted(fixed_cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1])

    # Process each contour
    i = 0
    for cnt in sorted_ctrs:
        # Check the area of contour, if it is very small ignore it
        if cv2.contourArea(cnt) < 0:
            continue

        # Get bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out contours that are not within our desired dimensions
        if not (w > 0 and h > 0 and w < 20 and h < 20):
            continue
        
        # Create a mask for the current contour
        mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        
        # Replace the contour area with white pixels
        processed_image[mask == 255] = [255, 255, 255]

        i += 1

    # Save the processed image directly to the specified path
    cv2.imwrite(output_path, processed_image)
    print(f"Processed {i} contours and saved processed image to '{output_path}'")

def extract_text_region(input_path, output_path):
    """
    Extract the main text region from the image
    """
    # Read image and convert to grayscale
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    # Functions for projection profiles
    def getHorizontalProjectionProfile(image):
        binary = np.where(image > 0, 1, 0)
        horizontal_projection = np.sum(binary, axis=1)
        return horizontal_projection

    def getVerticalProjectionProfile(image):
        binary = np.where(image > 0, 1, 0)
        vertical_projection = np.sum(binary, axis=0)
        return vertical_projection

    # Get horizontal projection profile
    h_projection = getHorizontalProjectionProfile(thresh)

    # Parameters for horizontal segmentation
    h_min_line_height = 10
    h_min_line_gap = 3
    h_threshold = np.mean(h_projection) * 0.3

    # Detect horizontal line boundaries
    h_line_boundaries = []
    in_line = False
    start_line = 0

    for i in range(len(h_projection)):
        if not in_line and h_projection[i] > h_threshold:
            in_line = True
            start_line = i
        elif in_line and (h_projection[i] <= h_threshold or i == len(h_projection) - 1):
            end_line = i
            if end_line - start_line >= h_min_line_height:
                h_line_boundaries.append((start_line, end_line))
            in_line = False

    # Merge horizontal lines that are very close together
    merged_h_lines = []
    i = 0
    while i < len(h_line_boundaries):
        current_start, current_end = h_line_boundaries[i]

        while i + 1 < len(h_line_boundaries) and h_line_boundaries[i+1][0] - current_end < h_min_line_gap:
            i += 1
            _, current_end = h_line_boundaries[i]

        merged_h_lines.append((current_start, current_end))
        i += 1

    # Find the largest horizontal segment based on height
    largest_h_segment = None
    largest_h_height = 0

    for start, end in merged_h_lines:
        height = end - start
        if height > largest_h_height:
            largest_h_height = height
            largest_h_segment = (start, end)

    # If no horizontal segments found, use the entire image
    if largest_h_segment is None:
        largest_h_segment = (0, img.shape[0])

    # Extract the largest horizontal segment with padding
    padding_h = 5
    y_start = max(0, largest_h_segment[0] - padding_h)
    y_end = min(img.shape[0], largest_h_segment[1] + padding_h)

    h_segment_img = img[y_start:y_end, :]
    h_segment_thresh = thresh[y_start:y_end, :]

    # Apply vertical projection to the largest horizontal segment
    v_projection = getVerticalProjectionProfile(h_segment_thresh)

    # Parameters for vertical segmentation
    v_min_column_width = 10
    v_min_column_gap = 3
    v_threshold = np.mean(v_projection) * 0.3

    # Detect vertical column boundaries
    v_column_boundaries = []
    in_column = False
    start_column = 0

    for i in range(len(v_projection)):
        if not in_column and v_projection[i] > v_threshold:
            in_column = True
            start_column = i
        elif in_column and (v_projection[i] <= v_threshold or i == len(v_projection) - 1):
            end_column = i
            if end_column - start_column >= v_min_column_width:
                v_column_boundaries.append((start_column, end_column))
            in_column = False

    # Merge vertical columns that are very close together
    merged_v_columns = []
    i = 0
    while i < len(v_column_boundaries):
        current_start, current_end = v_column_boundaries[i]

        while i + 1 < len(v_column_boundaries) and v_column_boundaries[i+1][0] - current_end < v_min_column_gap:
            i += 1
            _, current_end = v_column_boundaries[i]

        merged_v_columns.append((current_start, current_end))
        i += 1

    # Find the largest vertical segment based on width
    largest_v_segment = None
    largest_v_width = 0

    for start, end in merged_v_columns:
        width = end - start
        if width > largest_v_width:
            largest_v_width = width
            largest_v_segment = (start, end)

    # If no vertical segments found, use the entire width
    if largest_v_segment is None:
        largest_v_segment = (0, img.shape[1])

    # Extract the largest vertical segment with padding
    padding_v = 5
    x_start = max(0, largest_v_segment[0] - padding_v)
    x_end = min(img.shape[1], largest_v_segment[1] + padding_v)

    # Extract the final text region (intersection of largest horizontal and vertical segments)
    final_text_img = img[y_start:y_end, x_start:x_end]

    # Save the final text region with the specified filename
    cv2.imwrite(output_path, final_text_img)
    print(f"Extracted text region saved to {output_path}")
    print(f"Final text region dimensions: {final_text_img.shape}")

def super_resolve_image(input_image_path, output_folder):
    """
    Perform super-resolution on a single image using the specified model.
    
    Args:
    input_image_path (str): Full path to the input image
    output_folder (str): Folder where the super-resolved image will be saved
    """
    command = [
        "python", 
        r"C:\Users\91948\Desktop\final\super_res_mod.py",
        "-m", r"C:\Users\91948\Desktop\final\FSRCNN_x4.pb",
        "-i", input_image_path,
        "-o", output_folder
    ]
    
    try:
        # Run the super-resolution command
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode == 0:
            print(f"Successfully processed image: {input_image_path}")
        else:
            print(f"Error processing image: {input_image_path}")
            print(f"Error output: {result.stderr}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

class PalmLeafCharacterProcessor:
    def __init__(self, model_path, class_indices_path, ground_truth_dir):
        """
        Initialize the processor with model, class indices, and ground truth directory.
        
        Args:
            model_path (str): Path to the trained Keras model
            class_indices_path (str): Path to the class indices JSON file
            ground_truth_dir (str): Directory containing ground truth character images
        """
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        # Reverse the class indices dictionary
        self.class_names = {v: k for k, v in class_indices.items()}
        
        # Set image parameters
        self.IMG_SIZE = (64, 64)
        
        # Set ground truth directory
        self.ground_truth_dir = ground_truth_dir

    def preprocess_image(self, image_path):
        """
        Preprocess image for model prediction.
        
        Args:
            image_path (str): Path to the image
        
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.IMG_SIZE)
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(self, image_path):
        """
        Predict class for a single image.
        
        Args:
            image_path (str): Path to the image
        
        Returns:
            tuple: Predicted class name and confidence
        """
        preprocessed_img = self.preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = self.class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index]
        return predicted_class_name, confidence

    def preprocess_input_image(self, image_path):
        """
        Preprocess the input palm leaf image for character extraction.
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            tuple: Original image and binary image
        """
        # Read the input image
        input_image = cv2.imread(image_path)
        if input_image is None:
            raise ValueError(f"Error: Could not read the image at {image_path}")
        
        # Convert to grayscale
        grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        
        # Apply adaptive thresholding to create binary image
        binary_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15
        )
        
        return input_image, binary_image

    def refine_segmentation(self, binary_image):
        """
        Refine the binary image using morphological operations.
        
        Args:
            binary_image (numpy.ndarray): Input binary image
        
        Returns:
            numpy.ndarray: Refined binary image
        """
        # Create a small kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Dilate the image to connect nearby character components
        morphed_image = cv2.dilate(binary_image, kernel, iterations=1)
        
        return morphed_image

    def filter_components(self, binary_image):
        """
        Filter out small noise components based on area.
        
        Args:
            binary_image (numpy.ndarray): Input binary image
        
        Returns:
            numpy.ndarray: Filtered binary image
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        # Create an empty image for filtered components
        filtered_binary = np.zeros_like(binary_image)
        
        # Calculate average component area (excluding background)
        avg_area = np.mean(stats[1:, cv2.CC_STAT_AREA])
        
        # Filter components based on area
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > avg_area * 0.5:  # Remove small noise
                filtered_binary[labels == i] = 255
        
        return filtered_binary

    def extract_and_replace_characters(self, input_image, binary_image, 
                                       min_height=12, max_height=40, 
                                       min_width=12, max_width=55):
        """
        Extract character components, predict their class, and replace with ground truth.
        
        Args:
            input_image (numpy.ndarray): Original input image
            binary_image (numpy.ndarray): Binary image
            min_height (int): Minimum character height
            max_height (int): Maximum character height
            min_width (int): Minimum character width
            max_width (int): Maximum character width
        
        Returns:
            tuple: Image with replaced characters, number of replaced characters, and total contours meeting criteria
        """
        # Create a copy of the input image to modify
        output_image = input_image.copy()
        
        # Counters for replaced and total characters
        replaced_characters_count = 0
        total_contours_meeting_criteria = 0
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate through contours
        for idx, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the component meets size criteria
            if min_height <= h <= max_height and min_width <= w <= max_width:
                # Increment total contours meeting criteria
                total_contours_meeting_criteria += 1
                
                # Extract the character component
                character = input_image[y:y+h, x:x+w]
                
                # Use a context manager to create and remove a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, character)
                
                try:
                    # Predict the character
                    predicted_class, confidence = self.predict_image(temp_path)
                    
                    # Find ground truth image
                    ground_truth_path = os.path.join(self.ground_truth_dir, f"{predicted_class}.png")
                    
                    if os.path.exists(ground_truth_path):
                        # Read ground truth image
                        ground_truth_img = cv2.imread(ground_truth_path)
                        
                        # Resize ground truth to match original contour size
                        ground_truth_resized = cv2.resize(ground_truth_img, (w, h))
                        
                        # Replace the original contour with ground truth
                        output_image[y:y+h, x:x+w] = ground_truth_resized
                        
                        # Increment replaced characters count
                        replaced_characters_count += 1
                    
                    # Print prediction details
                    print(f"Contour {idx} - Predicted: {predicted_class} (Confidence: {confidence:.2f})")
                
                except Exception as e:
                    print(f"Error processing character: {e}")
                
                # Remove temporary file
                finally:
                    os.unlink(temp_path)
        
        # Print total contours meeting criteria
        print(f"Total contours meeting size criteria: {total_contours_meeting_criteria}")
        
        return output_image, replaced_characters_count, total_contours_meeting_criteria

    def process_palm_leaf(self, input_image_path, visualize=False):
        """
        Main method to process palm leaf image.
        
        Args:
            input_image_path (str): Path to the input palm leaf image
            visualize (bool): Whether to display visualization
        
        Returns:
            tuple: Processed image with replaced characters, 
                   number of replaced characters, 
                   and total contours meeting criteria
        """
        # Preprocess the image
        input_image, binary_image = self.preprocess_input_image(input_image_path)
        
        # Refine segmentation
        refined_image = self.refine_segmentation(binary_image)
        
        # Filter out noise
        filtered_image = self.filter_components(refined_image)
        
        # Extract and replace characters
        processed_image, num_replaced, total_contours = self.extract_and_replace_characters(
            input_image, filtered_image,
            min_height=30, max_height=170,
            min_width=30, max_width=170
        )
        
        # Visualize if requested
        if visualize:
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            plt.title("Processed Palm Leaf Characters")
            plt.axis('off')
            plt.show()
        
        return processed_image, num_replaced, total_contours

def main():
    # Base directory
    base_dir = r"C:\Users\Rahul\OneDrive\Desktop\New folder\final"
    
    # Input and output paths
    input_image = os.path.join(base_dir, "053.png")
    preprocessed_image = os.path.join(base_dir, "53_pre.png")
    cleaned_image = os.path.join(base_dir, "53_cleaned.png")
    extracted_image = os.path.join(base_dir, "53_extracted.jpg")
    super_resolved_image = os.path.join(base_dir, "53_extracted_fsrcnn_x4.jpg")
    processed_image = os.path.join(base_dir, "53_processed.png")

    # Configuration paths
    MODEL_PATH = os.path.join(base_dir, "palm_model.keras")
    CLASS_INDICES_PATH = os.path.join(base_dir, "class_indices.json")
    GROUND_TRUTH_DIR = os.path.join(base_dir, "ground_truth")

    # Image preprocessing
    preprocess_image(input_image, preprocessed_image)
    
    # Image cleaning
    clean_image(preprocessed_image, cleaned_image)
    
    # Extract text region
    extract_text_region(cleaned_image, extracted_image)
    
    # Super-resolve image
    super_resolve_image(extracted_image, base_dir)
    
    # Create processor
    processor = PalmLeafCharacterProcessor(
        MODEL_PATH, 
        CLASS_INDICES_PATH, 
        GROUND_TRUTH_DIR
    )

    # Process the palm leaf image
    processed_img, num_replaced, total_contours = processor.process_palm_leaf(
        super_resolved_image, 
        visualize=True
    )

    # Save the processed image
    cv2.imwrite(processed_image, processed_img)
    print(f"Number of characters replaced: {num_replaced}")
    print(f"Total contours meeting criteria: {total_contours}")
    print(f"Processed image saved to {processed_image}")

if __name__ == "__main__":
    main()