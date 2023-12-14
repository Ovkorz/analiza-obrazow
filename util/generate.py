import math
import numpy as np

class ImageObject:
    def __init__(self, image, bounding_box):
        self.image = image
        self.bounding_box = bounding_box  # Expected to be a tuple (x, y, width, height)

def pre_scale_images(background, object_collection, num_objects):
    # Calculate the average size for the objects based on the background size and the number of objects
    avg_size = (background.shape[0] * background.shape[1]) // num_objects

    # Go through the collection and pre-scale the images
    for i in range(len(object_collection)):
        # Calculate the scale factor based on the average size and the current object size
        object_size = object_collection[i].shape[0] * object_collection[i].shape[1]
        scale_factor = (avg_size / object_size) ** 0.5

        # Scale the object image
        if scale_factor < 1:
            scale_factor *= 0.8
            object_width = int(object_collection[i].shape[1] * scale_factor)
            object_height = int(object_collection[i].shape[0] * scale_factor)
            object_collection[i] = cv2.resize(object_collection[i], (object_width, object_height))

    return object_collection

def autocrop_image(image, margin=0, background_color=None):
    # Detect the background color by checking the color of the four corners of the image
    corner_colors = [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]
   
    if background_color is None:
        if all(np.all(color == [255, 255, 255]) for color in corner_colors):
            background_color = [255, 255, 255]  # white
        elif all(np.all(color == [0, 0, 0]) for color in corner_colors):
            background_color = [0, 0, 0]  # black
        else:
            raise ValueError('Unable to detect background color')

    # Find the bounding box of the non-background region
    mask = np.all(image != background_color, axis=-1)
    coords = np.argwhere(mask)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Crop the image
    cropped_image = image[x_min:x_max, y_min:y_max]

    # Add a margin to the image
    if margin > 0:
        cropped_image = cv2.copyMakeBorder(cropped_image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=background_color)

    return cropped_image

def detect_contours_gray(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get only white colors
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours in the mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def detect_contours_hsv(image):
    # Convert the image to HSV color space    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to remove small noise - opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for white color
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 255, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert the mask to get non-white colors
    mask = cv2.bitwise_not(mask)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def find_biggest_contour(contours):
    # Initialize maximum area and index
    max_area = 0
    max_index = -1

    # Iterate over all contours
    for i, contour in enumerate(contours):
        # Calculate area of contour
        area = cv2.contourArea(contour)

        # If this contour is bigger than the current biggest, update maximum
        if area > max_area:
            max_area = area
            max_index = i

    # Return the biggest contour
    if max_index != -1:
        return contours[max_index]
    else:
        return None

def filter_image(image):
    contours = detect_contours_gray(image)
    # contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the biggest contour
    biggest_contour = find_biggest_contour(contours)

    # Create an empty mask to draw the contour
    mask = np.zeros_like(image)

    # Draw the biggest contour on the mask
    cv2.drawContours(mask, [biggest_contour], -1, (255,255,255), thickness=cv2.FILLED)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(image, mask)

    return result

def create_shape(shape_type, size, shape_color: int):
    # Create a blank image with a white background
    # size = np.array(size, dtype=np.float64)
    image = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # # Set the color of the shape
    # if color == 'red':
    #     shape_color = (0, 0, 255)  # Red
    # elif color == 'green':
    #     shape_color = (0, 255, 0)  # Green
    # elif color == 'blue':
    #     shape_color = (255, 0, 0)  # Blue
    # else:
    #     shape_color = (0, 0, 0)  # Black
    
    # Draw the shape on the image
    if shape_type == 'circle':
        center = (size // 2, size // 2)
        radius = size // 2
        cv2.circle(image, center, radius, shape_color, -1)
    elif shape_type == 'square':
        top_left = (0, 0)
        bottom_right = (size, size)
        cv2.rectangle(image, top_left, bottom_right, shape_color, -1)
    elif shape_type == 'triangle':
        points = np.array([[size // 2, 0], [0, size], [size, size]], np.int32)
        cv2.fillPoly(image, [points], shape_color)
    
    return image

def create_compound_shape(size, num_shapes=None, color_pallette = None):
    # Create a blank image with a white background
    image = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Generate a random number of shapes
    if num_shapes is None:
        num_shapes = np.random.randint(2, 5)
    
    # Generate a random color for each shape
    if color_pallette is None:
        color_pallette = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    num_colors = len(color_pallette)
    random_indices = random.sample(range(num_colors), min(num_shapes, num_colors))
    shape_colors = np.array(color_pallette, dtype=np.uint8)[random_indices]
    
    # Create each shape and paste it on the image
    for i in range(num_shapes):
        # Generate a random shape type
        shape_type = np.random.choice(['circle', 'square', 'triangle'])
        
        # Generate a random size for the shape
        shape_size = np.random.randint(size // 3, size // 1.5)
        
        # Generate a random location for the shape
        x = np.random.randint(0, size - shape_size)
        y = np.random.randint(0, size - shape_size)
        
        # Create the shape
        # shape = create_shape(shape_type, shape_size, shape_colors[i])
        shape_color = tuple([int(j) for j in shape_colors[i]])
        # shape_color = shape_colors[i]

        # Draw the shape on the image
        if shape_type == 'circle':
            center = (x + shape_size // 2, y + shape_size // 2)
            radius = shape_size // 2
            cv2.circle(image, center, radius, shape_color, -1)
        elif shape_type == 'square':
            top_left = (x, y)
            bottom_right = (x + shape_size, y + shape_size)
            cv2.rectangle(image, top_left, bottom_right, shape_color, -1)
        elif shape_type == 'triangle':
            points = np.array([[(2*x + shape_size) // 2, y], [ x, y + shape_size ], [x + shape_size, y + shape_size]], np.int32)
            cv2.fillPoly(image, [points], shape_color)


    return image

def plot_collection(collection, cv2_color_scheme=True):
    # Calculate the number of rows and columns based on the number of images
    collection = collection.copy()
    if cv2_color_scheme:   
        for i in range(len(collection)):
            collection[i] = cv2.cvtColor(collection[i], cv2.COLOR_BGR2RGB)

    num_images = len(collection)
    num_rows = min(int(math.ceil(num_images / 5)), 3)
    num_cols = min(5, num_images)
    
    # Calculate the figsize based on the number of columns
    figsize = (min(num_cols * 4, 20), num_rows * 4)
    
    # Create a figure to display the images
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    # Plot the images
    for i in range(num_images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(collection[i])
    
        contours = detect_contours_hsv(collection[i])
        # Display the number of contours on the image
        ax.set_title(f'Contours: {len(contours)}')


    # Show the plot
    plt.show()


def boxes_overlap(box1, box2):
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError('Boxes must contain 4 coordinates: x,y,w,h')
   
    # Extract the coordinates from the boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    return not (x1 + w1 < x2
                or x1 > x2 + w2
                or y1 + h1 < y2
                or y1 > y2 + h2)
   
    # Check if the boxes overlap along the x-axis
# def intersects(self, other):
#     return not (self.top_right.x < other.bottom_left.x
#                 or self.bottom_left.x > other.top_right.x
#                 or self.top_right.y < other.bottom_left.y
#                 or self.bottom_left.y > other.top_right.y)


