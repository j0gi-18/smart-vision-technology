import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Split the image into its respective color channels (B, G, R)
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # Function to preprocess a single channel (normalization, equalization, unsharp masking)
    def preprocess_channel(channel):
        # Normalize the channel
        normalized_channel = cv2.normalize(channel, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Apply histogram equalization
        equalized_channel = cv2.equalizeHist((normalized_channel * 255).astype(np.uint8))
        
        # Apply unsharp masking
        blurred_channel = cv2.GaussianBlur(equalized_channel, (9, 9), 10.0)
        unsharp_masked_channel = cv2.addWeighted(equalized_channel, 1.5, blurred_channel, -0.5, 0)
        
        return unsharp_masked_channel
    
    # Apply preprocessing to each channel
    processed_b = preprocess_channel(b_channel)
    processed_g = preprocess_channel(g_channel)
    processed_r = preprocess_channel(r_channel)
    
    # Merge the processed channels back into a color image
    processed_image = cv2.merge([processed_b, processed_g, processed_r])
    
    return processed_image

def display_comparison(original_image, processed_image):
    # Plot the original and preprocessed images side by side
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Preprocessed image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Preprocessed Image')
    plt.axis('off')

    # Show the images
    plt.show()

# Load an image
image = cv2.imread("Aashirvaad Aaata.jpg")

# Preprocess the image
processed_image = preprocess_image(image)

# Display comparison
display_comparison(image, processed_image)
