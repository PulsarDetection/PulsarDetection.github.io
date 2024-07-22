import numpy as np
import matplotlib.pyplot as plt
import os

def load_npz(file_path):
    data = np.load(file_path)
    return data['images'], data['labels']

def save_image(image, output_dir, index):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image as a PNG file
    file_path = os.path.join(output_dir, f'image_{index:04d}.png')
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')  # Hide axes
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to avoid memory issues

if __name__ == '__main__':
    images, labels = load_npz('pulsar_final.npz')
    
    output_dir = 'saved_images'  # Directory to save the images
    num_images = 5  # Number of images to save
    for i in range(len(images)):
        save_image(images[i], output_dir, i)
        # break  # Remove this line to save all images
        if i == num_images - 1:
            break

    print(images[0].shape)
