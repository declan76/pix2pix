
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont

def read_fits(file_path, file_path_str=None):
    """
    Read the data from a FITS (Flexible Image Transport System) file.

    Parameters:
    - file_path (str or bytes): Path to the FITS file.
    - file_path_str (str, optional): String representation of the file path. If not provided, 
                                     it will be derived from the file_path.

    Returns:
    - data (numpy.ndarray): Data contained in the FITS file.

    Raises:
    - ValueError: If no data is found in the FITS file.
    """
    if file_path_str is None:
        file_path_str = file_path if isinstance(file_path, str) else file_path.numpy().decode('utf-8')
    
    with fits.open(file_path_str) as hdul:
        data = hdul[0].data
        if data is None:
            print(50*"-")
            print(f"Error: No data in FITS file: {file_path_str}")
            raise ValueError
    
    return data


def generate_images(model, input_image_tensor, target_image_tensor, input_filename, image_path, mode="train"):
    """
    Generate and save a collage of input, predicted, target, and error images based on the provided tensors.

    Parameters:
    - model (object): The model used for prediction.
    - input_image_tensor (tensor): Tensor representation of the input image.
    - target_image_tensor (tensor): Tensor representation of the target image.
    - input_filename (str): Filename of the input image (used for naming the output collage).
    - image_path (str): Path to save the generated collage.
    - mode (str, optional): Mode of operation, either "train" or "eval". Default is "train".

    Returns:
    - str: Path to the saved collage image.

    Note:
    - The function first predicts the image using the provided model.
    - It then converts the tensors to image arrays and saves them.
    - A collage is created using the saved images, and labels are added to each section.
    - The saved individual images are then removed, and the path to the collage is returned.
    """
    predicted_image_tensor = model.model(input_image_tensor, training=True)
    
    input_image_array     = (input_image_tensor[0].numpy() * 0.5 + 0.5)
    predicted_image_array = (predicted_image_tensor[0].numpy() * 0.5 + 0.5)
    target_image_array    = (target_image_tensor[0].numpy() * 0.5 + 0.5)
    error_image_array     = np.abs(target_image_tensor[0].numpy() - predicted_image_tensor[0].numpy())
    
    tf.keras.preprocessing.image.save_img('input_image.png', input_image_array)
    tf.keras.preprocessing.image.save_img('predicted_image.png', predicted_image_array)
    tf.keras.preprocessing.image.save_img('target_image.png', target_image_array)
    tf.keras.preprocessing.image.save_img('error_image.png', error_image_array)
    
    input_image     = Image.open('input_image.png')
    predicted_image = Image.open('predicted_image.png')
    target_image    = Image.open('target_image.png')
    error_image     = Image.open('error_image.png')
    
    spacing        = 20
    label_height   = 20  
    collage_width  = 2 * input_image.width + spacing
    collage_height = 2 * input_image.height + 3 * label_height  # Added extra space for labels
    collage        = Image.new("RGB", (collage_width, collage_height), color=(255, 255, 255))
    
    collage.paste(input_image, (0, label_height))
    collage.paste(predicted_image, (input_image.width + spacing, label_height))
    collage.paste(target_image, (0, 2 * label_height + input_image.height))
    collage.paste(error_image, (input_image.width + spacing, 2 * label_height + input_image.height))
    
    draw = ImageDraw.Draw(collage)
    font = ImageFont.load_default()
    draw.text((10, 0), f"Input (T1)", font=font, fill=(0, 0, 0))
    draw.text((input_image.width + spacing + 10, 0), "Predicted (T2)", font=font, fill=(0, 0, 0))
    draw.text((10, label_height + input_image.height + 10), f"Ground Truth (T2)", font=font, fill=(0, 0, 0))
    draw.text((input_image.width + spacing + 10, label_height + input_image.height + 10), "Error", font=font, fill=(0, 0, 0))
    
    if mode == "train":
        filename  = f"step_{input_filename}.png"
        save_path = os.path.join(image_path, "generated_images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif mode == "eval":
        filename  = f"{input_filename}.png"
        save_path = image_path

    final_image_path = os.path.join(save_path, filename)
    collage.save(final_image_path)
    
    os.remove('input_image.png')
    os.remove('predicted_image.png')
    os.remove('target_image.png')
    os.remove('error_image.png')

    return final_image_path
