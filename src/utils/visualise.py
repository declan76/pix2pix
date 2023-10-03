import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

def generate_images(model, input_image_tensor, target_image_tensor, input_filename, image_path, mode="train"):
    predicted_image_tensor = model.model(input_image_tensor, training=True)
    
    # Convert tensors to numpy arrays for saving
    input_image_array     = (input_image_tensor[0].numpy() * 0.5 + 0.5)
    predicted_image_array = (predicted_image_tensor[0].numpy() * 0.5 + 0.5)
    target_image_array    = (target_image_tensor[0].numpy() * 0.5 + 0.5)
    error_image_array     = np.abs(target_image_tensor[0].numpy() - predicted_image_tensor[0].numpy())
    
    # Save each image using TensorFlow's save_img method
    tf.keras.preprocessing.image.save_img('input_image.png', input_image_array)
    tf.keras.preprocessing.image.save_img('predicted_image.png', predicted_image_array)
    tf.keras.preprocessing.image.save_img('target_image.png', target_image_array)
    tf.keras.preprocessing.image.save_img('error_image.png', error_image_array)
    
    # Load the saved images using PIL
    input_image     = Image.open('input_image.png')
    predicted_image = Image.open('predicted_image.png')
    target_image    = Image.open('target_image.png')
    error_image     = Image.open('error_image.png')
    
    # Define spacing and adjust collage dimensions
    spacing        = 20
    label_height   = 20  
    collage_width  = 2 * input_image.width + spacing
    collage_height = 2 * input_image.height + 3 * label_height  # Added extra space for labels
    collage        = Image.new("RGB", (collage_width, collage_height), color=(255, 255, 255))
    
    # Paste images with spacing
    collage.paste(input_image, (0, label_height))
    collage.paste(predicted_image, (input_image.width + spacing, label_height))
    collage.paste(target_image, (0, 2 * label_height + input_image.height))
    collage.paste(error_image, (input_image.width + spacing, 2 * label_height + input_image.height))
    
    # Add labels below images
    draw = ImageDraw.Draw(collage)
    font = ImageFont.load_default()
    draw.text((10, 0), f"Input (T1)", font=font, fill=(0, 0, 0))
    draw.text((input_image.width + spacing + 10, 0), "Predicted (T2)", font=font, fill=(0, 0, 0))
    draw.text((10, label_height + input_image.height + 10), f"Ground Truth (T2)", font=font, fill=(0, 0, 0))
    draw.text((input_image.width + spacing + 10, label_height + input_image.height + 10), "Error", font=font, fill=(0, 0, 0))
    
    # Decide the filename based on the mode
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
    
    # Optionally, remove the individual images if you don't need them
    os.remove('input_image.png')
    os.remove('predicted_image.png')
    os.remove('target_image.png')
    os.remove('error_image.png')

    return final_image_path
