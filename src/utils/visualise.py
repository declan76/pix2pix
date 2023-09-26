import os
import numpy as np
import matplotlib.pyplot as plt

def generate_images(model, test_input, tar, step, experiment_dir):
    prediction = model.model(test_input, training=True)
    plt.figure(figsize=(20, 20))

    display_list = [test_input[0], prediction[0], tar[0]]
    title = ['Input Image (T1)', 'Predicted Image (T2)', 'Ground Truth (T2)']

    for i in range(3):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    
    # Compute the error image
    error_image = np.abs(tar[0] - prediction[0])
    plt.subplot(2, 2, 4)
    plt.title('Error Image (Heatmap)')
    plt.imshow(error_image, cmap='hot')  # Using a heatmap style for better visualization
    plt.axis('off')
    
    save_path = os.path.join(experiment_dir, 'generated_images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, f'step_{step}.png'))
    plt.close()
