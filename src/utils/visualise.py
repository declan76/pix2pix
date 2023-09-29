import os
import numpy as np
import matplotlib.pyplot as plt


def generate_images(model, test_input, tar, name, image_path, mode="train"):
    prediction = model.model(test_input, training=True)
    plt.figure(figsize=(20, 20))

    display_list = [test_input[0], prediction[0], tar[0]]
    title = ["Input Image (T1)", "Predicted Image (T2)", "Ground Truth (T2)"]

    for i in range(3):
        # Normalize and clip the image values to [0, 1]
        img = np.clip(display_list[i] * 0.5 + 0.5, 0, 1)

        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        plt.imshow(img)
        plt.axis("off")

    # Compute the error image
    error_image = np.abs(tar[0] - prediction[0])
    # Normalize and clip the error image values to [0, 1]
    error_image = np.clip(error_image, 0, 1)

    plt.subplot(2, 2, 4)
    plt.title("Error Image (Heatmap)")
    plt.imshow(
        error_image, cmap="hot"
    )  # Using a heatmap style for better visualization
    plt.axis("off")

    # Decide the filename based on the mode
    if mode == "train":
        filename = f"step_{name}.png"
        save_path = os.path.join(image_path, "generated_images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif mode == "eval":
        filename = f"{name}.png"
        save_path = image_path

    plt.savefig(os.path.join(save_path, filename))
    plt.close()
