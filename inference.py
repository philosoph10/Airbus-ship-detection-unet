import cv2
import tensorflow as tf
import numpy as np
import argparse
from utils import dice


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")

    parser.add_argument("--model", default='./models/best_model/unet-model.keras', type=str, help="Path to the model")
    parser.add_argument("--image-src", required=True, type=str, help="Path to the source image file")
    parser.add_argument("--image-tgt", required=True, type=str, help="Path to the target image file")

    args_parsed = parser.parse_args()
    return args_parsed


if __name__ == '__main__':
    # parse command line arguments
    args = parse_arguments()

    model = tf.keras.models.load_model(args.model, custom_objects={'dice': dice})

    # Read the source image
    image = cv2.imread(args.image_src)

    # Prepare the image for the model
    image_ = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image_ = np.expand_dims(image_, axis=0) / 255.

    # Perform inference
    res = model.predict(image_)

    # Get the binary mask
    bin_mask = np.argmax(res[0], axis=2)

    bin_mask_expanded = cv2.resize(bin_mask, (768, 768), interpolation=cv2.INTER_NEAREST)

    # Color ship pixels into white
    image[bin_mask_expanded == 1] = [255, 255, 255]

    cv2.imwrite(args.image_tgt, image)
