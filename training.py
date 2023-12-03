import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from utils import rle_to_mask, unet_model, dice
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")

    # Add command line arguments
    parser.add_argument("--checkpoint-dir", default='./models/checkpoint', type=str, help="Directory to save model checkpoints")
    parser.add_argument("--num-epochs", default=3, type=int, help="Number of epochs to train the model")
    parser.add_argument("--reduce-dataset", default=True, type=bool, help="Whether to use all data or reduce")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size")
    parser.add_argument("--model-save-dir", default=None, type=str, help="Directory to save the final model")

    # Parse the command line arguments
    args_parsed = parser.parse_args()

    return args_parsed


# noinspection PyShadowingNames
def preprocess_data(batch_size=16, reduce_dataset=True):
    """
    prepare images for training
    :param batch_size: batch size
    :param reduce_dataset: whether to keep all images, or a subset (10k ship images and 2k non-ship images)
    :return: a tuple of training, validation, and test batches, and the train size
    """
    def one_hot(a, num_classes):
        """
        convert labels to one-hot
        :param a: a list of labels
        :param num_classes: the number of possible classes
        :return: one-hot encoded labels
        """
        return np.squeeze(np.eye(num_classes)[a])

    def load_train_image(tensor):
        """
        load a single image
        :param tensor: the tensor with image path
        :return: a tuple of the input image, its mask tensor, and the sample weights
        """
        # Extract the path from the tensor
        path = tf.get_static_value(tensor).decode("utf-8")

        # Extract image ID from the path
        image_id = os.path.basename(path)

        # Read the input image, resize, and normalize
        input_image = cv2.imread(path)
        input_image = tf.image.resize(input_image, (256, 256))
        input_image = tf.cast(input_image, tf.float32) / 255.0

        # Extract encoded mask from the DataFrame
        encoded_mask = ships[ships['ImageId'] == image_id].iloc[0]['EncodedPixels']

        # Create an input mask (binary) from the encoded mask
        input_mask = np.zeros((256, 256) + (1,), dtype=np.int8)
        if not pd.isna(encoded_mask) and encoded_mask != '':
            input_mask = rle_to_mask(encoded_mask)
            input_mask = cv2.resize(input_mask, (256, 256), interpolation=cv2.INTER_AREA)
            input_mask = np.expand_dims(input_mask, axis=2)

        # Convert the binary mask to one-hot encoding
        one_hot_segmentation_mask = one_hot(input_mask, 2)
        input_mask_tensor = tf.convert_to_tensor(one_hot_segmentation_mask, dtype=tf.float32)

        # Define class weights and calculate sample weights
        class_weights = tf.constant([0.001, 1. - 0.001], tf.float32)
        sample_weights = tf.gather(class_weights, indices=tf.cast(input_mask_tensor, tf.int32),
                                   name='cast_sample_weights')

        return input_image, input_mask_tensor, sample_weights

    # Read the dataframe
    ships = pd.read_csv("./data/train_ship_segmentations_v2.csv")
    ships['EncodedPixels'] = ships['EncodedPixels'].astype('string')

    # Remove invalid image
    ships = ships[ships['ImageId'] != '6384c3e78.jpg']

    # Aggregate encoded pixels for all ships in the image
    ships = ships.fillna({'EncodedPixels': ''})
    ships = ships.groupby('ImageId')['EncodedPixels'].agg(' '.join).reset_index()

    # Divide the data into ship images and non-ship images
    ships_images = ships.loc[ships['EncodedPixels'] != '', 'ImageId']
    non_ships_images = ships.loc[ships['EncodedPixels'] == '', 'ImageId']

    # Sample a subset of images, up to 10k (2k) if available, otherwise take all
    ships_images = ships_images.sample(min(10000, len(ships_images))).tolist()
    non_ships_images = non_ships_images.sample(min(2000, len(non_ships_images))).tolist()

    images_list = np.append(non_ships_images, ships_images)

    if not reduce_dataset:
        images_list = np.array(list(os.listdir('./data/train_v2')))

    # Prepare datasets
    images_list = tf.data.Dataset.list_files([f'./data/train_v2/{name}' for name in images_list], shuffle=True)
    train_images = images_list.map(lambda x: tf.py_function(load_train_image, [x], [tf.float32, tf.float32]),
                                   num_parallel_calls=tf.data.AUTOTUNE)

    val_test_len = int(len(images_list) / 6.)

    print(f'Dataset size:\n\t#train = {len(images_list)-2*val_test_len}\n\t#val = '
          f'{val_test_len}\n\t#test = {val_test_len}')

    # Break the dataset into train, validation, and test parts
    validation_dataset = train_images.take(val_test_len)
    test_dataset = train_images.skip(val_test_len).take(val_test_len)
    train_dataset = train_images.skip(2*val_test_len)

    # Prepare batches
    train_batches = (train_dataset.repeat().batch(batch_size))

    validation_batches = validation_dataset.batch(batch_size)

    test_batches = test_dataset.batch(batch_size)

    return train_batches, validation_batches, test_batches, len(images_list)-2*val_test_len


if __name__ == '__main__':
    # parse command line arguments
    args = parse_arguments()

    # Prepare the data for training
    train_batches, validation_batches, test_batches, train_size = preprocess_data(batch_size=args.batch_size,
                                                                                  reduce_dataset=args.reduce_dataset)

    # Define the model
    model = unet_model(input_shape=[256, 256, 3])

    # Compile the model with cross-entropy loss and Adam optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice])

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # create a checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        args.checkpoint_dir,
        monitor='val_dice',  # Monitoring validation dice coefficient
        save_best_only=True,
        mode='max',  # 'max' for dice coefficient, 'min' for loss, depending on what you are monitoring
        verbose=1
    )

    # calculate the number of steps per epoch
    steps_per_epoch = train_size // args.batch_size

    model.fit(train_batches,
              epochs=args.num_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=validation_batches,
              callbacks=[checkpoint_callback]
              )

    if args.model_save_dir is not None:
        os.makedirs(args.model_save_dir, exist_ok=True)
        full_path = os.path.join(args.model_save_dir, 'model.keras')
        model.save(full_path)
