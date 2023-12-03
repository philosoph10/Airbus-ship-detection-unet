import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def unet_model(input_shape):
    """
    define the unet model architecture
    :param input_shape: shape of the input, including the number of channels, like [128, 128, 3]
    :return: the model
    """
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the encoder
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    # Permit fine-tuning
    down_stack.trainable = True

    # Define the decoder
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    # Define the model flow
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Run the encoder
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Run the decoder and establish skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # Define the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same')

    x = last(x)
    x = tf.keras.activations.softmax(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def dice(targets, inputs, smooth=1e-6):
    """
    calculate the Dice-Sorensen coefficient
    :param targets: the model's predictions
    :param inputs: the ground-truth labels
    :param smooth: smoothing parameter, to prevent division by 0
    :return: the metric value
    """
    axis = [1, 2, 3]
    intersection = tf.reduce_sum(targets * inputs, axis=axis)

    # Add smooth parameter to avaoid division by 0
    dice_coef = (2 * intersection + smooth) / (tf.reduce_sum(targets, axis=axis) +
                                               tf.reduce_sum(inputs, axis=axis) + smooth)
    return dice_coef
