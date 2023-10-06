import tensorflow as tf

class Generator:
    """
    Generator class for a Generative Adversarial Network (GAN). 
    This class defines the architecture and loss function for the generator model.
    """
    
    # Number of output channels for the generator model
    OUTPUT_CHANNELS = 3
    
    # Weight for the L1 loss in the generator loss function
    LAMBDA = 100
    
    # Binary cross-entropy loss object for the generator
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        """
        Downsample layer for the generator model.
        
        Args:
        - filters (int): Number of filters for the convolutional layer.
        - size (int): Kernel size for the convolutional layer.
        - apply_batchnorm (bool, optional): Whether to apply batch normalization. Defaults to True.
        
        Returns:
        - tf.keras.Sequential: A sequential model containing the downsample layer.
        """
        initializer = tf.random_normal_initializer(0.0, 0.02)
        result      = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(
                filters,
                size,
                strides            = 2,
                padding            = "same",
                kernel_initializer = initializer,
                use_bias           = False,
            )
        )

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        """
        Upsample layer for the generator model.
        
        Args:
        - filters (int): Number of filters for the transposed convolutional layer.
        - size (int): Kernel size for the transposed convolutional layer.
        - apply_dropout (bool, optional): Whether to apply dropout. Defaults to False.
        
        Returns:
        - tf.keras.Sequential: A sequential model containing the upsample layer.
        """
        initializer = tf.random_normal_initializer(0.0, 0.02)
        result      = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides            = 2,
                padding            = "same",
                kernel_initializer = initializer,
                use_bias           = False,
            )
        )
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def build_model(self):
        """
        Builds the generator model architecture.
        """
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        # Define the downsample layers
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
        ]

        # Define the upsample layers
        up_stack = [
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),
            self.upsample(64, 4),
        ]

        # Final transposed convolutional layer
        initializer = tf.random_normal_initializer(0.0, 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS,
            4,
            strides            = 2,
            padding            = "same",
            kernel_initializer = initializer,
            activation         = "tanh",
        )

        x = inputs

        # Downsampling
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling with skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        # Define the generator model
        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def generator_loss(self, disc_generated_output, gen_output, target):
        """
        Computes the generator loss.
        
        Args:
        - disc_generated_output (tf.Tensor): Discriminator's prediction on the generated images.
        - gen_output (tf.Tensor): Generated images from the generator.
        - target (tf.Tensor): Real target images.
        
        Returns:
        - total_gen_loss (tf.Tensor): Total generator loss.
        - gan_loss (tf.Tensor): GAN loss component.
        - l1_loss (tf.Tensor): L1 loss component.
        """
        gan_loss       = Generator.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss        = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (Generator.LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss
