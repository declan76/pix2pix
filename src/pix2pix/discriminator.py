import tensorflow as tf

class Discriminator:
    """
    Discriminator class for a Generative Adversarial Network (GAN). 
    This class defines the architecture and loss function for the discriminator model.
    """
    
    # Binary cross-entropy loss object for the discriminator
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        """
        Downsample layer for the discriminator model.
        
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

    def build_model(self):
        """
        Builds the discriminator model architecture.
        """
        initializer = tf.random_normal_initializer(0.0, 0.02)

        # Input layers for the source and target images
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

        # Concatenate the source and target images
        x = tf.keras.layers.concatenate([inp, tar])

        # Downsample layers
        down1 = self.downsample(64, 4, False)(x)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)

        # Zero padding and convolutional layers
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv      = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

        # Batch normalization and activation layers
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2  = tf.keras.layers.ZeroPadding2D()(leaky_relu)

        # Final convolutional layer
        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        # Define the discriminator model
        self.model = tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        Computes the discriminator loss.
        
        Args:
        - disc_real_output (tf.Tensor): Discriminator's prediction on the real images.
        - disc_generated_output (tf.Tensor): Discriminator's prediction on the generated images.
        
        Returns:
        - tf.Tensor: Total discriminator loss.
        """
        real_loss       = Discriminator.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss  = Discriminator.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
