import tensorflow as tf
import time
from IPython import display
from models.generator import Generator
from models.discriminator import Discriminator
from evaluation.visualise import generate_images

class Trainer:
    def __init__(self, generator: Generator, discriminator: Discriminator, summary_writer, checkpoint_prefix):
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator = generator
        self.discriminator = discriminator
        self.summary_writer = summary_writer
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator.model,
                                              discriminator=self.discriminator.model)

    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator.model(input_image, training=True)

            disc_real_output = self.discriminator.model([input_image, target], training=True)
            disc_generated_output = self.discriminator.model([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.model.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

    def fit(self, train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                generate_images(self.generator, example_input, example_target)
                print(f"Step: {step//1000}k")

            self.train_step(input_image, target, step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
