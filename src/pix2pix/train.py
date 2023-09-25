import time
import tensorflow as tf
from IPython import display
from prettytable import PrettyTable
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator
from utils.visualise import generate_images

class Trainer:
    def __init__(self, generator: Generator, discriminator: Discriminator, summary_writer, checkpoint_prefix):
        self.generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator               = generator
        self.discriminator           = discriminator
        self.summary_writer          = summary_writer
        self.checkpoint_prefix       = checkpoint_prefix
        self.checkpoint              = tf.train.Checkpoint(
            generator_optimizer     = self.generator_optimizer,
            discriminator_optimizer = self.discriminator_optimizer,
            generator               = self.generator.model,
            discriminator           = self.discriminator.model)
            
        # Initialize loss attributes
        self.gen_total_loss = None
        self.gen_gan_loss   = None
        self.gen_l1_loss    = None
        self.disc_loss      = None


    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator.model(input_image, training=True)

            disc_real_output      = self.discriminator.model([input_image, target], training=True)
            disc_generated_output = self.discriminator.model([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients     = gen_tape.gradient(gen_total_loss, self.generator.model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.model.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


    def fit(self, train_ds, test_ds, steps, experiment_dir):
        _, example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        try:
            for step, (_, input_image, target) in train_ds.repeat().take(steps).enumerate():
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target, step)
                if (step) % 1000 == 0:
                    display.clear_output(wait=True)

                    # Extracting the losses for printing
                    gen_total_loss_value = gen_total_loss.numpy()
                    gen_gan_loss_value   = gen_gan_loss.numpy()
                    gen_l1_loss_value    = gen_l1_loss.numpy()
                    disc_loss_value      = disc_loss.numpy()

                    # Time taken for the last 1k steps
                    time_taken = time.time() - start
                    start      = time.time()

                    # Create a table using PrettyTable
                    table = PrettyTable()
                    table.field_names = ["Step", "Metric", "Value"]
                    table.add_row(["", "Time taken for last 1k steps", f"{time_taken:.2f} sec"])
                    table.add_row(["", "Generator Total Loss", f"{gen_total_loss_value:.4f}"])
                    table.add_row([f"{step//1000}k", "Generator GAN Loss", f"{gen_gan_loss_value:.4f}"])
                    table.add_row(["", "Generator L1 Loss", f"{gen_l1_loss_value:.4f}"])
                    table.add_row(["", "Discriminator Loss", f"{disc_loss_value:.4f}"])

                    print(table)

                    # Call the generate_images function with the run_timestamp
                    generate_images(self.generator, example_input, example_target, step, experiment_dir)

                # Training step
                if (step+1) % 10 == 0:
                    print('.', end='', flush=True)

                # Save the model every 5k steps
                if (step + 1) % 5000 == 0 or (step + 1) == steps:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current progress...")
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print("Progress saved. Exiting now.")

        except Exception as e:
            print(f"Error encountered at step {step}.")
            print(f"Input Image Shape: {input_image.shape}")
            print(f"Target Shape: {target.shape}")
            raise e  # re-raise the exception to see the traceback
