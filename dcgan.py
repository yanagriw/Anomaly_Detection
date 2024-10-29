#!/usr/bin/env python3

import argparse
import datetime
import os
import re

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch
from mnist_v2 import MNIST

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
parser.add_argument("--train", default=False, action="store_true", help="Flag to train the model; otherwise, evaluation only.")
parser.add_argument("--model_save_path", default="gan_model", type=str, help="Path to save the trained model.")
parser.add_argument("--anomaly_threshold", default=0.1, type=float, help="Anomaly detection threshold.")


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)


# The GAN model
class GAN(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = torch.distributions.Normal(torch.zeros(args.z_dim), torch.ones(args.z_dim))

        self.generator = keras.Sequential([
            keras.layers.InputLayer(input_shape=(args.z_dim,)),
            keras.layers.Dense(1024),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(MNIST.H // 4 * MNIST.W // 4 * 64),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Reshape((MNIST.H // 4, MNIST.W // 4, 64)),
            keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2DTranspose(MNIST.C, kernel_size=4, strides=2, padding='same', activation='sigmoid')
        ])

        self.discriminator = keras.Sequential([
            keras.layers.InputLayer(input_shape=(MNIST.H, MNIST.W, MNIST.C)),
            keras.layers.Conv2D(32, kernel_size=5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(64, kernel_size=5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(1024),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.tb_callback = TorchTensorBoardCallback(args.logdir)

    # We override `compile`, because we want to use two optimizers.
    def compile(
            self, discriminator_optimizer: keras.optimizers.Optimizer, generator_optimizer: keras.optimizers.Optimizer,
            loss: keras.losses.Loss, metric: keras.metrics.Metric,
    ) -> None:
        super().compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss = loss
        self.metric = metric
        self.built = True

    def train_step(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.size(0)

        random_latent_vectors = self._z_prior.sample((batch_size,))
        generated_images = self.generator(random_latent_vectors, training=True)
        predictions_on_generated = self.discriminator(generated_images, training=True)
        generator_loss = self.loss(torch.ones_like(predictions_on_generated), predictions_on_generated)

        self.generator.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.apply_gradients(
            zip([variable.value.grad for variable in self.generator.trainable_variables],
                self.generator.trainable_variables))

        real_predictions = self.discriminator(images, training=True)
        fake_predictions = self.discriminator(generated_images.detach(), training=True)
        real_loss = self.loss(torch.ones_like(real_predictions), real_predictions)
        fake_loss = self.loss(torch.zeros_like(fake_predictions), fake_predictions)
        discriminator_loss = real_loss + fake_loss

        self.discriminator.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.apply_gradients(
            zip([variable.value.grad for variable in self.discriminator.trainable_variables],
                self.discriminator.trainable_variables))

        self.metric.update_state(torch.ones_like(real_predictions), real_predictions)
        self.metric.update_state(torch.zeros_like(fake_predictions), fake_predictions)

        self._loss_tracker.update_state(discriminator_loss + generator_loss)
        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
            **self.get_metrics_result(),
        }

    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(path, "generator.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(path, "discriminator.pth"))

    def load_model(self, path: str):
        self.generator.load_state_dict(torch.load(os.path.join(path, "generator.pth")))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, "discriminator.pth")))

    def detect_anomaly(self, images: torch.Tensor, threshold: float) -> torch.Tensor:
        # Detect anomalies using the discriminator's prediction
        scores = self.discriminator(images, training=False)
        return (scores < threshold).float()  # Anomaly if score is below the threshold

def simulate_anomalies(images: torch.Tensor) -> torch.Tensor:
    # Add Gaussian noise to simulate anomalies
    noise = torch.normal(mean=0, std=0.5, size=images.shape)
    noisy_images = images + noise
    return torch.clip(noisy_images, 0, 1)

def plot_simulated_anomalies(simulated_anomalies):
    # Plot the first four simulated anomalies
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(simulated_anomalies[i].cpu().numpy().squeeze(), cmap="gray")
        ax.axis("off")
    plt.suptitle("First 4 Simulated Anomalies")
    plt.show()

def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset, size={"train": args.train_size})
    train = mnist.train.transform(lambda example: example["image"] / 255)
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test = mnist.test.transform(lambda example: example["image"] / 255)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)

    # Create the network and train
    network = GAN(args)

    if args.train:
        network.compile(
            discriminator_optimizer=keras.optimizers.Adam(),
            generator_optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
            metric=keras.metrics.BinaryAccuracy("discriminator_accuracy"),
        )
        network.fit(train, epochs=args.epochs)
        network.save_model(args.model_save_path)
    else:
        network.load_model(args.model_save_path)

        # Simulate anomalies and evaluate
        normal_data = next(iter(test_loader))
        simulated_anomalies = simulate_anomalies(normal_data)
        print(f"Number of simulated anomalies: {len(simulated_anomalies)}")

        normal_anomalies_detected = network.detect_anomaly(normal_data, args.anomaly_threshold)
        simulated_anomalies_detected = network.detect_anomaly(simulated_anomalies, args.anomaly_threshold)

        print("Normal data - detected anomalies:", normal_anomalies_detected.sum().item())
        print("Simulated anomalies - detected anomalies:", simulated_anomalies_detected.sum().item())

        plot_simulated_anomalies(simulated_anomalies[:4])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)