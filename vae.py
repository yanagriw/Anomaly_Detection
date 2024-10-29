#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import torch

from mnist_v2 import MNIST

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--decoder_layers", default=[500, 500], type=int, nargs="+", help="Decoder layers.")
parser.add_argument("--encoder_layers", default=[500, 500], type=int, nargs="+", help="Encoder layers.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
parser.add_argument("--anomaly_threshold", default=0.05, type=float, help="Anomaly detection threshold.")
parser.add_argument("--model_save_path", default="vae_model", type=str, help="Path to save the trained model.")
parser.add_argument("--train", default=False, action="store_true", help="Set this flag to train the model; otherwise, it will perform evaluation only.")


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


# The VAE model
class VAE(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.built = True

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = torch.distributions.Normal(torch.zeros(args.z_dim), torch.ones(args.z_dim))

        input_shape = (MNIST.H, MNIST.W, MNIST.C)
        encoder_inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Flatten()(encoder_inputs)
        for units in args.encoder_layers:
            x = keras.layers.Dense(units, activation='relu')(x)
        z_mean = keras.layers.Dense(args.z_dim)(x)
        z_log_var = keras.layers.Dense(args.z_dim)(x)
        z_log_var = keras.layers.Activation(lambda x: torch.exp(x))(z_log_var)
        self.encoder = keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var])

        decoder_inputs = keras.layers.Input(shape=(args.z_dim,))
        x = decoder_inputs
        for units in args.decoder_layers:
            x = keras.layers.Dense(units, activation='relu')(x)
        x = keras.layers.Dense(MNIST.H * MNIST.W * MNIST.C, activation='sigmoid')(x)
        decoder_outputs = keras.layers.Reshape((MNIST.H, MNIST.W, MNIST.C))(x)
        self.decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

        self.tb_callback = TorchTensorBoardCallback(args.logdir)

    def train_step(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.zero_grad()

        z_mean, z_sd = self.encoder(images, training=True)
        z_dist = torch.distributions.Normal(z_mean, z_sd)
        z = z_dist.rsample()
        reconstructed_images = self.decoder(z, training=True)
        reconstruction_loss = self.compute_loss(z, images, reconstructed_images)
        latent_loss = torch.distributions.kl.kl_divergence(z_dist, self._z_prior).mean()
        loss = reconstruction_loss * (MNIST.H * MNIST.W * MNIST.C) + latent_loss * self._z_dim
        loss.backward()
        self.optimizer.apply_gradients(zip([variable.value.grad for variable in self.trainable_variables], self.trainable_variables))

        self._loss_tracker.update_state(loss)
        return {"reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss, "loss": loss}

    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(path, "decoder.pth"))

    def load_model(self, path: str):
        self.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pth")))
        self.decoder.load_state_dict(torch.load(os.path.join(path, "decoder.pth")))

    def detect_anomaly(self, images: torch.Tensor) -> torch.Tensor:
        # Pass images through encoder and decoder
        z_mean, z_sd = self.encoder(images, training=False)
        z_dist = torch.distributions.Normal(z_mean, z_sd)
        z = z_dist.rsample()
        reconstructed_images = self.decoder(z, training=False)

        # Calculate reconstruction error and adjust axis to match dimensionality
        reconstruction_error = keras.losses.mean_squared_error(images, reconstructed_images)
        # Calculate mean across all axes except batch dimension
        reconstruction_error = reconstruction_error.mean(dim=list(range(1, reconstruction_error.ndim)))

        # Classify as anomaly if reconstruction error exceeds the threshold
        return (reconstruction_error > args.anomaly_threshold).float()

def simulate_anomalies(images: torch.Tensor) -> torch.Tensor:
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

def main(args: argparse.Namespace) -> float:
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
    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)

    # Create the network and train
    network = VAE(args)

    if args.train:
        network.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy())
        network.fit(train, epochs=args.epochs)

        network.save_model(args.model_save_path)

    else:
        network.load_model(args.model_save_path)

        normal_data = next(iter(test))
        simulated_anomalies = simulate_anomalies(normal_data)
        print(f"Number of simulated anomalies: {len(simulated_anomalies)}")

        normal_anomalies_detected = network.detect_anomaly(normal_data)
        simulated_anomalies_detected = network.detect_anomaly(simulated_anomalies)

        print("Normal data - detected anomalies:", normal_anomalies_detected.sum().item())
        print("Simulated anomalies - detected anomalies:", simulated_anomalies_detected.sum().item())

        plot_simulated_anomalies(simulated_anomalies[:4])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)