#!/usr/bin/env python3

# Adapted from speaker similarity code for singer identity
# Uses SSL singer identity models from SonyCSLParis/ssl-singer-identity

import numpy as np
import torch

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def singer_model_setup(
    model_name="byol",
    model_path=None,
    use_gpu=False,
    input_sr=44100,
    torchscript=False,
    cache_dir="versa_cache/singer_identity",
):
    """
    Setup singer identity model

    Args:
        model_name (str): Pretrained model name such as 'byol',
            'contrastive', 'contrastive-vc', 'uniformity', or 'vicreg'.
        model_path (str): Path to local model (if None, downloads from HuggingFace)
        use_gpu (bool): Whether to use GPU
        input_sr (int): Input sample rate (will be upsampled to 44.1kHz if different)
        torchscript (bool): Whether to load torchscript version
        cache_dir (str): Directory for downloaded pretrained model files

    Returns:
        model: Loaded singer identity model
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        from singer_identity import load_model
    except ImportError:
        raise ImportError("Please run `install_ssl-singer-identity.sh` in tools.")

    if model_path is not None:
        # Load from local path
        model = load_model(
            model_path,
            source=model_path,
            input_sr=input_sr,
            torchscript=torchscript,
            savedir=cache_dir,
        )
    else:
        # Load from HuggingFace Hub
        model = load_model(
            model_name,
            input_sr=input_sr,
            torchscript=torchscript,
            savedir=cache_dir,
        )

    model = model.to(device)
    model.eval()
    return model


def singer_metric(model, pred_x, gt_x, fs, target_sr=44100):
    """
    Compute singer similarity between two audio signals

    Args:
        model: Singer identity model
        pred_x (np.ndarray): Generated/predicted audio signal
        gt_x (np.ndarray): Ground truth audio signal
        fs (int): Sample rate of input audio
        target_sr (int): Target sample rate (44.1kHz for singer models)

    Returns:
        dict: Dictionary containing singer similarity score
    """
    # Resample to target sample rate if needed (singer models expect 44.1kHz)
    if fs != target_sr:
        gt_x = resample_audio(gt_x, fs, target_sr)
        pred_x = resample_audio(pred_x, fs, target_sr)

    # Convert to torch tensors and add batch dimension
    device = next(model.parameters()).device

    # Ensure audio is in the right format (batch_size, n_samples)
    if pred_x.ndim == 1:
        pred_x = pred_x[np.newaxis, :]  # Add batch dimension
    if gt_x.ndim == 1:
        gt_x = gt_x[np.newaxis, :]  # Add batch dimension

    pred_tensor = torch.FloatTensor(pred_x).to(device)
    gt_tensor = torch.FloatTensor(gt_x).to(device)

    # Extract embeddings
    with torch.no_grad():
        embedding_pred = model(pred_tensor).squeeze(0).cpu().numpy()  # shape: (1000,)
        embedding_gt = model(gt_tensor).squeeze(0).cpu().numpy()  # shape: (1000,)

    # Compute cosine similarity
    similarity = np.dot(embedding_pred, embedding_gt) / (
        np.linalg.norm(embedding_pred) * np.linalg.norm(embedding_gt)
    )

    return {"singer_similarity": similarity}


def singer_metric_batch(model, audio_batch, fs, target_sr=44100):
    """
    Compute singer embeddings for a batch of audio signals

    Args:
        model: Singer identity model
        audio_batch (np.ndarray): Batch of audio signals (batch_size, n_samples)
        fs (int): Sample rate of input audio
        target_sr (int): Target sample rate (44.1kHz for singer models)

    Returns:
        np.ndarray: Singer embeddings (batch_size, 1000)
    """
    # Resample if needed
    if fs != target_sr:
        resampled_batch = []
        for i in range(audio_batch.shape[0]):
            resampled = resample_audio(audio_batch[i], fs, target_sr)
            resampled_batch.append(resampled)
        audio_batch = np.array(resampled_batch)

    # Convert to torch tensor
    device = next(model.parameters()).device
    audio_tensor = torch.FloatTensor(audio_batch).to(device)

    # Extract embeddings
    with torch.no_grad():
        embeddings = model(audio_tensor).cpu().numpy()  # shape: (batch_size, 1000)

    return embeddings


def compute_similarity_matrix(embeddings):
    """
    Compute pairwise cosine similarity matrix for a set of embeddings

    Args:
        embeddings (np.ndarray): Embeddings matrix (n_samples, embedding_dim)

    Returns:
        np.ndarray: Similarity matrix (n_samples, n_samples)
    """
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

    return similarity_matrix


class SingerMetric(BaseMetric):
    """Singer identity embedding cosine similarity."""

    def _setup(self):
        self.model_name = self.config.get("model_name", "byol")
        self.model_path = self.config.get("model_path")
        self.use_gpu = self.config.get("use_gpu", False)
        self.input_sr = self.config.get("input_sr", 44100)
        self.torchscript = self.config.get("torchscript", False)
        self.target_sr = self.config.get("target_sr", 44100)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/singer_identity")
        self.model = singer_model_setup(
            model_name=self.model_name,
            model_path=self.model_path,
            use_gpu=self.use_gpu,
            input_sr=self.input_sr,
            torchscript=self.torchscript,
            cache_dir=self.cache_dir,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return singer_metric(
            self.model,
            np.asarray(predictions),
            np.asarray(references),
            fs,
            target_sr=self.target_sr,
        )

    def get_metadata(self):
        return _singer_metadata()


def _singer_metadata():
    return MetricMetadata(
        name="singer",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["singer_identity", "librosa", "numpy", "torch"],
        description="Singer identity embedding cosine similarity",
        paper_reference="https://hal.science/hal-04186048v1",
        implementation_source="https://github.com/SonyCSLParis/ssl-singer-identity",
    )


def register_singer_metric(registry):
    """Register singer similarity with the registry."""
    registry.register(
        SingerMetric,
        _singer_metadata(),
        aliases=["singer_similarity", "singer_identity"],
    )


if __name__ == "__main__":
    # Example usage

    # Create some random audio data (normally you'd load real audio files)
    sample_rate = 44100
    duration = 4  # seconds
    n_samples = sample_rate * duration

    # Generate two random audio signals
    audio_a = np.random.random(n_samples) * 0.1  # Scale down to realistic audio range
    audio_b = np.random.random(n_samples) * 0.1

    print("Setting up singer identity model...")

    # Setup model (will download from HuggingFace on first use)
    try:
        model = SingerMetric({"model_name": "byol", "use_gpu": False})
        print("Model loaded successfully!")

        # Compute similarity between two audio signals
        print("Computing singer similarity...")
        result = model.compute(audio_a, audio_b, metadata={"sample_rate": sample_rate})
        print(f"Singer similarity: {result['singer_similarity']:.4f}")

        # Example of batch processing
        print("\nExample of batch processing...")
        audio_batch = np.array([audio_a, audio_b, np.random.random(n_samples) * 0.1])
        embeddings = singer_metric_batch(model, audio_batch, sample_rate)
        print(f"Embeddings shape: {embeddings.shape}")

        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(embeddings)
        print(f"Similarity matrix:\n{similarity_matrix}")

    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the singer_identity package:")
        print("pip install git+https://github.com/SonyCSLParis/ssl-singer-identity.git")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to network issues or missing dependencies.")
