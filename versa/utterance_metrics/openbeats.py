"""Script to run inference with OpenBEATs models.

This script allows prediction of sound event/class, 
and extraction of embeddings from 
any layer using OpenBEATs models.

Usage:
1. To predict sound events/classes:
    python openbeats.py --model_path <path_to_model> \
        --audio_path <path_to_audio>

2. To extract embeddings:
    python openbeats.py --model_path <path_to_model> \
        --audio_path <path_to_audio> --extract_embeddings
Note: For predicting class, please use a fine-tuned checkpoint 
with the decoder.

Available checkpoints:
TODO(shikhar): Add list of checkpoints

"""

import torch

import numpy as np

# TODO(shikhar): after OpenBEATs is merged into ESPnet:
# 1. Switch the import to espnet.
# 2. Remove the model folder in versa
# PR: https://github.com/espnet/espnet/pull/6052
from versa.models.openbeats.encoder import BeatsEncoder
from versa.models.openbeats.decoder import LinearDecoder


def _maybe_load_decoder(model_path, encoder, model_config):
    """Load the decoder if it exists in the model checkpoint.

    Args:
        model_path: Path to the model checkpoint.
        encoder: Encoder model.
        model_config: Model configuration.

    Returns:
        Decoder model if it exists, otherwise None.
    """
    decoder = None
    weights = torch.load(model_path, map_location="cpu")
    if "linear_decoder" in weights:
        weights = weights["linear_decoder"]
        n_classes = weights.shape[0]
        decoder = LinearDecoder(
            vocab_size=n_classes,
            encoder_output_size=encoder.output_size(),
        )
        decoder.load_state_dict(weights)
    return decoder


def openbeats_setup(
    model_path,
    use_gpu=False,
):
    device = "cuda" if use_gpu else "cpu"
    encoder = BeatsEncoder(
        input_size=1,
        beats_ckpt_path=model_path,
    )
    encoder.to(device)
    encoder.eval()

    decoder = _maybe_load_decoder(model_path, encoder, encoder.config)
    if decoder is not None:
        decoder.to(device)
        decoder.eval()
        return (encoder, decoder)
    return encoder


def _prepare_openbeats_input(audio, fs, model):
    """Prepare input audio for model inference.

    Args:
        audio: Input audio.
        fs: Sampling rate of the input audio.

    Returns:
        Tensor containing the audio data.
    """
    if fs != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    device = next(model.parameters()).device
    audio = torch.tensor(audio, dtype=torch.float32).to(device)
    return audio


def openbeats_class_prediction(model, x, fs):
    """Predict sound events/classes from audio.

    Args:
        model: OpenBEATs model.
        x: Input audio.
        fs: Sampling rate of the input audio.

    Returns:
        Dictionary with predicted classes and their probabilities.
    """
    assert isinstance(model, tuple), "Model should be a tuple of encoder and decoder"
    encoder, decoder = model
    audio = _prepare_openbeats_input(x, fs, encoder)
    with torch.no_grad():
        ilens = torch.full(
            (audio.size(0),), audio.size(1), dtype=torch.long, device=audio.device
        )
        embedding, hlens, _ = encoder(xs_pad=audio, ilens=ilens, waveform_input=True)
        probs = decoder(hs_pad=embedding, hlens=hlens)

    return {"class_probabilities": probs}


def openbeats_embedding_extraction(model, x, fs):
    """Extract embeddings from audio.

    Args:
        model: OpenBEATs model.
        x: Input audio.
        fs: Sampling rate of the input audio.

    Returns:
        Dictionary where value is the extracted embedding.
    """
    audio = _prepare_openbeats_input(x, fs, model)
    with torch.no_grad():
        ilens = torch.full(
            (audio.size(0),), audio.size(1), dtype=torch.long, device=audio.device
        )
        embedding, _, _ = model(xs_pad=audio, ilens=ilens, waveform_input=True)

    return {"embedding": embedding.to("cpu").numpy()}


if __name__ == "__main__":
    import argparse
    import librosa

    parser = argparse.ArgumentParser(description="Run OpenBEATs inference.")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file")
    parser.add_argument(
        "--extract_embeddings",
        action="store_true",
        help="Extract embeddings instead of predicting classes",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    # Load model
    model = openbeats_setup(
        model_path=args.model_path,
        use_gpu=args.use_gpu,
    )
    # Load audio
    audio, fs = librosa.load(args.audio_path, sr=None, mono=False)
    # Infer
    if args.extract_embeddings:
        embedding = openbeats_embedding_extraction(model, audio, fs)
        print("Extracted Embedding: ", embedding)
    else:
        prediction = openbeats_class_prediction(model, audio, fs)
        print("Predicted Classes: ", prediction)
