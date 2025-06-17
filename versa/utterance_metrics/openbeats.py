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

3. To compute similarity between two audio files:
    python openbeats.py --model_path <path_to_model> \
        --audio_path <path_to_audio> --compute_similarity \
        --reference_audio_path <path_to_reference_audio> --output_dir <output_directory>

Note: For predicting class, please use a fine-tuned checkpoint 
with the decoder.

Available checkpoints:
TODO(shikhar): Add list of checkpoints

"""

import torch
import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

# TODO(shikhar): after OpenBEATs is merged into ESPnet:
# 1. Switch the import to espnet.
# 2. Remove the model folder in versa
# PR: https://github.com/espnet/espnet/pull/6052
from versa.models.openbeats.encoder import BeatsEncoder
from versa.models.openbeats.decoder import LinearDecoder

# All checkpoints represent the best performing OpenBEATs
# checkpoint from TODO(shikhar): arxiv link1, link2.
OPENBEATS_CHECKPOINTS = {
    # Pretrained models
    "PT_MD_LARGE_iter1": "",  # Pretrained on multi-domain audio (bioacoustics included)
    "PT_MD_LARGE_iter2": "",  # Pretrained on multi-domain audio (bioacoustics included)
    "PT_MD_LARGE_iter3": "",  # Pretrained on multi-domain audio (bioacoustics included)
    "PT_AUDIO_LARGE_40K": "",  # Pretrained on 40k hours of audio
    "PT_SPEECH_AUDIO_LARGE_75K": "",  # Pretrained on 75k hours of speech + audio
    # Finetuned models (on top of PT_MD_LARGE)
    "FT_MD_LARGE_AS20K": "",  # Fine-tuned on AudioSet-20K from PT_MD_LARGE
    "FT_MD_LARGE_AS2M": "",  # Fine-tuned on AudioSet-2M from PT_MD_LARGE
    "FT_MD_LARGE_FSD50K": "",  # Fine-tuned on FSD50K from PT_MD_LARGE
    "FT_MD_LARGE_ESC": "",  # Fine-tuned on ESC-50 from PT_MD_LARGE
    # Fine-tuned on BEANS bioacoustics datasets
    "FT_MD_LARGE_WATKINS": "",
    "FT_MD_LARGE_CBI": "",
    "FT_MD_LARGE_HUMBUGDB": "",
    "FT_MD_LARGE_DOGS": "",
    "FT_MD_LARGE_BATS": "",
    "FT_MD_LARGE_DCASE21": "",
    "FT_MD_LARGE_RFCX": "",
    "FT_MD_LARGE_GIBBONS": "",
    "FT_MD_LARGE_HICEAS": "",
    "FT_MD_LARGE_ENABIRDS": "",
    # Fine-tuned on music-related datasets
    "FT_MD_LARGE_GTZAN_GENRE": "",
    "FT_MD_LARGE_NSYNTH_I": "",
    "FT_MD_LARGE_NSYNTH_P": "",
}


def validate_input_arguments(args):
    """Validate input arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If any required arguments are invalid or missing.
        FileNotFoundError: If required files don't exist.
    """
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    if args.compute_similarity:
        if args.reference_audio_path is None:
            raise ValueError(
                "Reference audio path must be specified when computing similarity."
            )
        if not os.path.exists(args.reference_audio_path):
            raise FileNotFoundError(
                f"Reference audio file not found: {args.reference_audio_path}"
            )

    if args.extract_embeddings and args.output_dir is None:
        raise ValueError(
            "Output directory must be specified when extracting embeddings."
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


def _get_ckpt_components(model_path, checkpoint_type=None):
    """Separate the components of the checkpoint.
    Args:
        model_path: Path or URL to the model checkpoint.
        checkpoint_type: Type of checkpoint, either 'pretrained' or 'finetuned'.
                       Detect automatically if not specified.
    Returns:
        Tuple containing:
            - encoder_state_dict: State dict for the encoder.
            - decoder_state_dict: State dict for the decoder (if exists).
            - cfg: ESPnet configuration of the BEATs model.
            - token_list: List of tokens (if exists).
    """
    model_url = None
    if model_path.startswith("http"):
        model_url = model_path
    elif model_path in OPENBEATS_CHECKPOINTS:
        model_url = OPENBEATS_CHECKPOINTS[model_path]
    if model_url:
        checkpoint = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
    else:
        checkpoint = torch.load(model_path, map_location="cpu")
    if checkpoint_type:
        is_pretrained = checkpoint_type == "pretrained"
    else:
        is_pretrained = (
            len(checkpoint.get("token_list", [])) == 0
        )  # because pretrained checkpoints do not have token_list
    if is_pretrained:
        # Pre-trained checkpoint BEATs style
        encoder_state_dict = checkpoint["model"]
        decoder_state_dict = None
        cfg = checkpoint["cfg"]
        token_list = None
    else:
        # Fine-tuned checkpoint ESPnet style
        encoder_state_dict = {
            k[len("encoder.") :]: v
            for k, v in checkpoint["model"].items()
            if k.startswith("encoder.")
        }
        decoder_state_dict = {
            k[len("decoder.") :]: v
            for k, v in checkpoint["model"].items()
            if k.startswith("decoder.")
        }
        cfg = checkpoint["cfg"]
        token_list = checkpoint["token_list"][:-2]  # Exclude <blank> and <unk> tokens
    return encoder_state_dict, decoder_state_dict, cfg, token_list


def openbeats_setup(
    model_path,
    use_gpu=False,
):
    device = "cuda" if use_gpu else "cpu"
    encoder_state_dict, decoder_state_dict, cfg, token_list = _get_ckpt_components(
        model_path
    )
    encoder = BeatsEncoder(
        input_size=1,
        beats_config=cfg,
    )
    encoder.load_state_dict(encoder_state_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    if decoder_state_dict is None:
        return encoder

    decoder = LinearDecoder(
        vocab_size=len(token_list),
        encoder_output_size=encoder.output_size(),
    )
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device)
    decoder.eval()
    return (encoder, decoder, token_list)


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
        model: OpenBEATs encoder, decoder and token_list.
        x: Input audio.
        fs: Sampling rate of the input audio.

    Returns:
        Dictionary with predicted classes and their probabilities.
    """
    assert isinstance(
        model, tuple
    ), "Model should be a tuple of encoder, decoder and token_list."
    encoder, decoder, token_list = model
    audio = _prepare_openbeats_input(x, fs, encoder)
    with torch.no_grad():
        ilens = torch.full(
            (audio.size(0),), audio.size(1), dtype=torch.long, device=audio.device
        )
        embedding, hlens, _ = encoder(xs_pad=audio, ilens=ilens, waveform_input=True)
        probs = decoder(hs_pad=embedding, hlens=hlens)
        class_probs = {k: v for k, v in zip(token_list, probs.to("cpu").numpy()[0])}

    return {"class_probabilities": class_probs}


def openbeats_embedding_extraction(model, x, fs, embedding_output_file=None):
    """Extract embeddings from audio.

    Args:
        model: OpenBEATs model.
        x: Input audio.
        fs: Sampling rate of the input audio.
        embedding_output_file: Path to save the extracted embeddings.

    Returns:
        Dictionary where value is the numpy file containing extracted embedding.
    """
    assert embedding_output_file is not None, "Output file path must be specified."
    audio = _prepare_openbeats_input(x, fs, model)
    with torch.no_grad():
        ilens = torch.full(
            (audio.size(0),), audio.size(1), dtype=torch.long, device=audio.device
        )
        embedding, _, _ = model(xs_pad=audio, ilens=ilens, waveform_input=True)

        if not os.path.exists(embedding_output_file):
            os.makedirs(os.path.dirname(embedding_output_file), exist_ok=True)
        with open(embedding_output_file, "wb") as f:
            np.save(f, embedding.to("cpu").numpy())

    return {"embedding_file": embedding_output_file}


def openbeats_embedding_similarity(model, x, ref_x):
    """Compute embedding similarity between input and reference audio.

    Args:
        model: OpenBEATs model.
        x: Input audio and sampling rate.
        ref_x: Reference audio and sampling rate.

    Returns:
        Dictionary where value is the similarity score between the embeddings.
    """
    audio1, fs1 = x
    audio2, fs2 = ref_x

    audio1_prepared = _prepare_openbeats_input(audio1, fs1, model)
    audio2_prepared = _prepare_openbeats_input(audio2, fs2, model)

    with torch.no_grad():
        ilens1 = torch.full(
            (audio1_prepared.size(0),),
            audio1_prepared.size(1),
            dtype=torch.long,
            device=audio1_prepared.device,
        )
        embedding1, _, _ = model(
            xs_pad=audio1_prepared, ilens=ilens1, waveform_input=True
        )
        ilens2 = torch.full(
            (audio2_prepared.size(0),),
            audio2_prepared.size(1),
            dtype=torch.long,
            device=audio2_prepared.device,
        )
        embedding2, _, _ = model(
            xs_pad=audio2_prepared, ilens=ilens2, waveform_input=True
        )

    embedding1_np = embedding1.to("cpu").numpy()
    embedding2_np = embedding2.to("cpu").numpy()

    # (batch_size, features)
    embedding1_flat = embedding1_np.reshape(embedding1_np.shape[0], -1)
    embedding2_flat = embedding2_np.reshape(embedding2_np.shape[0], -1)
    similarity_score = cosine_similarity(embedding1_flat, embedding2_flat)[0, 0]
    return {
        "similarity_score": similarity_score,
    }


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
    parser.add_argument(
        "--compute_similarity",
        action="store_true",
        help="Compute similarity between input and reference audio embeddings",
    )
    parser.add_argument(
        "--reference_audio_path",
        type=str,
        default=None,
        help="Path to reference audio file for embedding similarity",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory path to save output embeddings as npy files",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    validate_input_arguments(args)

    # Load model
    model = openbeats_setup(
        model_path=args.model_path,
        use_gpu=args.use_gpu,
    )

    # Load audio
    audio, fs = librosa.load(args.audio_path, sr=None, mono=False)

    # Infer
    if args.compute_similarity:
        ref_audio, ref_fs = librosa.load(args.reference_audio_path, sr=None, mono=False)
        similarity_result = openbeats_embedding_similarity(
            model, (audio, fs), (ref_audio, ref_fs)
        )
        print("Embedding Similarity Result: ", similarity_result)
    elif args.extract_embeddings:
        embedding_output_file = os.path.join(
            args.output_dir, os.path.basename(args.audio_path) + "_embedding.npy"
        )
        embedding = openbeats_embedding_extraction(
            model, audio, fs, embedding_output_file
        )
        print("Extracted Embedding: ", embedding)
    else:
        prediction = openbeats_class_prediction(model, audio, fs)
        print("Predicted Classes: ", prediction)
