from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A package for versatile evaluation of speech and audio"

setup(
    name="versa-speech-audio-toolkit",
    version="1.0.0",
    author="Jiatong Shi",
    author_email="ftshijt@gmail.com",
    description="A package for versatile evaluation of speech and audio",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wavlab-speech/versa.git",
    
    packages=find_packages(),
    python_requires=">=3.8",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    
    keywords=["speech", "audio", "metrics", "evaluation", "machine learning"],
    
    install_requires=[
        # Core ML and Deep Learning
        "torch",
        "torchaudio", 
        "transformers>=4.36.2",
        "accelerate",
        "huggingface-hub",
        "safetensors",
        "tokenizers",
        "einops",
        "opt-einsum",
        
        # Audio Processing
        "librosa",
        "soundfile",
        "audioread",
        "resampy",
        "torchlibrosa",
        "pyworld",
        "pysptk",
        
        # Speech and Audio Evaluation Metrics
        "pesq",
        "pystoi", 
        "mir-eval",
        "fast-bss-eval",
        "ci-sdr",
        "speechmos",
        
        # Text Processing and Distance Metrics
        "Levenshtein",
        "editdistance",
        "Distance",
        "rapidfuzz",
        "sentencepiece",
        
        # Scientific Computing
        "scikit-learn",
        "sympy",
        "threadpoolctl",
        
        # Configuration and Utilities
        "hydra-core",
        "omegaconf",
        "pyyaml",
        "protobuf",
        "python-dateutil",
        "lazy_loader",
        
        # Build and Compatibility
        "Cython",
        "setuptools",
        "importlib-metadata",
        "idna",
        
        # Optional/External Services
        "kaggle",
        "kaldiio",
        "fastdtw",
        "onnxruntime",
        
        # Git Dependencies - Speech/Audio Frameworks
        "espnet @ git+https://github.com/ftshijt/espnet.git@espnet_inference#egg=espnet",
        "espnet-tts-frontend",
        "espnet_model_zoo",
        "s3prl",
        
        # Git Dependencies - Audio Models
        # NOTE: Using latest commit for Python 3.13 compatibility
        "openai-whisper @ git+https://github.com/openai/whisper.git",
        
        # Git Dependencies - Evaluation Metrics
        "discrete-speech-metrics @ git+https://github.com/ftshijt/DiscreteSpeechMetrics.git@v1.0.2",
        # Additional Dependencies  
        "torch-complex",
        "cdpam",
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.3.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "versa-score=versa.bin.scorer:main",
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
)
