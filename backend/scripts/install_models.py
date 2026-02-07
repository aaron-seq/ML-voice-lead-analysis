#!/usr/bin/env python3
"""
ML Voice Lead Analysis - ML Model Installation Script

This script downloads and installs required machine learning models:
- spaCy language models for NLP processing
- Additional ML models as needed

Usage:
    python scripts/install_models.py
    python scripts/install_models.py --model en_core_web_sm
    python scripts/install_models.py --force
"""

import sys
import argparse
import logging
from pathlib import Path

try:
    import spacy
    from spacy.cli import download as spacy_download
except ImportError:
    print("Error: spaCy is not installed. Please install it first:")
    print("  pip install spacy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default models to install
DEFAULT_SPACY_MODELS = [
    'en_core_web_md',  # Medium English model (recommended)
    # 'en_core_web_lg',  # Large model (better accuracy, more resources)
]

FALLBACK_MODELS = [
    'en_core_web_sm',  # Small model as fallback
]


def check_model_installed(model_name: str) -> bool:
    """
    Check if a spaCy model is already installed.
    
    Args:
        model_name: Name of the spaCy model
        
    Returns:
        bool: True if model is installed, False otherwise
    """
    try:
        spacy.load(model_name)
        return True
    except OSError:
        return False


def install_spacy_model(model_name: str, force: bool = False) -> bool:
    """
    Install a spaCy language model.
    
    Args:
        model_name: Name of the spaCy model to install
        force: Force reinstallation even if already installed
        
    Returns:
        bool: True if installation successful, False otherwise
    """
    if not force and check_model_installed(model_name):
        logger.info(f"Model '{model_name}' is already installed. Skipping.")
        return True
    
    try:
        logger.info(f"Downloading and installing spaCy model: {model_name}")
        logger.info("This may take a few minutes depending on your connection...")
        
        # Download the model
        spacy_download(model_name)
        
        # Verify installation
        if check_model_installed(model_name):
            logger.info(f"Successfully installed: {model_name}")
            return True
        else:
            logger.error(f"Installation verification failed for: {model_name}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to install {model_name}: {str(e)}")
        return False


def install_all_models(force: bool = False) -> bool:
    """
    Install all required ML models.
    
    Args:
        force: Force reinstallation of all models
        
    Returns:
        bool: True if all installations successful
    """
    success = True
    
    logger.info("="*60)
    logger.info("ML Voice Lead Analysis - Model Installation")
    logger.info("="*60)
    
    # Install primary spaCy models
    logger.info("\nInstalling primary spaCy models...")
    for model in DEFAULT_SPACY_MODELS:
        if not install_spacy_model(model, force):
            logger.warning(f"Failed to install primary model: {model}")
            success = False
    
    # If primary models failed, try fallback
    if not success:
        logger.info("\nAttempting to install fallback models...")
        for model in FALLBACK_MODELS:
            if install_spacy_model(model, force):
                logger.info("Fallback model installed successfully.")
                success = True
                break
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("Model installation completed successfully!")
        logger.info("You can now run the application.")
    else:
        logger.error("Model installation failed. Please check the errors above.")
        logger.error("You may need to install models manually:")
        logger.error("  python -m spacy download en_core_web_md")
    logger.info("="*60)
    
    return success


def verify_installation() -> bool:
    """
    Verify that at least one required model is installed.
    
    Returns:
        bool: True if at least one model is available
    """
    all_models = DEFAULT_SPACY_MODELS + FALLBACK_MODELS
    
    for model in all_models:
        if check_model_installed(model):
            logger.info(f"Verified: {model} is installed and ready.")
            return True
    
    logger.error("No spaCy models are installed!")
    return False


def main():
    """
    Main entry point for the model installation script.
    """
    parser = argparse.ArgumentParser(
        description='Install ML models for Voice Lead Analysis'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific spaCy model to install (e.g., en_core_web_md)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reinstallation of models'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify if models are installed, do not install'
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify:
            # Verification mode
            if verify_installation():
                logger.info("Model verification successful.")
                sys.exit(0)
            else:
                logger.error("Model verification failed.")
                sys.exit(1)
        
        elif args.model:
            # Install specific model
            if install_spacy_model(args.model, args.force):
                logger.info("Installation successful.")
                sys.exit(0)
            else:
                logger.error("Installation failed.")
                sys.exit(1)
        
        else:
            # Install all default models
            if install_all_models(args.force):
                sys.exit(0)
            else:
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("\nInstallation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
