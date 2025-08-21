# GAN Artifact Detector

A deepfake detection system that leverages insights from GAN training dynamics and loss function optimization to identify specific visual artifacts in generated content.

## Project Overview

This project builds on research into adaptive loss functions for GANs, using knowledge of how different loss weightings create detectable visual signatures in generated content.

## Architecture

- **Smoothing Artifact Detector**: Identifies over-smoothing from high pixel-loss training
- **Texture Artifact Detector**: Detects perceptual loss-induced inconsistencies  
- **Mode Collapse Detector**: Finds adversarial loss-related generation artifacts

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
