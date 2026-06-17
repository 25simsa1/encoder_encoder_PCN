# Bidirectional Predictive Coding for Multimodal Image-Text Conversion (Ongoing research)

This repository implements a multimodal learning framework based on **bidirectional predictive coding**, designed to learn joint representations between images and text. The goal is to explore how structured latent spaces can support consistent cross-modal generation and inference.



## Overview
Modern multimodal systems typically rely on large-scale encoder-decoder architectures or autoregressive transformers. While effective, these approaches often lack explicit mechanisms for verifying how information is stored, updated, or retrieved internally.

This project investigates an alternative perspective:

> Can multimodal learning be framed as a **bidirectional predictive coding process over shared states**, where both modalities iteratively refine a consistent internal representation?

We implement a bidirectional system where image and text encoders jointly participate in reconstructing and predicting each other through shared states.



## Key Idea

The core hypothesis is that multimodal understanding emerges from **constraint satisfaction in latent space**, rather than purely feedforward mapping.

We model this as:

* Image → Latent (shared states) → Text reconstruction
* Text → Latent (shared states) → Image reconstruction
* Iterative refinement via predictive error minimization

This creates a system where shared states behave like a **shared memory substrate** across modalities.


## Architecture

The system consists of:

* **Image Encoder (CNN)**
* **Text Encoder (Transformer-based encoder)**
* **Shared States**
* **Predictive error with bidirectional consistency objective**

### Training Objective

The model optimizes a combined reconstruction and consistency loss:

* Image reconstruction loss
* Text reconstruction loss
* Cross-modal alignment loss
* Latent consistency regularization

## Features

* Bidirectional image ↔ text learning
* Shared latent representation space
* Bidirectional Predictive Coding training loop
* Encoder-decoder architecture
* Experiments on nocaps-style multimodal data



## Tech Stack

* TensorFlow
* CNN vision encoder
* Transformer text encoders

## Author

Anthony Yeh,
Colby College
