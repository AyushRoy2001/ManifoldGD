<p align="center">
  <img width="600" height="400" alt="intro_2" src="https://github.com/user-attachments/assets/6b711007-2996-427b-a69b-bd16eca058fe" />
</p>

# [CVPR 2026] ManifoldGD: Training-Free Hierarchical Manifold Guidance for Diffusion-Based Dataset Distillation

<p align="center">
  <strong>Ayush Roy</strong>¹ &middot; 
  <strong>Wei-Yang Alex Lee</strong>¹ &middot;
  <strong>Rudrasis Chakraborty</strong>² &middot;
  <strong>Vishnu Suresh Lokhande</strong>¹
</p>

<p align="center">
  ¹ University at Buffalo, SUNY &bull; ² Lawrence Livermore National Lab (LLNL)
</p>

## Abstract

In recent times, large datasets hinder efficient model training while also containing redundant concepts. Dataset distillation aims to synthesize compact datasets that preserve the knowledge of large-scale training sets while drastically reducing storage and computation. Recent advances in diffusion models have enabled training-free distillation by leveraging pre-trained generative priors; however, existing guidance strategies remain limited. Current score-based methods either perform unguided denoising or rely on simple mode-based guidance toward instance prototype centroids (IPC centroids), which often are rudimentary and suboptimal. We propose Manifold-Guided Distillation (ManifoldGD), a training-free diffusion-based framework that integrates manifold consistent guidance at every denoising timestep. Our method employs IPCs computed via a hierarchical, divisive clustering of VAE latent features—yielding a multi-scale coreset of IPCs that captures both coarse semantic modes and fine intra-class variability. Using a local neighborhood of the extracted IPC centroids, we create the latent manifold for each diffusion denoising timestep. At each denoising step, we project the mode-alignment vector onto the local tangent space of the estimated latent manifold, thus constraining the generation trajectory to remain manifold-faithful while preserving semantic consistency. This formulation improves representativeness, diversity, and image fidelity without requiring any model retraining. Empirical results demonstrate consistent gains over existing training-free and training-based baselines in terms of FID, $\ell_2$ distance among real and synthetic dataset embeddings, and classification accuracy, establishing ManifoldGD as the first geometry-aware training-free data distillation framework.

## Final code coming soon!
