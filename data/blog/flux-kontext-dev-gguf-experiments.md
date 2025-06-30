---
title: Flux Kontext Dev GGUF workflow and Experiments ComfyUI
date: '2024-01-14'
tags: ['flux', 'comfyui', 'gguf', 'image-gen']
draft: false
summary: "Exploring GGUF model conversion workflow and experiments with ComfyUI for optimized AI inference"
---

## GGUF Flux Kontext Dev workflow

- used the comfyui template kontext dev workflow
- using q4_k_m quantized model
- replaced the unet and dual clip loader with the gguf ones
- also using load image from url (mtb) node to use url of the rabbit image

Performance: 12 secs/iteration , 20 steps, 303s, 5mins 3 secs, 03:53
## Basic Style Transfer
<ImageGrid
  images={[
    {
      url: "https://raw.githubusercontent.com/Comfy-Org/example_workflows/main/flux/kontext/dev/rabbit.jpg",
      caption: "Original input image"
    },
    {
      url: "https://huggingface.co/datasets/sandeshrajx/blog_images/resolve/main/flux-kontext-dev-test/style-transfer-1-default.png",
      caption: "Default style transfer output"
    }
  ]}
/>

## hyper flux 8 step lora tests

- after applying the hyper flux 8 step lora, 8 steps, 0.15 strength

<ImageGrid
  images={[
    {
      url: "https://huggingface.co/datasets/sandeshrajx/blog_images/resolve/main/flux-kontext-dev-test/style-transfer-hyper-8step-0.15.png",
      caption: "0.15 strength, after applying the hyper flux 8 step lora, 8 steps"
    },
    {
      url: "https://huggingface.co/datasets/sandeshrajx/blog_images/resolve/main/flux-kontext-dev-test/style-transfer-hyper-8step-0.3.png", 
      caption: "hyper flux, 0.3 strength"
    },
    {
      url: "https://huggingface.co/datasets/sandeshrajx/blog_images/resolve/main/flux-kontext-dev-test/style-transfer-hyper-8step-0.4.png",
      caption: "0.4 strength, 3.5 clip, instead of 2.5"
    },
    {
      url: "https://huggingface.co/datasets/sandeshrajx/blog_images/resolve/main/flux-kontext-dev-test/style-transfer-hyper-8step-no-reference-latent.png",
      caption: "No reference latent conditioning"
    }
  ]}
  columns={4}
  maxWidth="none"
/>

- changing the strength to 0.6 gave noisy image
- removed the Reference latent conditioning, and the output changes, so theres defnitely some effect.

## Remove object
- removing object seems to work well even when using the hyper flux 8 step lora.

<ImageGrid
  images={[
    {
      url: "https://arxiv.org/html/2506.15742v2/extracted/6566027/img/cc/img1.jpg",
      caption: "Original image from arXiv paper"
    },
    {
      url: "https://huggingface.co/datasets/sandeshrajx/blog_images/resolve/main/flux-kontext-dev-test/remove-object-portrait-kontext-dev.png",
      caption: "Removing object works flux kontext dev, hyper flux 8 step lora"
    }
  ]}
/>

This is still W.I.P. expect more things.