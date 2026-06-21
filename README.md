# Crop Disease Detection — Live Deployment

A fine-tuned Vision Transformer that classifies crop leaf images into 38 disease/healthy categories across 14 crop species, with GradCAM-based explainability showing which part of each image the model used to make its decision.

**Live app:** https://huggingface.co/spaces/PraiseO2/crop-disease-detection

Upload a leaf photo, get a prediction and a heatmap. No setup required.

## What this is

This Space is the permanently hosted version of a model originally trained and evaluated in a Colab notebook, then prototyped behind a FastAPI service before being rewritten as a standalone deployment here. It loads its checkpoint directly from a HuggingFace Hub model repo (`PraiseO2/crop-disease-vit`) rather than from any personal storage, so it runs independently with no dependency on a notebook session staying alive.

## Results

- 99.68% accuracy on a held-out test set of 8,147 images
- 99.73% validation accuracy at the saved checkpoint (epoch 4 of 5)

These numbers come from PlantVillage, a lab-controlled dataset with consistent backgrounds and lighting per class. Published critiques note leaf-grouping leakage in this dataset — the same physical leaf photographed multiple times can end up split across train and test. High accuracy here is real, but it's a benchmark number, not a guarantee of field performance.

I tested this directly with a real-world photo showing insect damage and irregular discoloration — visually unlike PlantVillage's clean single-symptom images. The model returned 30.2% confidence, barely above the ~2.6% random baseline across 38 classes. It didn't confidently hallucinate a wrong diagnosis on an out-of-distribution image — it correctly signaled uncertainty instead, which is the behavior you actually want from a model facing something it wasn't trained on.

## Explainability

GradCAM was run across batches of 15 and 40 disease-class test images, measuring what fraction of the model's strongest attention landed on the leaf versus the background. Results were consistent across both batches: mean on-leaf attention around 50-54%, with high variance by disease type.

Diseases with large, well-defined lesions (Target Spot, Spider Mites) showed strong on-leaf attention, 80-90%+. Diseases presenting as small or scattered marks — Bacterial Spot, Late Blight, Powdery Mildew — consistently showed attention landing mostly on background instead of the symptom itself, despite the model still reaching correct classifications.

This points to the model partly relying on leaf silhouette or holistic image statistics rather than fine-grained symptom localization for some disease types — a known risk with single-background lab datasets. It's documented here rather than hidden behind the headline accuracy number.

## How it works

```
Leaf image upload
       ↓
ViT-Base (fine-tuned, 85.8M params) → class prediction + confidence
       ↓
GradCAM → attention heatmap over the model's final transformer layer
       ↓
Gradio interface displays diagnosis + heatmap
```

The checkpoint was originally trained under an older transformers library version with different internal layer naming. Since this Space installs a newer transformers version (required for compatibility with the platform's Gradio build), `app.py` includes a key-remapping step that translates the checkpoint's saved parameter names to whatever the installed library version expects at load time — the weights themselves are untouched, only the names are translated.

## Stack

- **Model**: `google/vit-base-patch16-224`, fine-tuned (classifier head replaced: 768 → 38 classes)
- **Explainability**: `pytorch-grad-cam`, with a layer-discovery function that locates the model's final transformer block programmatically rather than assuming a fixed internal path — this keeps it resilient to transformers library version changes
- **Serving**: Gradio, running natively on HuggingFace Spaces infrastructure
- **Model hosting**: HuggingFace Hub model repo, loaded via `hf_hub_download`

## Related

- Training notebook, full evaluation, and the Colab-based FastAPI prototype: [Crop-Disease-Detection-ViT-fastAPI](https://github.com/PraiseO2/Crop-Disease-Detection-ViT-fastAPI)
- Model weights: [PraiseO2/crop-disease-vit](https://huggingface.co/PraiseO2/crop-disease-vit)

## Limitations

PlantVillage's lab conditions mean this accuracy won't transfer cleanly to farmer-captured field photos with natural backgrounds and lighting. GradCAM analysis shows inconsistent symptom localization for diseases with small or scattered visual markers. No field validation has been done — that's the obvious next step before this is more than a strong benchmark result.

## Background

B.Agric (Hons.), Crop Production and Protection, Obafemi Awolowo University. Part of a broader effort to build practical AI tooling for smallholder agriculture in sub-Saharan Africa, alongside a RAG-based agronomic advisory system and a commodity price forecasting model.

---

**Praise Omoolorun Adetifa**
[GitHub](https://github.com/PraiseO2) · [LinkedIn](https://linkedin.com/in/praise-adetifa-3732161b2) · adetifapraisea1@gmail.com
