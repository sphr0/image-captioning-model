# image-captioning-model
Generates a description for a given image.
The goal of this project is to find the best model architecture that follows these criterea:
1. High scores with the available evaluation metrics.
2. Needs to run fast, be lightweight, and not take too much space. (low number of params)
4. Needs to follow state-of-the-art trends in Vision-Language models.
5. The models shouldn't be too similar to one another so more architectures are covered.

After evaluating each model, the superior architecture will be deployed on Gradio.

From many different proposed models, I have chosen the following three:
1. ViT + GPT2 (with cross-attention)
2. BLIP (without CapFilt)
3. GIT

Each model will be written from scratch at first, then will be downloaded with pretrained weights from HuggingFace.

# Reference Papers
- ClipCap: CLIP Prefix for Image Captioning (Mokady, R.) 2021
-  An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy, A.) 2020
-  Attention is All You Need (Vaswani, A) 2017
-  Language Models are Unsupervised Multitask Learners (Radford, A.) 2019
-  BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li, J) 2022
- Align before Fuse: Vision and Language Representation Learning with Momentum Distillation (Li, J) 2021
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin, J) 2019
- GIT: A Generative Image-to-text Transformer for Vision and Language (Wang, J) 2022
-  Learning Transferable Visual Models From Natural Language Supervision (Radford, A.) 2021
- Unified Language Model Pre-training for Natural Language Understanding and Generation (Dong, L) 2019
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (Li, J) 2023
- Improved Baselines with Visual Instruction Tuning (Liu, H) 2024
