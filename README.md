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
