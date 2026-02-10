# Image Captioning with a Frozen CNN Encoder + Transformer Decoder (PyTorch)

This project builds an **end-to-end image captioning model** that generates natural-language descriptions for images. It uses a **pretrained image-classification CNN** as a frozen visual encoder and a **Transformer decoder** (implemented with PyTorch’s `nn.TransformerDecoder`) to produce captions autoregressively.

The goal is to demonstrate practical **multimodal / vision-language modeling** skills: transfer learning, tokenization, data augmentation, sequence modeling with Transformers, and autoregressive generation.

---

## Results

![alt text](<Screenshot 2026-02-10 182906.png>)

![alt text](<Screenshot 2026-02-10 182741.png>) 

![alt text](<Screenshot 2026-02-10 182620.png>) 

![alt text](<Screenshot 2026-02-10 182218.png>) 

![alt text](<Screenshot 2026-02-10 183541.png>)

---

## Highlights

* **Visual encoder:** pretrained **EfficientNet-B0** (`timm`) used as a feature extractor (`num_classes=0`, global average pooling)
* **Frozen CNN weights:** encoder runs under `torch.no_grad()` and `requires_grad=False`
* **Trainable bridge:** a linear projection maps CNN features → Transformer model dimension
* **Text decoder:** multi-layer Transformer decoder with causal masking for next-token prediction
* **Data pipeline:** Flickr8k captions parsing + custom word-level tokenizer (NLTK) + padding/truncation
* **Image preprocessing:** Albumentations resize/augment + ImageNet normalization
* **Training optimizations:** AMP (mixed precision when CUDA available), gradient clipping, and fast DataLoader settings
* **Inference:** greedy decoding or sampling with `temperature` + `top_k`

---

## Dataset

* **Flickr8kVersion** downloaded from Kaggle: [https://www.kaggle.com/datasets/adityajn105/flickr8k/data](https://www.kaggle.com/datasets/adityajn105/flickr8k/data)
* The notebook expects the dataset to live in Google Drive like:

```
Flickr8kVersion/
  Images/
    <all .jpg files>
  captions.txt
```

### Caption parsing

* `captions.txt` is read line-by-line and converted into a DataFrame of `(filename, caption)` pairs.
* One known missing image is skipped to avoid training crashes:

  * `2258277193_586949ec62.jpg`

---

## Model Architecture

### High-level flow

1. **Image → CNN encoder (EfficientNet-B0)** → pooled feature vector
2. **Linear projection** to match Transformer `model_dim`
3. **Transformer decoder** attends to the image embedding (as *memory*) and previous caption tokens
4. **Vocabulary projection** outputs logits over tokens for each time step

### Encoder (frozen)

* Implemented via `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')`
* Output is a single feature vector per image.
* Encoder weights are frozen:

  * `for p in model.cnn_encoder.parameters(): p.requires_grad = False`
  * forward pass uses `with torch.no_grad(): ...`

### Bridge layer

* `project: nn.Linear(in_features, model_dim)`
* The projected embedding is reshaped into Transformer “memory” as `(B, 1, D)`.

### Decoder (Transformer)

* Token embeddings: `nn.Embedding(vocab_size, model_dim)`
* Positional embeddings: `nn.Embedding(context_length, model_dim)`
* Decoder stack:

  * `nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=2*model_dim, dropout=prob, batch_first=True, norm_first=True)`
  * wrapped by `nn.TransformerDecoder(..., num_layers=num_blocks)`
* Output head: `nn.Linear(model_dim, vocab_size)`

### Training-time masking

* **Causal mask** (prevents looking ahead): `generate_square_subsequent_mask(T)`
* **Padding mask** (ignores `<PAD>` tokens): `tgt_key_padding_mask = (tokens == pad_id)`

---

## Tokenization & Vocabulary

This notebook uses a simple, robust **word-level tokenizer** based on NLTK:

* Tokenization: `wordpunct_tokenize`
* Normalization:

  * lowercasing
  * keep only alphanumeric tokens (`t.isalnum()`)

Vocabulary is built from caption word frequencies and includes special tokens:

* `<UNKNOWN>` (id 0)
* `<PAD>` (id 1)
* `<START>` (id 2)
* `<END>` (id 3)

Each caption is converted into a fixed-length integer sequence:

* `[<START>] + tokens + [<END>]`
* **Padding:** pad to `context_length` with `<PAD>`
* **Truncation:** if too long, truncate and force the final token to `<END>`

---

## Image Preprocessing (Albumentations)

All images are resized to **224×224** and normalized with **ImageNet mean/std**.

* Common transforms:

  * `Resize(224, 224)`
  * `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))`
  * `ToTensorV2()`
* Training-only augmentation:

  * `HorizontalFlip()`
  * `ColorJitter()`

---

## Training

### Objective

The model is trained with **teacher forcing** using **next-token prediction**:

* Input tokens: `<START> w1 w2 ... w_{T-2}`
* Target tokens: `w1 w2 ... w_{T-2} <END>/<PAD>`

Loss:

* `CrossEntropyLoss(ignore_index=pad_id)`

### Hyperparameters (from the notebook)

* `context_length = 20`
* `model_dim = 512`
* `num_blocks = 6`
* `num_heads = 16`
* `dropout = 0.1`

Training loop settings:

* `batch_size = 128`
* `num_epochs = 40`
* Optimizer: `AdamW(lr=2e-5, weight_decay=0.01)`
* Gradient clipping: `clip_grad_norm_(..., 2.0)`
* AMP (if CUDA is available): `torch.cuda.amp.autocast` + `GradScaler`

DataLoader performance options:

* `num_workers = 8`
* `pin_memory = True`
* `persistent_workers = True`
* `prefetch_factor = 2`

### Saving

Weights are saved to:

```bash
torch.save(model.state_dict(), 'weights.pt')
```

---

## Loading a Trained Model

If you’ve already trained the model, you can skip the training cells and load the saved weights:

```python
model.load_state_dict(torch.load('weights.pt'))
```

Make sure you recreate the model with the **same hyperparameters** used during training (same `context_length`, vocab, `model_dim`, etc.).

---

## Caption Generation (Inference)

The notebook includes a `generate_caption(...)` function that:

1. Loads and preprocesses an image
2. Encodes it **once** using the frozen CNN
3. Autoregressively decodes tokens until `<END>` or `context_length` is reached

Supported decoding modes:

* **Greedy decoding:** `temperature=0.0`
* **Sampling:** set `temperature>0` and optionally `top_k` (e.g., `top_k=50`)

Example (as used in the notebook):

```python
caption = generate_caption(
    model=model,
    image_path=image_path,
    vocabulary=vocabulary,
    context_length=context_length,
    device=device,
    temperature=0.6,
    top_k=50,
)
print(caption)
```

---

## How to Run (Colab)

1. Open the notebook: `image_captioning_transformer.ipynb`
2. Install dependencies:

   * `timm`, `albumentations`
3. Mount Google Drive and set the working directory:

   * `drive/MyDrive/Flickr8kVersion`
4. Run:

   * dataset loading
   * tokenization
   * model definition
   * training loop (optional if you already have weights)
   * save weights
5. For inference:

   * load `weights.pt`
   * upload an image in Colab
   * run caption generation


---

## Notes, Limitations, and Next Improvements

This project is intentionally focused on the **core multimodal modeling pipeline**. The results are not so great due to a few factors:

* The training data is small, which also reduces vocabulary size significantly.
* The number of parameters is low. More attention heads and more transformer blocks will increase the number of parameters and this, along with a bigger dataset, will help with better learning.

Possible Improvements:

* Use a pre-trained BERT/GPT decoder and finetune it on the current dataset
* Add **train/val/test splits** and report captioning metrics.
* Unfreeze and fine-tune the CNN:

  * unfreeze only the last blocks for a small LR
* Use a stronger vision encoder (e.g., ViT, ConvNeXt) or CLIP-like features
* Replace the single “memory token” with spatial features (patch/grid features) so the decoder can attend to richer image content

---

## Tech Stack

* Python
* PyTorch
* `timm` (pretrained CNN encoder)
* Albumentations (`Resize`, augmentation, normalization)
* OpenCV (image loading)
* NLTK (tokenization)
* Pandas (caption table)

---

## Acknowledgements

* Flickr8k dataset (via Kaggle)
* `timm` and PyTorch for pretrained models and Transformer building blocks
