# Architecture - Image Classification - Dog vs Lion with SVM

> **Docs:** [README](../README.md) | [Architecture](ARCHITECTURE.md) | [HLD](HLD.md) | [LLD](LLD.md) | [Model Details](MODEL_DETAILS.md) | [AI Primer](AI_ML_GENAI_PRIMER.md) | [Presentation](presentation/Image_Classification_SVM.pptx)

## 1. System context

A Colab notebook that builds its own dataset from the web, trains an SVM on raw pixels, persists the model and offers an interactive 'paste an image URL' prediction step.

## 2. Architecture diagram

```mermaid
flowchart LR
  A[Bing Image Search] -->|bing-image-downloader<br/>30 per class| B[(images/dog, images/lion)]
  B --> C[skimage imread]
  C --> D[resize 150x150x3]
  D --> E[flatten -> 67,500 features]
  E --> F[train_test_split 70/30]
  F --> G[GridSearchCV<br/>SVC linear / rbf, C, gamma]
  G --> H[Best SVM]
  H --> I[accuracy + confusion matrix]
  H --> J[(img_model.p pickle)]
  U[Image URL] --> K[imread + resize + flatten]
  J --> L[predict]
  K --> L
  L --> M[dog / lion]
```

## 3. Layered view

| Layer | Responsibility | Implementation |
|---|---|---|
| Acquisition layer | Download images per category | bing-image-downloader |
| Storage layer | Local folders `images/dog`, `images/lion` | Colab filesystem |
| Preprocessing layer | Read, resize to 150x150x3, scale 0-1, flatten | scikit-image, NumPy |
| Model layer | SVC with hyper-parameter grid search | scikit-learn |
| Persistence layer | `img_model.p` | pickle |
| Inference layer | URL -> preprocess -> predict -> label | scikit-image + model |

## 4. End-to-end data flow

```mermaid
sequenceDiagram
  participant U as User
  participant NB as Notebook
  participant B as Bing
  participant SK as scikit-learn
  NB->>B: download("dog", 30), download("lion", 30)
  B-->>NB: images/dog/*.jpg, images/lion/*.jpg
  NB->>NB: imread -> resize(150,150,3) -> flatten
  NB->>SK: train_test_split(0.3, rs=102)
  NB->>SK: GridSearchCV(SVC(probability=True), 12 configs).fit
  SK-->>NB: best estimator
  NB->>NB: accuracy 0.611, confusion matrix
  NB->>NB: pickle.dump -> img_model.p
  U->>NB: image URL
  NB->>NB: imread(url) -> resize -> flatten
  NB->>SK: model.predict
  SK-->>U: PREDICTED OUTPUT: dog
```

## 5. Deployment view

Runs in Google Colab; the pickled model could be loaded into a Streamlit/Gradio app or FastAPI endpoint for serving.

## 6. Target (production / GenAI-ready) architecture

A modern embedding-based version:

```mermaid
flowchart LR
  I[Image upload / URL] --> P[Pre-trained encoder<br/>CLIP / ResNet / ViT]
  P --> E[512-d embedding]
  E --> C[Linear classifier or zero-shot text prompts]
  E --> V[(Vector DB)]
  C --> L[Label + confidence]
  V --> S[Similar-image search]
```
