# 🧠 Phase 10 — ML/DL Essentials for AI Engineers
> **Complete Study Notes | Theory + Code + Use Cases | Interview Prep**
> Part of: GenAI + LLMOps Engineering Roadmap 2026
> Estimated time: 35–45 hrs over 3–4 weeks
> **You don't need to become a data scientist. You need just enough ML/DL to use pre-trained models intelligently, fine-tune them, and build multimodal AI features.**
> Prerequisites: Phase 1–9 (especially Phase 9 fine-tuning section)

---

## 📑 Table of Contents

1. [Why ML/DL Matters for a GenAI Engineer](#1-why-mldl-matters-for-a-genai-engineer)
2. [10.1 — ML Foundations: Just Enough](#101--ml-foundations-just-enough)
3. [10.2 — Deep Learning: Practical Only](#102--deep-learning-practical-only)
4. [10.3 — CLIP & Multimodal Models](#103--clip--multimodal-models)
5. [10.4 — Fashion AI & Domain-Specific CLIP](#104--fashion-ai--domain-specific-clip)
6. [10.5 — Other Useful Pre-Trained Models](#105--other-useful-pre-trained-models)
7. [Phase 10 Projects](#-phase-10-projects)
8. [Interview Cheat Sheet](#-interview-cheat-sheet)
9. [Quick Revision Cards](#-quick-revision-cards)

---

## 1. Why ML/DL Matters for a GenAI Engineer

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You've spent 9 phases calling LLM APIs, building RAG, fine-tuning with Unsloth, deploying on AWS. Why do you now need ML/DL basics?

```
Three reasons:

1. SOME PROBLEMS DON'T NEED AN LLM
   "Is this query about billing or shipping?" 
   → A 2KB LogisticRegression model answers in 0.5ms for $0.
   → GPT-4o takes 1.5s and costs $0.011.
   
   Knowing when NOT to use an LLM is a senior skill.

2. CLIP IS THE BRIDGE BETWEEN TEXT AND IMAGES
   Every multimodal AI product (visual search, image Q&A, product similarity)
   uses CLIP or a variant under the hood.
   Without understanding CLIP embeddings, you can't build these features.

3. INTERVIEWS WILL TEST THIS
   "What's the difference between CLIP and a CNN?"
   "When would you use KMeans on your RAG pipeline outputs?"
   "What is contrastive learning?"
   If you blank on these, you look like someone who only calls APIs.
```

### The Right Mental Model

```
DATA SCIENTIST:        Trains models from scratch, optimises loss functions,
                       cares about model internals, publishes papers.

GENAI ENGINEER (you):  Uses pre-trained models as building blocks.
                       Knows enough theory to:
                         • Pick the right pre-trained model for a task
                         • Fine-tune intelligently (already done in Phase 9)
                         • Embed images/text in the same space (CLIP)
                         • Debug when the model behaves unexpectedly
                         • Answer interview questions confidently

This phase = the minimum viable ML/DL knowledge for a GenAI engineer.
```

---

## 10.1 — ML Foundations: Just Enough

### Learning Types

```
SUPERVISED LEARNING:
  You give it labelled examples: (input → correct output)
  Model learns to predict output from input.
  Examples: email spam (email → spam/not), intent classifier (query → intent)
  Your use case: classify query intent BEFORE calling the LLM

UNSUPERVISED LEARNING:
  No labels. Model finds structure in data by itself.
  Examples: cluster similar documents, find anomalies
  Your use case: cluster 10K customer support tickets to find common themes

SELF-SUPERVISED LEARNING:
  Labels created from the data itself.
  "Predict the next word" (language model pretraining)
  "Predict if two images match" (CLIP contrastive training)
  Your use case: THIS is how the models you fine-tune were originally trained

REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF):
  Human rates outputs → model learns to produce preferred outputs
  Your use case: DPO/ORPO in Phase 9 is a modern replacement for full RLHF

WHEN TO USE WHICH:
  Have labelled data, well-defined outputs → Supervised (classifier, regressor)
  No labels, want to explore structure → Unsupervised (clustering, anomaly detection)
  Want to understand patterns without labels → Self-supervised (pre-training)
```

### Train / Val / Test Split — Why It Matters

```python
"""
Why three splits?

Train:  what the model learns from (sees during training)
Val:    used to tune hyperparameters and detect overfitting
        (seen indirectly — you pick the model that scores best on val)
Test:   final report card — touched ONCE at the very end
        If you use test to pick models, you're "cheating" — overfitting to test

Data leakage: information from future leaks into past during training.
  Example: you're predicting customer churn.
  WRONG: include "cancelled_date" in features (it happens AFTER the event you predict)
  WRONG: normalise entire dataset, THEN split (val/test stats contaminate train)
  CORRECT: split first, then fit scaler on train set only, transform val/test
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.random.randn(1000, 20)   # 1000 samples, 20 features
y = np.random.randint(0, 2, 1000)

# Split first
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
# Train: 700 | Val: 150 | Test: 150

# Fit scaler on TRAIN only — transform all splits with train's stats
scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train)   # fit + transform
X_val    = scaler.transform(X_val)          # transform only (no fit)
X_test   = scaler.transform(X_test)         # transform only
# ✅ Correct: val/test don't leak into training normalization
```

### Overfitting vs Underfitting

```python
"""
Underfitting:  model too simple → bad on train AND val
               Fix: more features, more complex model, more training

Overfitting:   model memorised training data → great on train, bad on val
               Fix: regularisation, dropout, early stopping, more data

The bias-variance tradeoff:
  High bias (underfitting):    model makes systematic errors regardless of data
  High variance (overfitting): model too sensitive to training data, fails on new data
  
  Sweet spot: low bias + low variance = generalises well
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Underfitting: logistic regression on complex non-linear data
lr    = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LogReg  train F1: {f1_score(y_train, lr.predict(X_train)):.3f}")
print(f"LogReg  val   F1: {f1_score(y_val,   lr.predict(X_val)):.3f}")
# LogReg train F1: 0.51 | val F1: 0.50 ← both bad = underfitting

# Overfitting: deep tree with no constraints
rf_overfit = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_overfit.fit(X_train, y_train)
print(f"RF(deep) train F1: {f1_score(y_train, rf_overfit.predict(X_train)):.3f}")
print(f"RF(deep) val   F1: {f1_score(y_val,   rf_overfit.predict(X_val)):.3f}")
# RF train F1: 1.00 | val F1: 0.58 ← huge gap = overfitting

# Just right: constrained tree + regularisation
rf_good = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,           # ← regularisation: limit tree depth
    min_samples_leaf=5,    # ← regularisation: require min samples per leaf
    random_state=42,
)
rf_good.fit(X_train, y_train)
print(f"RF(good) train F1: {f1_score(y_train, rf_good.predict(X_train)):.3f}")
print(f"RF(good) val   F1: {f1_score(y_val,   rf_good.predict(X_val)):.3f}")
# RF train F1: 0.82 | val F1: 0.79 ← small gap = generalising well
```

### Scikit-Learn for AI Engineers

```python
"""
The 5 sklearn tools you'll actually use as a GenAI engineer:

1. RandomForestClassifier  → query intent classifier, doc type classifier
2. LogisticRegression      → binary decisions (is this spam? is this relevant?)
3. KMeans                  → cluster similar documents or embeddings
4. Pipeline                → chain preprocessing + model cleanly
5. cross_val_score         → quick reliable evaluation

You won't need SVM, Decision Trees, Naive Bayes, etc. day-to-day.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

# ── Use case 1: Query intent classifier ────────────────────────────────
# Before calling GPT-4o, classify what kind of query it is.
# Cheap classifier → route to right agent / system prompt

def build_intent_classifier(X_train, y_train):
    """
    X_train: TF-IDF or sentence embedding vectors of queries
    y_train: intent labels ("billing", "shipping", "returns", "general")
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Evaluate with cross-validation (more reliable than single split)
from sklearn.feature_extraction.text import TfidfVectorizer

sample_queries = [
    "Where is my order?",         # shipping
    "I want to return my item",   # returns
    "My invoice is wrong",        # billing
    "What are your products?",    # general
    "Track my package",           # shipping
    "Refund request",             # returns
]
sample_labels = ["shipping", "returns", "billing", "general", "shipping", "returns"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X          = vectorizer.fit_transform(sample_queries).toarray()
y          = sample_labels

# With just 6 samples, skip cross-val — for illustration
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)
pred = clf.predict(vectorizer.transform(["How do I send back a product?"]))
print(f"Intent: {pred[0]}")   # → returns

# ── Use case 2: Cluster similar support tickets ─────────────────────────
# Find common themes across 10K tickets WITHOUT labels

def cluster_support_tickets(embeddings: np.ndarray, n_clusters: int = 8):
    """
    embeddings: sentence-transformer embeddings of support tickets
    Returns: cluster labels per ticket
    """
    # Reduce dimensions first (KMeans works better in lower dims)
    pca        = PCA(n_components=50, random_state=42)
    reduced    = pca.fit_transform(embeddings)

    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels     = kmeans.fit_predict(reduced)

    # Find representative tickets per cluster (closest to centroid)
    centroids  = kmeans.cluster_centers_
    for cluster_id in range(n_clusters):
        cluster_mask  = labels == cluster_id
        cluster_vecs  = reduced[cluster_mask]
        distances     = np.linalg.norm(cluster_vecs - centroids[cluster_id], axis=1)
        representative= np.argmin(distances)
        print(f"Cluster {cluster_id} ({cluster_mask.sum()} tickets): representative index {representative}")

    return labels

# ── Use case 3: Visualise embedding space ──────────────────────────────
def visualise_embeddings(embeddings: np.ndarray, labels: list, n_samples: int = 500):
    """
    Reduce high-dim embeddings to 2D for visualisation.
    Use UMAP (faster) or t-SNE (better quality for small datasets).
    """
    # PCA first (UMAP/t-SNE works better on ~50 dims)
    pca      = PCA(n_components=50)
    reduced  = pca.fit_transform(embeddings[:n_samples])

    # t-SNE for 2D visualisation
    tsne     = TSNE(n_components=2, perplexity=30, random_state=42)
    coords   = tsne.fit_transform(reduced)

    # Plot (use in Jupyter notebook)
    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels[:n_samples]))
    for label in unique_labels:
        mask = [l == label for l in labels[:n_samples]]
        plt.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.6)
    plt.legend()
    plt.title("Embedding Space Visualisation (t-SNE)")
    plt.show()
    # In practice: use this to audit RAG quality — do similar docs cluster together?

# ── Use case 4: Anomaly detection on LLM outputs ──────────────────────
from sklearn.ensemble import IsolationForest

def detect_anomalous_llm_outputs(
    output_features: np.ndarray,   # features: length, sentiment, perplexity
) -> np.ndarray:
    """
    Detect LLM responses that are statistically unusual.
    Use case: catch hallucinations or broken responses before showing to user.
    Isolation Forest: anomalies are "isolated" quickly by random cuts.
    """
    iso = IsolationForest(contamination=0.05, random_state=42)
    labels = iso.fit_predict(output_features)
    # -1 = anomaly, 1 = normal
    anomaly_indices = np.where(labels == -1)[0]
    print(f"Anomalies detected: {len(anomaly_indices)} / {len(output_features)}")
    return anomaly_indices

# ── Use case 5: Cross-validation ──────────────────────────────────────
def evaluate_classifier_reliably(X, y):
    """
    cross_val_score: 5-fold stratified CV — more reliable than single train/val split.
    Use this whenever you have > 200 examples.
    """
    clf    = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    scores = cross_val_score(
        clf, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted",
        n_jobs=-1,
    )
    print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Individual folds: {[f'{s:.3f}' for s in scores]}")
    return scores

# ── ML vs LLM decision guide ──────────────────────────────────────────
WHEN_ML_NOT_LLM = """
Use classical ML when:
  Task:     binary / multi-class classification (10 classes or fewer)
  Data:     100+ labelled examples available
  Speed:    need < 10ms response (real-time routing, streaming pipeline)
  Cost:     calling LLM 1M times/day is unaffordable
  Privacy:  can't send data to external APIs

Examples:
  Query intent:         "Is this about billing, shipping, or returns?" → LogisticRegression
  Document type:        "Is this a contract, invoice, or report?" → RandomForest
  Spam detection:       "Is this message spam?" → LogisticRegression + TF-IDF
  Sentiment (simple):   "Is this review positive or negative?" → LinearSVC
  Anomaly detection:    "Is this LLM response unusual?" → IsolationForest
  Clustering:           "Group these 10K tickets by topic" → KMeans

Use LLM when:
  Open-ended generation:  writing, summarisation, explanation
  Complex reasoning:      multi-step, needs world knowledge
  Low volume:             < 10K queries/day (cost acceptable)
  No labelled data:       zero-shot classification
  Nuance required:        subtle intent, ambiguous queries
"""
print(WHEN_ML_NOT_LLM)
```

---

## 10.2 — Deep Learning: Practical Only

### Neural Network Intuition

```python
"""
A neural network is a chain of matrix multiplications with non-linear functions in between.

Input → [Linear → Activation] × N → Output

Linear layer:   y = xW + b  (W = learned weights, b = learned bias)
Activation:     non-linear function applied after linear (ReLU, sigmoid, etc.)
Why activation: without it, N linear layers = 1 linear layer (no expressive power)

Forward pass:   compute output from input (left to right)
Loss:           measure how wrong the output is
Backprop:       compute gradient of loss w.r.t. each weight (right to left, chain rule)
Update:         weights -= learning_rate × gradient
"""
```

### PyTorch Basics — What You Must Know

```python
# pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Tensors — the building block ────────────────────────────────────────
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)         # torch.Size([2, 2])
print(x.dtype)         # torch.float32
print(x.device)        # cpu

# GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x      = x.to(device)

# Operations: same as numpy but differentiable
y = x * 2 + 1
z = torch.matmul(x, x.T)
print(z)

# ── Autograd — automatic differentiation ─────────────────────────────
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x     # y = x³ + 2x
y.backward()            # compute dy/dx
print(x.grad)           # dy/dx at x=2: 3x² + 2 = 14 ✓

# ── Define a model with nn.Module ─────────────────────────────────────
class IntentClassifier(nn.Module):
    """
    Simple feedforward network for query intent classification.
    Input: sentence embedding (384-dim from MiniLM)
    Output: intent logits (num_classes)
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),           # regularisation — randomly zero out 30% of neurons
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = IntentClassifier()
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Dataset + DataLoader ──────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels     = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return self.embeddings[idx], self.labels[idx]

# Synthetic data for demo
N          = 1000
embeddings = torch.randn(N, 384)
labels     = torch.randint(0, 4, (N,))

train_ds   = IntentDataset(embeddings[:800], labels[:800])
val_ds     = IntentDataset(embeddings[800:], labels[800:])

train_dl   = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl     = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ── Training loop ──────────────────────────────────────────────────────
model     = IntentClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss   = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == batch_y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss   = criterion(logits, batch_y)
            total_loss += loss.item()
            correct    += (logits.argmax(1) == batch_y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc'}")
for epoch in range(1, 11):
    tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer)
    vl_loss, vl_acc = eval_epoch(model,   val_dl,   criterion)
    scheduler.step()
    print(f"{epoch:<6} {tr_loss:<12.4f} {tr_acc:<12.3f} {vl_loss:<12.4f} {vl_acc:.3f}")

# ── Save + load ────────────────────────────────────────────────────────
torch.save(model.state_dict(), "intent_classifier.pt")
# Load:
model_loaded = IntentClassifier()
model_loaded.load_state_dict(torch.load("intent_classifier.pt", map_location="cpu"))
model_loaded.eval()
```

### CNN Intuition — What You Need to Know

```python
"""
CNN (Convolutional Neural Network):
  Built for image data. Key insight: nearby pixels are related.
  Convolution: a small filter (e.g., 3×3) slides across the image,
               detecting local patterns (edges, textures, shapes).
  Pooling:     downsamples feature maps, reducing spatial size.
  
  Layer structure:
    Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Output

YOU DON'T TRAIN CNNs FROM SCRATCH.
Always start from a pre-trained ImageNet model and fine-tune.
ResNet50 trained from scratch: 90 hours on 8 GPUs.
Fine-tune ResNet50 for your task: 20 minutes on 1 GPU.

Key models:
  ResNet50/101: classic, reliable, everywhere
  EfficientNetB0-B7: better accuracy per FLOP
  ViT (Vision Transformer): attention-based, state of art for large datasets
  CLIP visual encoder: ViT variant trained with contrastive loss on image-text pairs
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# ── Transfer learning: fine-tune ResNet50 ─────────────────────────────
def build_image_classifier(num_classes: int, freeze_backbone: bool = True):
    """
    Load pre-trained ResNet50, replace head with our num_classes head.
    freeze_backbone=True: only train the new head (fastest, less data needed)
    freeze_backbone=False: train all layers (slower, needs more data)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False   # freeze all layers

    # Replace the final classification head
    in_features     = model.fc.in_features   # 2048 for ResNet50
    model.fc        = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    # Only model.fc parameters have requires_grad=True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    return model

# ── Standard image transforms ────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                          std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])
```

### HuggingFace Pipeline API — Zero-Code ML

```python
"""
HuggingFace pipeline: use any pre-trained model in 2 lines.
This is the main way GenAI engineers use DL models in production.
"""

from transformers import pipeline

# ── Text classification ────────────────────────────────────────────────
text_clf = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1,
)
result = text_clf("This product is absolutely amazing!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# ── Zero-shot classification (no training needed) ──────────────────────
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result    = zero_shot(
    "Where is my order? It's been 5 days.",
    candidate_labels=["shipping", "billing", "returns", "general"],
)
print(result["labels"][0], result["scores"][0])
# shipping 0.87

# ── Image classification ───────────────────────────────────────────────
image_clf = pipeline("image-classification", model="google/vit-base-patch16-224")
result    = image_clf("path/to/product_image.jpg")
print(result[:3])
# [{'label': 'running shoe', 'score': 0.82}, ...]

# ── Feature extraction (get embeddings) ───────────────────────────────
feat_pipe = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
embeddings= feat_pipe(["What is RAG?", "How does retrieval augmented generation work?"])
# Returns list of embeddings — first two should be similar (same topic)

# ── HuggingFace Trainer API — fine-tune any model ─────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

dataset   = load_dataset("imdb", split={"train": "train[:1000]", "test": "test[:200]"})
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model     = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2,
)

def tokenise(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenised = dataset.map(tokenise, batched=True)

args = TrainingArguments(
    output_dir="sentiment_model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=tokenised["train"],
    eval_dataset=tokenised["test"],
)
trainer.train()
trainer.save_model("sentiment_model_v1")
```

---

## 10.3 — CLIP & Multimodal Models

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Every image model before CLIP worked the same way: take labelled images (cat: 1, dog: 0), train model to predict label. To add a new category, you need new labelled data and retrain.

CLIP breaks this pattern completely. Instead of labels, it trains on **image-text pairs** scraped from the internet: "a photo of a red sneaker" paired with the image. The model learns to place matching image-text pairs **close together** in a shared vector space, and non-matching pairs **far apart**.

Result: you can search for "red sneaker" and the model finds the right images — without ever having been trained on "red sneaker" as a category.

```
Traditional vision model:
  Training data: 1M labelled images (cat, dog, car, ...)
  Can classify: only those categories
  Adding "red sneaker": need new labelled data + retrain

CLIP:
  Training data: 400M internet image-text pairs
  Can classify: ANYTHING you can describe in text
  "red sneaker", "a sad person", "invoice with blue header" → just works
  Zero new labelled data needed
  
Key insight: TEXT IS THE LABEL.
  Instead of "class 42 = sneaker", CLIP uses "a photo of a sneaker".
  Natural language labels are infinitely flexible.
```

### CLIP Deep Dive

```python
# pip install transformers torch Pillow

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import numpy as np
import requests

# ── Load CLIP ─────────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"  # 151M params, fast
# or: "openai/clip-vit-large-patch14"        # 427M params, better quality
# or: "patrickjohncyh/fashion-clip"          # fine-tuned on fashion data

model     = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()   # inference mode

# ── Embed an image ──────────────────────────────────────────────────────
def embed_image(image_path_or_url: str) -> np.ndarray:
    """Returns normalised 512-dim embedding for one image"""
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # Normalise to unit length (required for cosine similarity)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].numpy()

# ── Embed text ─────────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    """Returns normalised 512-dim embedding for text"""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].numpy()

# ── Embed a batch (efficient) ──────────────────────────────────────────
def embed_images_batch(images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    all_features = []
    for i in range(0, len(images), batch_size):
        batch  = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        all_features.append(features.numpy())
    return np.vstack(all_features)

def embed_texts_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        all_features.append(features.numpy())
    return np.vstack(all_features)

# ── Cosine similarity ─────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # already normalised → dot product = cosine sim

# ── Zero-shot classification ──────────────────────────────────────────
def zero_shot_classify(image_path: str, candidate_labels: list[str]) -> dict:
    """
    Classify an image into any categories — no training required.
    Works by comparing image embedding to each label's text embedding.
    """
    img_emb  = embed_image(image_path)

    # Format labels as descriptive sentences (improves accuracy)
    prompts  = [f"a photo of a {label}" for label in candidate_labels]
    txt_embs = embed_texts_batch(prompts)

    # Cosine similarity between image and each label
    sims     = np.array([cosine_similarity(img_emb, t) for t in txt_embs])
    # Softmax to convert to probabilities
    probs    = np.exp(sims * 100) / np.exp(sims * 100).sum()

    results = sorted(
        [{"label": l, "score": float(p)} for l, p in zip(candidate_labels, probs)],
        key=lambda x: x["score"], reverse=True,
    )
    return results

# Demo
"""
results = zero_shot_classify(
    "product_images/blue_tshirt.jpg",
    ["t-shirt", "jeans", "sneakers", "dress", "jacket"]
)
print(results)
# [{'label': 't-shirt', 'score': 0.87}, {'label': 'jacket', 'score': 0.06}, ...]
"""
```

### CLIP + Qdrant — Visual Search Engine

```python
"""
The key insight: because CLIP puts text and images in the SAME vector space,
you can query with text → find similar images
         or query with image → find similar images
all in a single Qdrant collection.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from pathlib import Path
import uuid

CLIP_DIM   = 512   # clip-vit-base-patch32 output dimension
COLLECTION = "product_images"

qdrant = QdrantClient("localhost", port=6333)
qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
)

# ── Index a product catalog ──────────────────────────────────────────
def index_product_catalog(
    catalog: list[dict],   # [{product_id, image_path, name, category, price}]
):
    points = []
    images = [Image.open(p["image_path"]).convert("RGB") for p in catalog]
    embeds = embed_images_batch(images)

    for product, embedding in zip(catalog, embeds):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "product_id": product["product_id"],
                "name":       product["name"],
                "category":   product["category"],
                "price":      product["price"],
                "image_path": product["image_path"],
            },
        ))

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} products")

# ── Text-to-image search ──────────────────────────────────────────────
def search_by_text(query: str, k: int = 10, category: str | None = None) -> list[dict]:
    """Find products matching a text description"""
    query_embedding = embed_text(query)

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_embedding.tolist(),
        query_filter=search_filter,
        limit=k,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results]

# ── Image-to-image search ─────────────────────────────────────────────
def search_by_image(image_path: str, k: int = 10) -> list[dict]:
    """Find products visually similar to an uploaded image"""
    query_embedding = embed_image(image_path)
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_embedding.tolist(),
        limit=k + 1,   # +1 because the image itself may be in the index
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results[:k]]

# ── FastAPI endpoints ─────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File
import io

app = FastAPI()

@app.get("/search/text")
async def text_search(q: str, k: int = 10, category: str | None = None):
    """GET /search/text?q=red+running+shoes&k=10"""
    results = search_by_text(q, k=k, category=category)
    return {"query": q, "results": results}

@app.post("/search/image")
async def image_search(file: UploadFile = File(...), k: int = 10):
    """POST /search/image — upload image, find similar products"""
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    embedding= embed_images_batch([image])[0]

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=embedding.tolist(),
        limit=k,
        with_payload=True,
    )
    return {"results": [{"score": r.score, **r.payload} for r in results]}
```

### CLIP Variants Comparison

```python
"""
CLIP MODEL COMPARISON TABLE
────────────────────────────────────────────────────────────────────────────
Model                    Params  Dim  Speed   Quality  Use Case
────────────────────────────────────────────────────────────────────────────
clip-vit-base-patch32    151M    512  ⭐⭐⭐⭐⭐  ⭐⭐⭐    Dev, fast prototyping
clip-vit-large-patch14   427M    768  ⭐⭐⭐    ⭐⭐⭐⭐   Production, better quality
SigLIP-base              86M     512  ⭐⭐⭐⭐⭐  ⭐⭐⭐⭐   Better zero-shot, Google
SigLIP-large             878M    1152 ⭐⭐      ⭐⭐⭐⭐⭐  Best zero-shot available
FashionCLIP              151M    512  ⭐⭐⭐⭐⭐  ⭐⭐⭐⭐⭐  Fashion domain only
BioCLIP                  86M     512  ⭐⭐⭐⭐   ⭐⭐⭐⭐⭐  Biomedical images
────────────────────────────────────────────────────────────────────────────

SigLIP vs CLIP:
  CLIP uses softmax loss (needs large batch to work well)
  SigLIP uses sigmoid loss (treats each pair independently)
  → SigLIP better quality at same compute, especially with smaller batches
  → Use SigLIP for new projects (2024+)

FashionCLIP:
  Fine-tuned on 700K fashion image-text pairs from Farfetch
  Understands fashion-specific vocabulary:
    "midi dress", "bomber jacket", "A-line skirt", "slim fit chinos"
  Outperforms vanilla CLIP on fashion retrieval by ~15% on standard benchmarks
  Same API: just change MODEL_NAME = "patrickjohncyh/fashion-clip"
"""

# SigLIP usage (better than CLIP for new projects)
from transformers import AutoProcessor, AutoModel

siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
siglip_proc  = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

def embed_image_siglip(image: Image.Image) -> np.ndarray:
    inputs   = siglip_proc(images=image, return_tensors="pt")
    with torch.no_grad():
        features = siglip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].numpy()

def embed_text_siglip(text: str) -> np.ndarray:
    inputs   = siglip_proc(text=[text], return_tensors="pt", padding="max_length")
    with torch.no_grad():
        features = siglip_model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].numpy()
```

### BLIP-2 — CLIP + LLM = Image Captioning & VQA

```python
"""
BLIP-2: vision encoder (CLIP-style) + Q-Former bridge + frozen LLM
  = you can ask questions about images in natural language

Use cases:
  Auto-caption product images (no human copywriter)
  VQA (Visual Question Answering) for product specs
  Multimodal RAG: generate text from image → feed to RAG pipeline
"""

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device     = "cuda" if torch.cuda.is_available() else "cpu"
blip2_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2      = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto",
)

def generate_caption(image_path: str) -> str:
    """Auto-generate product description from image"""
    image  = Image.open(image_path).convert("RGB")
    inputs = blip2_proc(image, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        ids    = blip2.generate(**inputs, max_new_tokens=100)
    caption = blip2_proc.decode(ids[0], skip_special_tokens=True)
    return caption.strip()

def visual_qa(image_path: str, question: str) -> str:
    """Ask a question about a product image"""
    image  = Image.open(image_path).convert("RGB")
    inputs = blip2_proc(image, text=question, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        ids    = blip2.generate(**inputs, max_new_tokens=100)
    answer = blip2_proc.decode(ids[0], skip_special_tokens=True)
    return answer.strip()

# Demo:
"""
caption = generate_caption("product_images/jacket.jpg")
# → "a blue denim jacket with brass buttons and a chest pocket"

answer = visual_qa("product_images/jacket.jpg", "What colour is the jacket?")
# → "blue"

answer = visual_qa("product_images/laptop.jpg", "Does it have a numeric keypad?")
# → "yes, it has a full numeric keypad on the right side"
"""
```

### Fine-Tuning CLIP on a Custom Domain

```python
"""
When to fine-tune CLIP vs use as-is:
  Use as-is:       general product images, standard categories
  Fine-tune:       highly specialised domain (medical scans, satellite images,
                   industrial parts, niche fashion categories)
  
Fine-tuning CLIP:
  Start from pre-trained CLIP weights (don't train from scratch)
  Provide domain-specific (image, text) pairs
  Contrastive loss: matching pairs pulled together, non-matching pushed apart
  Result: embeddings better calibrated for your domain vocabulary
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader

class DomainCLIPDataset(Dataset):
    """
    Domain fine-tuning dataset: pairs of (image, text_description).
    Example for medical domain:
      (chest_xray.jpg, "chest X-ray showing bilateral infiltrates")
    """
    def __init__(self, pairs: list[dict], processor: CLIPProcessor):
        self.pairs     = pairs   # [{image_path, caption}]
        self.processor = processor

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        pair   = self.pairs[idx]
        image  = Image.open(pair["image_path"]).convert("RGB")
        inputs = self.processor(
            text=pair["caption"],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

def finetune_clip_domain(
    pairs:       list[dict],
    output_dir:  str,
    epochs:      int = 3,
    batch_size:  int = 16,
    lr:          float = 1e-5,
):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset   = DomainCLIPDataset(pairs, processor)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            batch    = {k: v.to(device) for k, v in batch.items()}
            outputs  = model(**batch)

            # CLIP's contrastive loss is already computed by the model
            # logits_per_image: similarity matrix [batch × batch]
            logits   = outputs.logits_per_image
            labels   = torch.arange(len(logits)).to(device)

            # Cross-entropy both ways: image→text and text→image
            loss_i2t = nn.CrossEntropyLoss()(logits, labels)
            loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
            loss     = (loss_i2t + loss_t2i) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuned CLIP saved to {output_dir}")
```

---

## 10.4 — Fashion AI & Domain-Specific CLIP

### Full Fashion Visual Search Engine

```python
"""
Project: Fashion Visual Search Engine
  → Text search: "blue midi dress under ₹2000"
  → Image search: upload photo, find similar items
  → Multimodal: text + image combined query

Stack: FashionCLIP + Qdrant + FastAPI + React
"""

# ── FashionCLIP: same API, fashion-tuned weights ──────────────────────
FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"

fashion_model     = CLIPModel.from_pretrained(FASHION_CLIP_MODEL)
fashion_processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL)
fashion_model.eval()

def fashion_embed_image(image: Image.Image) -> np.ndarray:
    inputs   = fashion_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = fashion_model.get_image_features(**inputs)
    return (features / features.norm(p=2, dim=-1, keepdim=True))[0].numpy()

def fashion_embed_text(text: str) -> np.ndarray:
    inputs   = fashion_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = fashion_model.get_text_features(**inputs)
    return (features / features.norm(p=2, dim=-1, keepdim=True))[0].numpy()

# ── Outfit recommendation ─────────────────────────────────────────────
def recommend_outfit_completions(
    worn_items: list[str],    # image paths of items already chosen
    catalog:    list[dict],   # all available items
    k:          int = 5,
) -> list[dict]:
    """
    Given worn items → find items that complete the outfit.
    Technique: average embed all worn items → search for similar items.
    """
    images  = [Image.open(p).convert("RGB") for p in worn_items]
    embeds  = [fashion_embed_image(img) for img in images]
    outfit  = np.mean(embeds, axis=0)   # average outfit embedding
    outfit /= np.linalg.norm(outfit)    # re-normalise

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=outfit.tolist(),
        limit=k + len(worn_items),   # exclude worn items from results
    )

    # Filter out items already in the outfit
    worn_ids = set(Path(p).stem for p in worn_items)
    filtered = [r for r in results if r.payload.get("product_id") not in worn_ids]
    return [{"score": r.score, **r.payload} for r in filtered[:k]]

# ── Attribute extraction with BLIP-2 ────────────────────────────────
FASHION_ATTRIBUTES = [
    "color", "pattern", "sleeve length", "neckline", "material", "fit", "occasion"
]

async def extract_product_attributes(image_path: str) -> dict:
    """Use BLIP-2 to extract structured attributes from product image"""
    attributes = {}
    for attr in FASHION_ATTRIBUTES:
        question = f"What is the {attr} of this clothing item?"
        answer   = visual_qa(image_path, question)
        attributes[attr] = answer
    return attributes

# ── Multi-modal search: text + image combined ─────────────────────────
def search_multimodal(
    text_query: str | None = None,
    image_path: str | None = None,
    text_weight: float = 0.5,
    k: int = 10,
) -> list[dict]:
    """
    Combine text and image signals for richer search.
    text_weight=0.5: equal weight
    text_weight=0.8: text dominates (find items matching a description)
    text_weight=0.2: image dominates (find similar items to uploaded photo)
    """
    embeddings = []
    weights    = []

    if text_query and text_query.strip():
        embeddings.append(fashion_embed_text(text_query))
        weights.append(text_weight)

    if image_path:
        img = Image.open(image_path).convert("RGB")
        embeddings.append(fashion_embed_image(img))
        weights.append(1.0 - text_weight)

    if not embeddings:
        raise ValueError("Provide at least one of text_query or image_path")

    # Weighted combination
    weights_arr = np.array(weights) / sum(weights)
    combined    = sum(w * e for w, e in zip(weights_arr, embeddings))
    combined   /= np.linalg.norm(combined)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=combined.tolist(),
        limit=k,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results]

# ── FastAPI endpoints ─────────────────────────────────────────────────
@app.post("/search/multimodal")
async def multimodal_search_endpoint(
    text_query:  str | None   = None,
    text_weight: float        = 0.5,
    k:           int          = 10,
    image:       UploadFile | None = File(None),
):
    image_path = None
    if image:
        contents = await image.read()
        img      = Image.open(io.BytesIO(contents)).convert("RGB")
        tmp      = f"/tmp/{uuid.uuid4()}.jpg"
        img.save(tmp)
        image_path = tmp

    results = search_multimodal(
        text_query=text_query,
        image_path=image_path,
        text_weight=text_weight,
        k=k,
    )
    return {"results": results}
```

---

## 10.5 — Other Useful Pre-Trained Models

### Sentence Transformers — Better Embeddings for Text

```python
"""
For RAG and semantic search: use Sentence Transformers instead of CLIP for text-only tasks.
CLIP text encoder is optimised for image-text matching, not pure text similarity.
Sentence Transformers are specifically trained for text-text similarity.
"""

from sentence_transformers import SentenceTransformer

# ── Model options ─────────────────────────────────────────────────────
"""
all-MiniLM-L6-v2:           384-dim, 22M params, fastest, English only
all-mpnet-base-v2:          768-dim, 110M params, best English quality
paraphrase-multilingual-MiniLM-M12:  384-dim, 50 languages (including Hindi)
BAAI/bge-m3:                1024-dim, best multilingual, state of art 2026
"""

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode a batch (fast, handles padding automatically)
sentences = [
    "Where is my order?",
    "Track my package",
    "What is the return policy?",
    "How do I send something back?",
]
embeddings = model.encode(sentences, batch_size=32, normalize_embeddings=True)
print(embeddings.shape)   # (4, 384)

# Find similarities
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
print(f"Q1 vs Q2 (both about tracking): {sim_matrix[0,1]:.3f}")   # high
print(f"Q1 vs Q3 (different topics):    {sim_matrix[0,2]:.3f}")   # low
```

### YOLO — Real-Time Object Detection

```python
"""
YOLO: detect AND localise multiple objects in an image in real time.
CLIP: classify what's in the image (no bounding boxes).

Use YOLO when: you need bounding boxes, multiple objects, real-time video.
Use CLIP when: you need flexible text-driven search, zero-shot capability.

Use case for AI engineers:
  Product damage detection: find scratches/dents in returned product photos
  Invoice processing: detect tables, stamps, signatures
  Content moderation: detect specific objects that violate policy
"""

from ultralytics import YOLO

# Download + load (auto-downloads on first run)
model = YOLO("yolov8n.pt")     # nano: fastest, smallest (6M params)
# Or:    YOLO("yolov8x.pt")   # extra large: most accurate (68M params)

# Detect objects in an image
results = model("product_images/damaged_item.jpg")

for result in results:
    for box in result.boxes:
        class_id    = int(box.cls[0])
        class_name  = model.names[class_id]
        confidence  = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Detected: {class_name} ({confidence:.2f}) at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

# Fine-tune YOLO on custom data (few lines)
model = YOLO("yolov8n.pt")
model.train(
    data="custom_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)
```

### EasyOCR — Text Extraction from Images

```python
"""
EasyOCR: extract text from images (invoices, receipts, forms, screenshots).
Supports 80+ languages including Hindi, Tamil, Telugu, Gujarati.
Use case: "Smart Document Intelligence" — upload any form, extract structured data.
"""

import easyocr

# Init (downloads model on first run: ~1.5GB)
reader = easyocr.Reader(
    ["en", "hi"],    # languages: English + Hindi
    gpu=False,       # CPU inference (slow but works anywhere)
)

def extract_text_from_image(image_path: str) -> str:
    results = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results)

def extract_invoice_structured(image_path: str) -> dict:
    """Extract raw text then use GPT-4o to structure it"""
    raw_text = extract_text_from_image(image_path)

    # Use GPT-4o to extract structured data from OCR output
    import openai
    client = openai.OpenAI()
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Extract structured data from this invoice text:\n{raw_text}",
        }],
        response_format=InvoiceData,   # Pydantic model from earlier
    )
    return response.choices[0].message.parsed
```

### Anomaly Detection on LLM Outputs

```python
"""
Use IsolationForest to detect unusual LLM responses before showing to users.
Features to extract per LLM response:
  - Length (word count)
  - Sentiment score
  - Lexical diversity (unique words / total words)
  - Contains numbers? (factual answers often have numbers)
  - Perplexity proxy (n-gram repetition rate)
"""

from sklearn.ensemble import IsolationForest
import numpy as np
import re

def extract_response_features(response: str) -> np.ndarray:
    words          = response.split()
    unique_words   = set(w.lower() for w in words)
    has_number     = bool(re.search(r'\d+', response))
    sentences      = re.split(r'[.!?]+', response)
    avg_sent_len   = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
    repetition     = len(words) / max(len(unique_words), 1)

    return np.array([
        len(words),                             # response length
        len(unique_words) / max(len(words), 1), # lexical diversity (0-1)
        1.0 if has_number else 0.0,             # contains factual numbers
        avg_sent_len,                           # avg sentence length
        repetition,                             # repetition rate (high = bad)
        len(sentences),                         # number of sentences
    ])

class LLMOutputAnomalyDetector:
    """
    Train on normal LLM responses, detect anomalous ones.
    Useful for: catching broken responses, hallucination signals,
                model degradation after prompt changes.
    """
    def __init__(self, contamination: float = 0.05):
        self.model       = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted   = False

    def fit(self, responses: list[str]):
        features = np.array([extract_response_features(r) for r in responses])
        self.model.fit(features)
        self.is_fitted = True
        print(f"Anomaly detector fitted on {len(responses)} normal responses")

    def predict(self, response: str) -> dict:
        features    = extract_response_features(response).reshape(1, -1)
        label       = self.model.predict(features)[0]
        score       = self.model.score_samples(features)[0]
        is_anomalous= label == -1
        return {
            "is_anomalous": is_anomalous,
            "anomaly_score":round(float(score), 4),   # more negative = more anomalous
            "features": {
                "word_count":       int(features[0][0]),
                "lexical_diversity":round(float(features[0][1]), 3),
                "has_numbers":      bool(features[0][2]),
            },
        }
```

---

## 🏗️ Phase 10 Projects

### Project 10A — Fashion Visual Search Engine (15 hrs)

```
WHAT YOU BUILD:
  Full-stack fashion search engine using FashionCLIP + Qdrant.
  Users can search by text OR upload a photo to find similar products.

STEPS:
  1. Data: download a fashion dataset (Kaggle Fashion Product Images, ~44K items)
     or use a demo subset with 500 products from DeepFashion

  2. Indexing pipeline:
     for each product image → FashionCLIP embed → store in Qdrant
     store payload: product_id, name, category, price, image_path

  3. FastAPI endpoints:
     GET  /search/text?q=blue+midi+dress&category=dresses&k=10
     POST /search/image  (upload image → find similar)
     POST /search/multimodal  (text + image combined)

  4. Auto-caption with BLIP-2 (optional, CPU is slow → use GPT-4o Vision instead)
     For each product: generate caption from image → store in payload

  5. React UI:
     Search bar (text)
     Image upload button
     Product grid with image + name + price + similarity score
     Filter sidebar (category, price range)

  6. Deploy on Cloud Run:
     Dockerfile: CLIP model baked in (~1GB image)
     Cloud Run: --memory 4Gi (model needs RAM)
     Qdrant: run on separate Cloud Run service or Qdrant Cloud free tier

SHOWABLE:
  Text: "slim fit white button shirt for men" → correct results
  Image: upload a photo of a dress → similar dresses appear
  Multimodal: "red dress" + upload → finds red dresses similar to uploaded
```

### Project 10B — Multimodal Product Q&A (10 hrs)

```
WHAT YOU BUILD:
  Upload a product image → BLIP-2 captions it → RAG pipeline answers questions.

FLOW:
  User uploads product image
  BLIP-2: generate detailed caption ("a stainless steel water bottle, 1 litre,
           with black silicone grip and flip-top lid")
  Optionally also have: product description text from catalog
  Combined context → stored in Qdrant (both image embedding + text embedding)
  User asks: "Is this BPA free?" "What's the capacity?" "Can I put it in the dishwasher?"
  RAG pipeline retrieves from Qdrant → LLM answers based on caption + product text

STACK: BLIP-2, FashionCLIP/CLIP, Qdrant, FastAPI, React
```

### Project 10C — Smart Document Intelligence (8 hrs)

```
WHAT YOU BUILD:
  Upload invoice/receipt/form → EasyOCR extracts text → LLM structures it.
  CLIP classifies document type → routes to correct extraction template.

FLOW:
  User uploads any document image (invoice, receipt, delivery note, form)
  EasyOCR: extract raw text from image
  CLIP: classify document type
    embed image → compare cosine similarity to pre-embedded examples:
    "an invoice document", "a receipt", "a delivery note", "a tax form"
    → pick closest category
  Route to correct extraction:
    invoice   → extract: vendor, amount, date, line items, GST
    receipt   → extract: store, total, items, payment method
    delivery  → extract: tracking number, recipient, delivery date

STACK: EasyOCR, CLIP, GPT-4o (structured extraction), FastAPI, AWS Lambda
```

---

## 📋 Interview Cheat Sheet

**Q: What is CLIP and how does it work?**
```
CLIP = Contrastive Language-Image Pre-training (OpenAI, 2021).

Architecture:
  Dual encoder: image encoder (ViT) + text encoder (Transformer)
  Both encode to the SAME embedding space (512-dim for base-patch32)

Training:
  400M internet image-text pairs
  Contrastive loss: pull matching (image, text) pairs together
                    push non-matching pairs apart

Why it's powerful:
  Image "red sneaker" and text "a photo of a red sneaker" → high cosine sim
  This means: text query → find matching images (no training on those labels)
  = zero-shot classification: any text description = a new category

Key property:
  CLIP(image).dot(CLIP(text)) ≈ semantic similarity
  Normalise both to unit length → cosine sim = dot product
```

**Q: When would you use CLIP vs a CNN classifier?**
```
CNN classifier:
  Trained on fixed categories (ImageNet: 1000 classes)
  Adding new category: need labelled data + retrain
  Best for: high-accuracy classification of known categories
  Speed: very fast, small model

CLIP:
  Works with any text description = infinite categories
  Adding new category: just write the text description
  Best for: flexible search, zero-shot classification, multimodal RAG
  Speed: slightly slower (dual encoder)

Use CNN when:
  Fixed category set (cat/dog/car)
  Need highest accuracy on those categories
  Production with known categories and labelled training data

Use CLIP when:
  Dynamic categories (e-commerce: 100K product types)
  Text-driven search ("find products similar to this description")
  No labelled training data available
  Multimodal: images + text in same vector space
```

**Q: What is contrastive learning?**
```
Training objective: given a (image, text) pair:
  Positive pair: this image and its caption → maximise similarity
  Negative pairs: this image with all other captions in the batch → minimise similarity

The loss function: InfoNCE loss (same idea as CLIP loss)
  similarity_matrix = image_embeddings × text_embeddings.T   # [batch × batch]
  Labels = diagonal (matching pairs)
  CrossEntropyLoss on rows (image→text) and columns (text→image)
  Average both losses

Key insight: you don't need human labels.
  The caption IS the label.
  400M internet images all come with alt-text/captions = free labels.

SigLIP improvement:
  CLIP: softmax over full batch (needs large batches)
  SigLIP: sigmoid on each pair independently (works with small batches)
```

**Q: When would you use IsolationForest on LLM outputs?**
```
IsolationForest detects anomalies without labelled examples of "what's bad".

How it works:
  Build a random forest that tries to isolate each data point
  Normal points: hard to isolate (need many splits)
  Anomalies: easy to isolate (few splits, they're far from the crowd)
  Score: average path length → short = anomaly

Use for LLM outputs:
  Train on 1000 good responses (features: length, diversity, sentences, repetition)
  In production: score every new response
  Flag anomalies for human review BEFORE showing to users

Why this matters:
  Hallucinated responses are often: very short, repetitive, or unusually long
  Silent degradation: prompt change → model starts giving unusual responses
  No labels needed: just train on "normal" responses

Threshold: contamination=0.05 flags ~5% of responses as anomalous
  Tune based on your tolerance for false positives vs false negatives
```

**Q: When do you use ML vs an LLM for a task?**
```
Use ML (sklearn, small neural nets) when:
  Task is clearly defined + has fixed output space (2-20 classes)
  Have 100+ labelled training examples
  Need real-time response (< 10ms) — LLM is 1-30 seconds
  Cost-sensitive at scale (1M calls/day → LLM = $$$$, ML = $)
  Audit/compliance: need explainable decision (SHAP)
  Privacy: can't send data to external API

Examples: intent classification, document type routing,
          spam detection, anomaly detection, cluster similar docs

Use LLM when:
  Open-ended output (generation, summarisation, explanation)
  Complex reasoning requiring world knowledge
  Zero or few labelled examples
  Low volume (< 10K/day — cost is manageable)
  Nuanced tasks where rigid rules fail

Rule of thumb:
  "Can I write an if-else or train a simple classifier?"
  → Yes: use ML
  → No: use LLM

HireSense example:
  Resume screening: RandomForest + SHAP (ML) — fast, explainable, cheap
  Interview conversation: LangGraph + GPT-4o (LLM) — needs reasoning
```

---

## 🃏 Quick Revision Cards

```
CARD 1: Learning Types
  Supervised:     labelled (X → y), classify/regress
  Unsupervised:   no labels, find structure (cluster, reduce)
  Self-supervised: labels from data itself (CLIP, LLM pretraining)
  RLHF/DPO:       human preference → alignment
  Your job:       pick the right type for each task

CARD 2: Sklearn Toolbox for AI Engineers
  RandomForestClassifier: intent/doc type classifier
  LogisticRegression:     binary decisions, fast
  KMeans(n_clusters=8):  cluster support tickets/docs
  PCA(n_components=50):  dimensionality reduction before t-SNE
  IsolationForest:       anomaly detection on LLM outputs
  Pipeline([scaler, clf]): chain preprocessing + model
  cross_val_score:        reliable eval, 5-fold stratified

CARD 3: PyTorch Essentials
  Tensor:   torch.randn(B, D) — B=batch, D=dim
  nn.Module: define model with __init__ + forward()
  nn.Sequential: chain layers
  optimizer.zero_grad() → loss.backward() → optimizer.step()
  torch.no_grad(): inference mode (saves memory, faster)
  model.eval(): disables dropout/batchnorm for inference

CARD 4: Transfer Learning Pattern
  Load pre-trained: model = models.resnet50(weights=IMAGENET1K_V2)
  Freeze backbone: param.requires_grad = False
  Replace head: model.fc = nn.Linear(2048, num_classes)
  Train only head (default) or unfreeze all (if lots of data)
  Rule: < 1000 examples → freeze. > 10K examples → unfreeze later.

CARD 5: CLIP Core
  Architecture:  image encoder (ViT) + text encoder (Transformer)
  Training:      contrastive loss on 400M image-text pairs
  Output:        normalised 512-dim embedding (base-patch32)
  Key property:  embed_image(cat) ≈ embed_text("a photo of a cat")
  Use for:       zero-shot classify, text→image search, multimodal RAG

CARD 6: CLIP Variants
  clip-vit-base-patch32:    fast, 512-dim, general use
  clip-vit-large-patch14:   better, 768-dim, production
  SigLIP-base:              better zero-shot, sigmoid loss
  FashionCLIP:              fashion domain, same API
  BLIP-2:                   CLIP + LLM → image captioning + VQA
  All: same HuggingFace API, just change model name

CARD 7: Visual Search Pipeline
  Index:  image → FashionCLIP embed → Qdrant upsert
  Query:  text → embed_text → qdrant.search → top-K results
  Query:  image → embed_image → qdrant.search → similar items
  Multi:  0.5*embed_text + 0.5*embed_image → normalise → search
  CLIP_DIM = 512 (base) or 768 (large) → Qdrant VectorParams

CARD 8: HuggingFace Ecosystem
  pipeline():   zero-code inference for any task
  AutoModel:    load any model architecture
  AutoTokenizer: tokenise for any model
  Trainer:      fine-tune in < 20 lines
  datasets:     load/split/preprocess cleanly
  SentenceTransformer: better than CLIP for text-only similarity

CARD 9: Pre-Trained Model Selection
  Text embeddings:      SentenceTransformer (all-MiniLM-L6-v2)
  Image+text embed:     CLIP / SigLIP / FashionCLIP
  Image captioning:     BLIP-2 (free) or GPT-4o Vision (better)
  Object detection:     YOLO (ultralytics, 2 lines)
  OCR:                  EasyOCR (80 languages, Hindi works)
  Speech-to-text:       Whisper (local) or Deepgram (API)
  Face similarity:      deepface library
  Anomaly detection:    IsolationForest (sklearn)

CARD 10: ML vs LLM Decision
  Classification (fixed categories, labelled data) → sklearn
  Clustering (find structure, no labels) → KMeans
  Anomaly detection → IsolationForest
  Zero-shot classification (no data) → CLIP or LLM
  Text-image search → CLIP + Qdrant
  Generation, reasoning, conversation → LLM
  High volume + simple → ML every time
  Low volume + complex → LLM
```

---

## ✅ Phase 10 Completion Checklist

```
ML FOUNDATIONS
[ ] Explained supervised vs unsupervised vs self-supervised correctly
[ ] Train/val/test split: implemented correctly (scale after split)
[ ] Overfitting: demonstrated by train F1=1.0, val F1=0.58 with deep tree
[ ] Fixed with regularisation (max_depth=8, min_samples_leaf=5)
[ ] Intent classifier: built with TF-IDF + LogisticRegression, tested
[ ] KMeans clustering on embeddings: labels + representative items found
[ ] IsolationForest: trained on normal responses, flags anomalous ones
[ ] cross_val_score: 5-fold stratified, interpreted mean ± std
[ ] "ML vs LLM" decision: can confidently explain when to use each

DEEP LEARNING BASICS
[ ] Forward pass explained: Linear → ReLU → Dropout → Linear
[ ] Backprop explained: chain rule, gradients flow backwards
[ ] PyTorch tensor operations: shape, dtype, device, matmul
[ ] autograd: requires_grad=True, .backward(), .grad
[ ] IntentClassifier with nn.Sequential built and trained
[ ] Dataset + DataLoader custom class implemented
[ ] Training loop: zero_grad → forward → loss → backward → step
[ ] Train/val loss curves logged, overfitting identified
[ ] Transfer learning: ResNet50, freeze backbone, replace head
[ ] HuggingFace pipeline: 3+ different tasks tested

CLIP
[ ] CLIP architecture explained: dual encoder + contrastive loss
[ ] embed_image() and embed_text() implemented correctly (normalised!)
[ ] Zero-shot classification: 5+ categories, no training
[ ] Cosine similarity between image and text embeddings computed
[ ] Qdrant collection created: VectorParams(size=512, distance=COSINE)
[ ] Text-to-image search working end-to-end
[ ] Image-to-image search working end-to-end
[ ] SigLIP tested: is it better than CLIP on your dataset?
[ ] FashionCLIP: same API, fashion-specific queries tested
[ ] BLIP-2: image captioning + VQA working (even on CPU, slowly)
[ ] CLIP fine-tuning: contrastive loss code understood and run

FASHION AI
[ ] Product catalog indexed in Qdrant (100+ items minimum)
[ ] Text search: "blue midi dress" → correct results returned
[ ] Image search: upload photo → similar items returned
[ ] Multimodal: weighted combination of text + image embeddings
[ ] Outfit completion: average outfit embedding → find complementary items
[ ] FashionCLIP vs vanilla CLIP: quality difference demonstrated

OTHER MODELS
[ ] SentenceTransformer: encoded batch, similarity matrix computed
[ ] YOLO: object detection on 3 test images, bounding boxes printed
[ ] EasyOCR: text extracted from invoice/receipt image correctly
[ ] LLM output anomaly detection: IsolationForest trained + predicting

PROJECTS
[ ] Project 10A: Fashion search engine deployed (Cloud Run or localhost)
    Text search, image search, multimodal search all working
    React UI with product grid
[ ] Project 10B: Multimodal Q&A (BLIP-2 caption → RAG → LLM answer)
[ ] Project 10C: Document intelligence (OCR → CLIP route → LLM extract)
```

---

*Phase 10 — ML/DL Essentials for AI Engineers | GenAI + LLMOps Engineering Roadmap 2026*
*Covers: Supervised/Unsupervised Learning · PyTorch · Transfer Learning · CLIP · SigLIP · FashionCLIP · BLIP-2 · Visual Search · YOLO · EasyOCR · Anomaly Detection*
*Next: Phase 11 — Capstone Projects (pick 2–3, make them production-grade)*
