"""Lazy loading keeps the UI available even when ML dependencies are absent."""

import hashlib
import pickle
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_TOKENS = 350


class SentimentEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.version = None

    def _load(self):
        if self.model is not None:
            return
        import numpy as np
        import tensorflow as tf
        import tf_keras as keras
        from tf_keras.preprocessing.text import Tokenizer, text_to_word_sequence
        from tf_keras.preprocessing.sequence import pad_sequences

        class AttentionLayer(keras.layers.Layer):
            def call(self, inputs):
                query, value = inputs
                scores = tf.nn.softmax(
                    tf.matmul(query, value, transpose_b=True), axis=-1
                )
                return tf.matmul(scores, value)

        # Only trusted repository artifacts are deserialized; uploads are CSV text only.
        class TokenizerUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "keras.preprocessing.text" and name == "Tokenizer":
                    return Tokenizer
                return super().find_class(module, name)

        path = ROOT / "enhanced_sentiment_analysis_model.h5"
        if path.stat().st_size < 1024:
            raise RuntimeError("Model is a Git LFS pointer. Run git lfs pull.")
        with (ROOT / "tokenizer.pickle").open("rb") as f:
            self.tokenizer = TokenizerUnpickler(f).load()
        with (ROOT / "label_encoder.pickle").open("rb") as f:
            self.encoder = pickle.load(f)
        self.model = keras.models.load_model(
            path, custom_objects={"AttentionLayer": AttentionLayer}, compile=False
        )
        self.np, self.pad, self.words = np, pad_sequences, text_to_word_sequence
        self.version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    def metadata(self):
        return {
            "name": "Apple product review sentiment",
            "max_tokens": MAX_TOKENS,
            "loaded": self.model is not None,
            "version": self.version,
            "labels": [str(x) for x in self.encoder.classes_]
            if self.model is not None
            else [],
            "validation": "No held-out evaluation artifact is included in this repository.",
        }

    def analyze(self, reviews):
        with self.lock:
            self._load()
            sequences = self.tokenizer.texts_to_sequences(reviews)
            padded = self.pad(
                sequences, maxlen=MAX_TOKENS, padding="post", truncating="post"
            )
            scores = self.np.asarray(self.model(padded, training=False))
            if scores.ndim != 2 or scores.shape[1] != len(self.encoder.classes_):
                raise RuntimeError("Model output does not match saved labels")
            if (
                not self.np.isfinite(scores).all()
                or (scores < 0).any()
                or (scores > 1).any()
            ):
                raise RuntimeError("Invalid model scores")
            results = []
            for text, sequence, vector in zip(reviews, sequences, scores):
                tokens = self.words(
                    text,
                    filters=self.tokenizer.filters,
                    lower=self.tokenizer.lower,
                    split=self.tokenizer.split,
                )
                limit = self.tokenizer.num_words
                known = sum(
                    1
                    for t in tokens
                    if t in self.tokenizer.word_index
                    and (not limit or self.tokenizer.word_index[t] < limit)
                )
                coverage = known / len(tokens) if tokens else 0
                index = int(self.np.argmax(vector))
                confidence = float(vector[index])
                warnings = []
                if confidence < 0.7:
                    warnings.append("Low model score")
                if coverage < 0.5:
                    warnings.append("Limited vocabulary coverage")
                if len(sequence) > MAX_TOKENS:
                    warnings.append("Input truncated to 350 tokens")
                if not sequence:
                    warnings.append("No recognized tokens")
                results.append(
                    {
                        "text": text,
                        "label": str(self.encoder.classes_[index]),
                        "confidence": confidence,
                        "coverage": coverage,
                        "tokens": len(sequence),
                        "warnings": warnings,
                        "needs_review": bool(warnings),
                        "model_version": self.version,
                        "scores": {
                            str(label): float(value)
                            for label, value in zip(self.encoder.classes_, vector)
                        },
                    }
                )
            return results
