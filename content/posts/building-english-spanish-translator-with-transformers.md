---
title: "Building an English-Spanish Translator with Transformers"
date: "2024-07-14"
description: "A deep dive into building a neural machine translation model using transformer architecture. Learn how to preprocess parallel text data, implement attention mechanisms, and train a production-ready translation system."
tags: ["Machine Learning", "NLP", "Transformers", "Python", "TensorFlow"]
image: ""
featured: true
category: "Machine Learning"
---

Language barriers have always been a challenge in our increasingly connected world. While traditional translation methods rely on complex rule-based systems or statistical models, modern deep learning approaches—particularly transformer architectures—have revolutionized machine translation. In this post, I'll walk you through building a sequence-to-sequence transformer model that translates between English and Spanish.

## The Power of Transformers

Transformers have become the gold standard for natural language processing tasks. Unlike recurrent neural networks (RNNs) that process sequences sequentially, transformers use attention mechanisms to capture relationships between words regardless of their distance in the sentence. This parallel processing capability makes them both faster and more effective at understanding context.

## Project Overview

We'll build an end-to-end translation system through four main steps:

1. **Data Preprocessing** - Download and prepare parallel text datasets
2. **Model Architecture** - Implement positional embeddings, encoder, and decoder layers
3. **Training** - Train the model on English-Spanish sentence pairs
4. **Inference** - Use the trained model to translate new sentences

## Setting Up the Environment

First, let's import the necessary libraries. We'll be using TensorFlow and Keras for building our neural network:

```python
import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
```

## Step 1: Data Preprocessing

### Downloading the Dataset

We'll use a publicly available English-to-Spanish translation dataset from Anki, which contains thousands of parallel sentence pairs:

```python
text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"
```

### Parsing Sentence Pairs

Each line in our dataset contains an English sentence and its Spanish translation, separated by a tab. We'll add special `[start]` and `[end]` tokens to the Spanish sentences to help the model learn when to begin and end translations:

```python
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
    
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((eng, spa))
```

Here are some example pairs from our dataset:

```python
for _ in range(5):
    print(random.choice(text_pairs))
```

```
('I think Tom is working now.', '[start] Creo que ahora Tomás trabaja. [end]')
("I'm very interested in classical literature.", '[start] Me interesa mucho la literatura clásica. [end]')
('I appreciate you.', '[start] Te tengo cariño. [end]')
('Do you want to watch this program?', '[start] ¿Quieres ver este programa? [end]')
('We just have to stick together.', '[start] Sólo tenemos que permanecer juntos. [end]')
```

### Splitting the Data

Let's divide our data into training, validation, and test sets. We'll use 70% for training, 15% for validation, and 15% for testing:

```python
random.shuffle(text_pairs)

num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
num_test_samples = len(text_pairs) - num_val_samples - num_train_samples

train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```

```
118964 total pairs
83276 training pairs
17844 validation pairs
17844 test pairs
```

### Vectorizing Text Data

Neural networks work with numbers, not text. We'll use Keras's `TextVectorization` layer to convert our sentences into sequences of integers, where each integer represents a word in our vocabulary.

The English layer uses default preprocessing (removing punctuation and splitting on whitespace), while the Spanish layer includes custom handling for the inverted question mark (¿):

```python
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]

eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)
```

### Formatting the Dataset

At each training step, our model predicts the next word in the target sequence using the source sentence and all previous target words. We format our data accordingly:

```python
def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:]
    )

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
```

## Step 2: Building the Transformer

### Positional Embedding Layer

Since transformers don't inherently understand word order, we need positional embeddings to encode each word's position in the sequence:

```python
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
```

### Transformer Encoder

The encoder processes the source language (English) and creates contextual representations:

```python
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        else:
            padding_mask = None
            
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
```

### Transformer Decoder

The decoder generates the target language (Spanish) one word at a time, attending to both the encoder output and previously generated words:

```python
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = causal_mask
            
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
```

### Complete Transformer Model

Finally, we combine all components into a complete sequence-to-sequence model:

```python
embed_dim = 256
latent_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)
```

## Step 3: Training the Model

Now we can train our transformer. We'll use sparse categorical crossentropy as our loss function since we're predicting word indices:

```python
transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

transformer.fit(train_ds, epochs=30, validation_data=val_ds)
```

## Step 4: Inference - Translating New Sentences

To translate a new sentence, we use a technique called "greedy decoding" - generating one word at a time by always choosing the most probable next word:

```python
spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
            
    return decoded_sentence.replace("[start] ", "").replace(" [end]", "")
```

Let's test our model with some example translations:

```python
test_sentences = [
    "I love machine learning",
    "The weather is beautiful today",
    "Can you help me with this project?",
    "What time is the meeting?"
]

for sentence in test_sentences:
    translation = decode_sequence(sentence)
    print(f"English: {sentence}")
    print(f"Spanish: {translation}\n")
```

## Key Takeaways

Building a transformer-based translation system taught me several important lessons:

1. **Data preprocessing is crucial** - Clean, well-formatted parallel text makes a huge difference in translation quality
2. **Attention mechanisms are powerful** - They allow the model to focus on relevant parts of the input when generating each output word
3. **Positional encoding matters** - Since transformers process all words simultaneously, they need explicit position information
4. **Inference requires careful implementation** - Greedy decoding is simple but effective for generating translations

## Next Steps

This implementation serves as a solid foundation, but there's room for improvement:

- **Implement beam search** instead of greedy decoding for better translations
- **Add attention visualization** to understand what the model focuses on
- **Try byte-pair encoding (BPE)** to handle unknown words better
- **Experiment with pre-trained models** like BERT or GPT for transfer learning
- **Scale to larger datasets** for improved performance on diverse text

Neural machine translation has come a long way, and transformers have been at the forefront of this revolution. Whether you're building a production translation system or just exploring NLP, understanding these architectures opens up a world of possibilities in language AI.

The complete code for this project is available on my GitHub. Feel free to experiment with different languages, dataset sizes, and model architectures!
