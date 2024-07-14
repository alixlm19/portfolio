import CodeBlock from "@/components/CodeBlock";
import Image from "next/image";

export const products = [
    {
        href: "https://github.com/alixlm19/Image-Classifier-on-Noisy-Labels",
        title: "Image Classifier on Noisy Labels",
        description:
            "Implementation of the Multi-Label Fashion Image Classification with Minimal Human Supervision paper from Inoue et al. on the CIFAR-10 dataset.",
        thumbnail: "/images/image_classifier_on_noisy_labels.png",
        images: [
            "/images/image_classifier_on_noisy_labels.png",
        ],
        stack: ["Python", "TensorFlow", "Machine Learning", "CNN", "Classification", "Weak Labels"],
        slug: "noisy-labels-image-classification",
        content: (
            <div>
                <p>
                    We carry out model evaluation and selection for predictive analytics
                    on an imbalanced image data.
                </p>
                <p>
                    We will be dealing with a classification problem, where the training labels are not perfect.
                    This is a common phenomenon in data science. Getting accurate ground true labels
                    can be costly and time-consuming. Sometimes, it is even impossible.
                    The weakly supervised learning is a subject that addresses the issue
                    with imperfect labels. In particular, we are going to train
                    a predictive model where label noises exist.
                </p>{" "}
            </div>
        ),
    },
    {
        href: "https://github.com/alixlm19/English-Spanish-Translation-with-Transformers",
        title: "English-Spanish Translation with Transformers",
        description:
            "This project develops an English-Spanish translation model using transformer architectures. By training on a large corpus of parallel texts, the model achieves high accuracy and fluency. It involves data preprocessing, model training, and evaluation, showcasing the power of transformers in enhancing multilingual communication.",
        thumbnail: "/images/transformer-decoding.gif",
        images: [
            "/images/transformer.png",
            "/images/transformer-decoding.gif",
        ],
        stack: ["Python", "Tensorflow", "Machine Learning", "Transformers", "Autoencoders", "Embeddings"],
        slug: "transformers",
        content: (
            <div>
                <h2>Sequence-to-sequence transformer to translate Spanish to English.</h2>
                <h3>Project Walkthrough:</h3>
                <ul>
                    <li>Step 1. Data Preprocessing
                        <ul>
                            <li>Downloading</li>
                            <li>Parsing</li>
                            <li>Vectorizing text using the Keras <code>TextVectorization</code>
                                layer.</li>
                            <li>Formatting data for training</li>
                        </ul></li>
                    <li>Step 2. Implement a Transformer
                        <ul>
                            <li>A <code>PositionalEmbedding</code> layer </li>
                            <li>A <code>TransformerEncoder</code> layer and a
                                <code>TransformerDecoder</code> layer </li>
                            <li>Sequence all layers together to build a <code>Transformer</code>
                                model</li>
                        </ul></li>
                    <li>Step 3. Train the model</li>
                    <li>Step 4. Inference: Use the trained model to translate new sequences
                    </li>
                </ul>
                <h2>Setup</h2>

                <CodeBlock
                    code={
                        `import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization`}
                    lang="python" />
                <h2>Step 1. Data Preprocessing</h2>
                <h3>Downloading</h3>
                <p>First, download an English-to-Spanish translation dataset from <a href="">Anki.</a></p>
                <CodeBlock
                    code={
                        `text_file = keras.utils.get_file(
    fname=\"spa-eng.zip\",
    origin=\"http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip\",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / \"spa-eng\" / \"spa.txt\"`}
                    lang="python" />
                <h3>Parsing</h3>
                <p>Each line contains an English sentence and its corresponding Spanish

                    sentence. The English sentence is the <em>source sequence</em> and
                    Spanish one is the <em>target sequence</em>. We prepend the token
                    <code>[start]</code> and we append the token <code>[end]</code> to
                    the Spanish sentence.
                </p>
                <CodeBlock
                    code={
                        `with open(text_file) as f:
    lines = f.read().split(\"\\n\")[:-1]
text_pairs = []
for line in lines:
    eng, spa = line.split(\"\\t\")
    spa = \"[start] \" + spa + \" [end]\"
    text_pairs.append((eng, spa))`}
                    lang="python" />
                <p>Here&apos;s what our sentence pairs look like:</p>
                <CodeBlock
                    code={
                        `for _ in range(5):
    print(random.choice(text_pairs))`}
                    lang="python" />
                <pre><code>(&#39;I think Tom is working now.&#39;, &#39;[start] Creo que ahora Tomás trabaja. [end]&#39;)<br />
                    (&quot;I&#39;m very interested in classical literature.&quot;, &#39;[start] Me interesa mucho la literatura clásica. [end]&#39;)<br />
                    (&#39;I appreciate you.&#39;, &#39;[start] Te tengo cariño. [end]&#39;)<br />
                    (&#39;Do you want to watch this program?&#39;, &#39;[start] ¿Quieres ver este programa? [end]&#39;)<br />
                    (&#39;We just have to stick together.&#39;, &#39;[start] Sólo tenemos que permanecer juntos. [end]&#39;)<br />
                </code></pre>
                <p>Now, let&apos;s split the sentence pairs into a training set, a validation set, and a test set.</p>
                <CodeBlock
                    code={
                        `random.shuffle(text_pairs)

num_train_samples = 1000
num_val_samples = 1000
num_test_samples = 1000

num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
num_test_samples = len(text_pairs) - num_val_samples - num_train_samples

train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f\"{len(text_pairs)} total pairs\")
print(f\"{len(train_pairs)} training pairs\")
print(f\"{len(val_pairs)} validation pairs\")
print(f\"{len(test_pairs)} test pairs\")`}
                    lang="python" />
                <pre><code>
                    118964 total pairs<br />
                    83276 training pairs<br />
                    17844 validation pairs<br />
                    17844 test pairs<br />
                </code></pre>
                <h3>Vectorizing the text data</h3>
                <p>A <code>TextVectorization</code> layer vectorizes the text data into
                    integer sequences where each integer represents the index of a word in a
                    vocabulary.</p>
                <p>The English layer will use the default string standardization (strip
                    punctuation characters) and splitting scheme (split on whitespace),
                    while the Spanish layer will use a custom standardization, where we add
                    the character <code>¿</code> to the set of punctuation characters to
                    be stripped.
                </p>
                <CodeBlock
                    code={
                        `strip_chars = string.punctuation + \"¿\"
strip_chars = strip_chars.replace(\"[\", \"\")
strip_chars = strip_chars.replace(\"]\", \"\")

vocab_size = 15000
sequence_length = 20
batch_size = 64

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, \"[%s]\" % re.escape(strip_chars), \"\")

eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode=\"int\", output_sequence_length=sequence_length,
)
spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode=\"int\",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_eng_texts = [pair[0] for pair in train_pairs]

train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)`}
                    lang="python" />
                <h2>Formating</h2>
                <p>Recall that at each training step, the model will seek to predict
                    target words N+1 (and beyond) using the source sentence and the target

                    words 0 to N. As such, the training dataset will yield a tuple
                    <code>(inputs, targets)</code>, where:</p>
                <ul>
                    <li><code>inputs</code> is a dictionary with the keys

                        <code>encoder_inputs</code> and <code>decoder_inputs</code>.
                        <code>decoder_inputs</code> is the vectorized source sentence and
                        <code>encoder_inputs</code> is the target sentence &quot;so far&quot;, that is to
                        say, the words 0 to N used to predict word N+1 (and beyond) in the
                        target sentence.</li>
                    <li><code>target</code> is the target sentence offset by one step: it
                        provides the next words in the target sentence -- what the model will
                        try to predict.</li>
                </ul>
                <CodeBlock
                    code={
                        `def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return ({\"encoder_inputs\": eng, \"decoder_inputs\": spa[:, :-1],}, spa[:, 1:])


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)`}
                    lang="python" />
                <p>Let&apos;s take a quick look at the sequence shapes (we have batches of 64 pairs, and all sequences are 20 steps long):</p>
                <CodeBlock
                    code={
                        `for inputs, targets in train_ds.take(1):
    print(f&apos;inputs[\"encoder_inputs\"].shape: {inputs[\"encoder_inputs\"].shape}&apos;)
    print(f&apos;inputs[\"decoder_inputs\"].shape: {inputs[\"decoder_inputs\"].shape}&apos;)

    print(f\"targets.shape: {targets.shape}\")
                        `}
                    lang="python"
                />
                <pre><code>inputs[&quot;encoder_inputs&quot;].shape: (64, 20)<br />
                    inputs[&quot;decoder_inputs&quot;].shape: (64, 20)<br />
                    targets.shape: (64, 20)<br />
                </code></pre>
                <h2>Step 2. Implement a Transformer</h2>
                <h3>Transformer Overview</h3>
                <ul>
                    <li><p>The input will first be passed to an embedding layer and a
                        position embedding layer (we merge two layers into the
                        <code>PositionalEmbedding Layer</code>), to obtain <a
                            href="https://developers.google.com/machine-learning/glossary#embeddings">embeddings</a></p></li>
                    <li><p>Next, the input embedding of the source sequence will be passed
                        to the <code>TransformerEncoder</code>, which will produce a new
                        representation of it.</p></li>
                    <li><p>This new representation will then be passed to the
                        <code>TransformerDecoder</code>, together with the target sequence so

                        far (target words 0 to N). The <code>TransformerDecoder</code> will then
                        seek to predict the next words in the target sequence (N+1 and
                        beyond).</p></li>
                </ul>
                <p><Image src="/images/transformer.png" width={455} height={655} alt=""/></p>
                <p>Figure 2: The Transformer architecture as discussed in lecture, from <a href="https://arxiv.org/abs/1706.03762">&quot;Attention is all you need&quot;</a> (Vaswani et al., 2017).</p>
                <h3>Technical details</h3>
                <ul>
                    <li>The Transformer&apos;s encoder and decoder consist of N layers
                        (<code>num_layers</code>) each, containing <a
                            href="https://developers.google.com/machine-learning/glossary#multi-head-self-attention">multi-head
                            attention</a> (<code>tf.keras.layers.MultiHeadAttention</code>) layers
                        with M heads (<code>num_heads</code>), and point-wise feed-forward
                        networks.
                        <ul>
                            <li>The encoder leverages the self-attention mechanism.</li>
                            <li>The decoder (with N decoder layers) attends to the encoder&apos;s output
                                (with cross-attention to utilize the information from the encoder) and
                                its own input (with masked self-attention) to predict the next word. The
                                masked self-attention is causal—it is there to make sure the model can
                                only rely on the preceding tokens in the decoding stage.</li>
                        </ul></li>
                    <li>Multi-head attention: Each multi-head attention block gets three
                        inputs; Q (query), K (key), V (value). Instead of one single attention
                        head, Q, K, and V are split into multiple heads because it allows the
                        model to <a href="https://arxiv.org/abs/1706.03762">&quot;jointly attend to
                            information from different representation subspaces at different
                            positions&quot;</a>. You can read more about <a
                                href="https://storrs.io/attention/">single-head-attention</a> and <a
                                    href="https://storrs.io/multihead-attention/">multi-head-attention</a>.
                        The equation used to calculate the self-attention weights is as follows:
                        <span>{`$$\Large{Attention(Q, K, V) =
                            softmax_k\left(\frac{QK ^ T}{\sqrt{d_k}}\right) V} $$`}</span></li>
                </ul>
                <p><Image src="/images/transformer_multi-head-attention.png" width={438} height={447} alt=""/></p>
                <p>Figure 3: Multi-head attention from Google Research&apos;s <a href="https://arxiv.org/abs/1706.03762">&quot;Attention is all you need&quot;</a>(Vaswani et al., 2017).</p>
                <h3>Component 1: Position Encoding Layer</h3>
                <p>After text vectorization, both the input sentences (English) and
                    target sentences (Spanish) have to be converted to embedding vectors
                    using a <code>tf.keras.layers.Embedding</code> layer. This purpose of
                    the first embedding layer is to map word to a point in embedding space
                    where similar words are closer to each other.</p>
                <p>Next, a Transformer adds a <code>Positional Encoding</code> to the
                    embedding vectors. In this exercise, we would use a set of sines and
                    cosines at different frequencies (across the sequence). By definition,
                    nearby elements will have similar position encodings.</p>
                <p>We would use the following formula for calculating the positional
                    encoding:</p>
                <p><span>{`$$\Large{PE_{(pos, 2i)} = \sin(pos /
                    10000^{2i / \text{depth}})} $$</span> <span
                        class=\"math display\">$$\Large{PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i /
                        \text{depth}})} $$`}</span></p>
                <p>where <code>pos</code> takes value from 0 to <code>length - 1</code>
                    and <code>i</code> takes value from <code>0</code> to
                    <code>depth/2</code>.</p>
                <CodeBlock
                    code={
                        `def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    # compute angle_rads (angle in radians), the shape should be (pos, depth)
    angle_rads = positions / np.power(10_000, depths)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)`}
                    lang="python" />
                <p>You can verify and view your implementation of position_encoding below.</p>
                <CodeBlock
                    code={
                        `import matplotlib.pyplot as plt

pos_encoding = positional_encoding(length=2048, depth=512)

assert pos_encoding.shape == (2048, 512)

# Plot the dimensions.
plt.pcolormesh(pos_encoding.numpy().T, cmap=&apos;RdBu&apos;)
plt.ylabel(&apos;Depth&apos;)
plt.xlabel(&apos;Position&apos;)
plt.colorbar()
plt.show()`}
                    lang="python" />

                <p><Image src="/images/transformer-decoding_depth_vs_position.png" width={387} height={266} alt=""/></p>
                <p>Now, let&apos;s sequence the embedding and positional_encoding together to get a Positional Embedding Layer.</p>
                <CodeBlock
                    code={
                        `class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)

        self.pos_encoding = positional_encoding(length=sequence_length, depth=embed_dim)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x`
                    }
                    lang="python" />

                <h3>Component 2: Encoder Layer</h3>
                <CodeBlock
                    code={
                        `class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feed_forward = keras.Sequential(
            [layers.Dense(dense_dim, activation=\"relu\"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=\"int32\")
        else:
            padding_mask = None

        attention_output = self.attention(query = inputs, value = inputs, key = inputs, attention_mask = padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.feed_forward(proj_input)
        encoder_output = self.layernorm_2(proj_input + proj_output)

        return encoder_output

    def get_config(self):
        config = super().get_config()
        config.update({
            \"embed_dim\": self.embed_dim,
            \"dense_dim\": self.dense_dim,
            \"num_heads\": self.num_heads,
        })
        return config

    class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feed_forward = keras.Sequential(
            [layers.Dense(latent_dim, activation=\"relu\"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=\"int32\")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(query = inputs, value = inputs, key = inputs, attention_mask = causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query = out_1, value = encoder_outputs, key = encoder_outputs, attention_mask = padding_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.feed_forward(out_2)
        decoder_output = self.layernorm_3(out_2 + proj_output)

        return decoder_output

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype=\"int32\")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            \"embed_dim\": self.embed_dim,
            \"latent_dim\": self.latent_dim,
            \"num_heads\": self.num_heads,
        })
        return config`}
                    lang="python" />
                <h3>Component 3: Assemble the end-to-end model</h3>
                <p>Define the hyper-parameters</p>
                <CodeBlock
                    code={
                        `embed_dim = 64 #256
latent_dim = 512 #2048
num_heads = 4 #`}
                    lang="python" />
                <p>Assemble the layers</p>
                <CodeBlock
                    code={
                        `encoder_inputs = keras.Input(shape=(None,), dtype=\"int64\", name=\"encoder_inputs\")

x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)

encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype=\"int64\", name=\"decoder_inputs\")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name=\"decoder_state_inputs\")

x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = tf.keras.layers.Dropout(0.5)(x)
decoder_outputs = tf.keras.layers.Dense(units = vocab_size, activation = \"softmax\")(x)

# Note: the goal of this layer is to expand the dimension to match the vocabulary size in the target language, so choosing the layer output size accordingly
# Note: the output should be probabilities, so choose the activation function accordingly.
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name=\"transformer\"
)`}
                    lang="python" />
                <h2>Step 3. Training our model</h2>
                <p>We&apos;ll use accuracy as a quick way to monitor training progress on the validation data. Note that machine translation typically uses BLEU scores as well as other metrics, rather than accuracy.

                    Here we only train for a few epochs (to confirm everything), but to get the model to actually converge you should train for at least 30 epochs.</p>
                <pre><code>
                    Model: &quot;transformer&quot;<br />
                    __________________________________________________________________________________________________<br />
                    Layer (type)                   Output Shape         Param #     Connected to<br />
==================================================================================================<br/>
 encoder_inputs (InputLayer)    [(None, None)]       0           []<br/>
<br/>
 positional_embedding_4 (Positi  (None, None, 64)    960000      [&apos;encoder_inputs[0][0]&apos;]<br/>
 onalEmbedding)<br/>
<br/>
 decoder_inputs (InputLayer)    [(None, None)]       0           []<br/>
<br/>
<br/>
 transformer_encoder_2 (Transfo  (None, None, 64)    132736      [&apos;positional_embedding_4[0][0]&apos;]<br/>
 rmerEncoder)<br/>
<br/>
<br/>
 model_5 (Functional)           (None, None, 15000)  2134232     [&apos;decoder_inputs[0][0]&apos;,<br/>
                                                                  &apos;transformer_encoder_2[0][0]<br/>&apos;]
<br/>
==================================================================================================<br/>
Total params: 3,226,968<br/>
Trainable params: 3,226,968<br/>
Non-trainable params: 0<br/>
__________________________________________________________________________________________________<br/>
Epoch 1/30<br/>
1302/1302 [==============================] - 53s 38ms/step - loss: 1.8328 - accuracy: 0.3688 - val_loss: 1.4703 - val_accuracy: 0.4539<br/>
Epoch 2/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.5145 - accuracy: 0.4660 - val_loss: 1.3097 - val_accuracy: 0.5182<br/>
<br/>
Epoch 3/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.3744 - accuracy: 0.5148 - val_loss: 1.2292 - val_accuracy: 0.5565<br/>
Epoch 4/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.2884 - accuracy: 0.5498 - val_loss: 1.1951 - val_accuracy: 0.5806<br/>
Epoch 5/30<br/>
<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.2442 - accuracy: 0.5725 - val_loss: 1.1687 - val_accuracy: 0.5936<br/>
Epoch 6/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.2200 - accuracy: 0.5878 - val_loss: 1.1617 - val_accuracy: 0.5999<br/>
Epoch 7/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.2036 - accuracy: 0.5994 - val_loss: 1.1558 - val_accuracy: 0.6052<br/>
Epoch 8/30<br/>
1302/1302 [==============================] - 49s 37ms/step - loss: 1.1902 - accuracy: 0.6091 - val_loss: 1.1535 - val_accuracy: 0.6095<br/>
Epoch 9/30<br/>
1302/1302 [==============================] - 49s 38ms/step - loss: 1.1810 - accuracy: 0.6168 - val_loss: 1.1555 - val_accuracy: 0.6116<br/>
Epoch 10/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.1715 - accuracy: 0.6228 - val_loss: 1.1543 - val_accuracy: 0.6142<br/>
Epoch 11/30<br/>
<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.1624 - accuracy: 0.6288 - val_loss: 1.1536 - val_accuracy: 0.6150<br/>
Epoch 12/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.1530 - accuracy: 0.6336 - val_loss: 1.1583 - val_accuracy: 0.6144<br/>
Epoch 13/30<br/>
1302/1302 [==============================] - 49s 38ms/step - loss: 1.1453 - accuracy: 0.6377 - val_loss: 1.1535 - val_accuracy: 0.6178<br/>
Epoch 14/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.1371 - accuracy: 0.6416 - val_loss: 1.1585 - val_accuracy: 0.6178<br/>
Epoch 15/30<br/>
<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.1301 - accuracy: 0.6446 - val_loss: 1.1570 - val_accuracy: 0.6172<br/>
Epoch 16/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.1235 - accuracy: 0.6478 - val_loss: 1.1561 - val_accuracy: 0.6190<br/>
Epoch 17/30<br/>
1302/1302 [==============================] - 48s 37ms/step - loss: 1.1166 - accuracy: 0.6503 - val_loss: 1.1526 - val_accuracy: 0.6216<br/>
                </code></pre>
                <h2>Step 4. Inference: use the trained model to translate new sequences.</h2>
                <h3>Decoding test sentences</h3>
                <p>Finally, let&apos;s demonstrate how to translate brand new English
                sentences. We simply feed into the model the vectorized English sentence
                as well as the target token <code>[start]</code>, then we repeatedly
                generated the next token, until we hit the token
                <code>[end]</code>.</p>
                <CodeBlock
                    code={
                        `spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = \"[start]\"
    for i in range(max_decoded_sentence_length):

        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]

        probs = transformer([tokenized_input_sentence, tokenized_target_sentence], training = False)
        sampled_token_index = int(np.argmax(probs[:, i, :], 1))

        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += \" \" + sampled_token

        if sampled_token == \"[end]\":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]

for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)
    print(\"{}-->{}\".format(input_sentence, translated))`}
                    lang="python"/>
                <p>After 30 epochs, we get results such as:</p>
                <pre><code>
                    She handed him the money. [start] ella le pasó el dinero [end]<br/>
<br/>
                    Tom has never heard Mary sing. [start] tom nunca ha oído cantar a mary [end]<br/>
<br/>
                    Perhaps she will come tomorrow. [start] tal vez ella vendrá mañana [end]<br/>
<br/>
                    I love to write. [start] me encanta escribir [end]<br/>
<br/>
                    His French is improving little by little. [start] su francés va a [UNK] sólo un poco [end]<br/>
<br/>
                    My hotel told me to call you. [start] mi hotel me dijo que te [UNK] [end]<br/>
                </code></pre>
            </div>
        ),
    },
    {
        href: "https://www.alixleon.me/",
        title: "Statistical Insights into Diversified Portfolio Analysis: Unveiling Financial Trends and Strategies",
        description:
            "Financial analysis on big-tech companies stocks until 2021 using statistical financial methods.",
        thumbnail: "/images/stock_price_over_time.png",
        images: [
            "/images/stock_price_over_time.png",
            "/images/stock_price_over_time_2.png",
            "/images/pca_1.png",
            "/images/pca_2.png",
            "/images/pca_3.png",
        ],
        stack: ["R", "Portoflio Theory", "Hypothesis Testing", "Risk Management"],
        slug: "statistical-insights-diversified-portfolio-analysis",
        content: (
            <div>
                <p><strong>Introduction:</strong> The project centered on analyzing a diverse portfolio comprising stocks from the technology sector and various other industries. This approach aimed to evaluate the risk-return profile of each asset using advanced statistical techniques, providing valuable insights into their behavior over time.</p>

                <p><strong>Key Findings:</strong></p>
                <p>1. <strong>Non-Normal Distributions:</strong> One of the prominent discoveries was that the majority of assets in the portfolio did not adhere to normal distributions. Instead, they followed Skewed Standardized-t distributions, with some fitting a Generalized Error Distribution pattern.</p>
                <p>2. <strong>Performance Insights:</strong> Among the analyzed assets, Microsoft emerged as a standout performer with consistently high returns and a robust reward-to-risk ratio. This highlighted Microsoft&apos;s resilience and profitability within the portfolio context.</p>
                <p>3. <strong>Asset Independence:</strong> A significant observation was the near-zero covariance among many assets. This finding indicated a level of independence between assets, underscoring the importance of diversification in minimizing portfolio risk effectively.</p>
                <p>4. <strong>Copula Analysis:</strong> Employing t-Copula modeling, the study revealed an increased likelihood of joint extreme events among asset values. This aspect emphasized the necessity of robust risk management strategies to mitigate potential losses during market downturns.</p>
                <p>5. <strong>Portfolio Construction:</strong> Various portfolio strategies were explored, including the Minimum Variance Portfolio (MVP) and Efficient Portfolio, aiming to optimize returns while managing risk levels prudently. These strategies underscored the application of statistical methods in constructing balanced and efficient investment portfolios.</p>

                <p><strong>Methodology:</strong></p>
                <p>- <strong>Data Collection:</strong> Historical monthly prices and returns data spanning a five-year period were collected for stocks such as AMD, Microsoft, Apple Inc., Meta Platforms Inc., and others.</p>
                <p>- <strong>Statistical Techniques:</strong> The project employed a range of statistical tools including Q-Q plots, boxplots, and hypothesis testing to validate assumptions and derive meaningful insights from the data.</p>
                <p>- <strong>Risk Management:</strong> Techniques like Value at Risk (VaR) and Expected Shortfall (ES) were utilized to quantify and manage potential financial risks associated with the portfolio holdings.</p>

                <p><strong>Conclusion:</strong> The project exemplifies the critical role of statistical methods in financial analysis, providing actionable insights into asset performance, risk assessment, and portfolio optimization. These insights are invaluable for investors seeking to navigate the complexities of modern financial markets with confidence and precision.</p>

                <p><strong>Future Directions:</strong> Future research endeavors could expand on this analysis by incorporating a broader range of assets or integrating machine learning algorithms for predictive modeling. Such advancements could further enhance portfolio management strategies, offering deeper insights and improved decision-making capabilities.</p>

                <p><strong>Conclusion:</strong> In summary, the Statistical Methods In Finance final project offers a comprehensive exploration of portfolio dynamics and performance metrics, demonstrating the power of statistical rigor in illuminating financial trends and informing investment strategies. By leveraging these insights, investors can effectively manage risks and capitalize on opportunities in today&apos;s dynamic financial landscape.</p>
            </div>
        ),
    },
    {
        href: "https://alixleon.shinyapps.io/reu-project/",
        title: "Poor State, Rich State",
        description:
            "The Poor State, Rich State web app is an open-source tool built with R and Shiny that provides interactive visualizations and analyses of poverty data across different states.",
        thumbnail: "/images/poor_state-rich_state.jpg",
        images: [
            "/images/poor_state-rich_state.jpg",
        ],
        stack: ["R", "Shiny", "HTML", "CSS"],
        slug: "poorstaterichstate",
        content: (
            <div className="pt-4">
                <h2>What Does the Poor State, Rich State Web App Do?</h2>
                <p>
                    At its core, the <strong>Poor State, Rich State</strong> web app is a comprehensive tool for displaying and interpreting poverty statistics. Leveraging a robust dataset on poverty levels across various states, the app transforms this complex information into a format that is both easy to understand and visually appealing. Here are some of the key features and functionalities:
                </p>

                <ul>
                    <li><strong>Interactive Visualizations:</strong> The app features a variety of interactive charts and graphs, allowing users to explore poverty data dynamically. Users can filter data by state, time period, and other relevant criteria to gain deeper insights.</li>
                    <li><strong>User-Friendly Interface:</strong> The design prioritizes accessibility and ease of use, making it suitable for a wide range of users, including students, researchers, policymakers, and professionals in the social sciences.</li>
                    <li><strong>Data Comparison:</strong> Users can compare poverty statistics between states, track changes over time, and identify trends and patterns that may not be immediately apparent from raw data alone.</li>
                    <li><strong>Custom Reports:</strong> The app allows users to generate custom reports based on their selected criteria. These reports can be downloaded and used for academic research, policy making, or personal analysis.</li>
                    <li><strong>Educational Resource:</strong> As an educational tool, the app is a valuable resource for students learning about socio-economic issues. It provides a hands-on way to engage with real-world data and understand the impact of poverty across different regions.</li>
                    <li><strong>Open Source:</strong> Being an open-source project, the <strong>Poor State, Rich State</strong> web app encourages community contributions and collaboration. Developers and data scientists can contribute to the project, suggest improvements, and use the app as a foundation for their own projects.</li>
                </ul>

                <p>
                    The <strong>Poor State, Rich State</strong> web app exemplifies how data visualization and modern web technologies can be harnessed to make important socio-economic data accessible and actionable. Whether you are a student looking to enhance your understanding of poverty issues or a professional seeking reliable data for analysis, this app provides a powerful, user-friendly platform to meet your needs.
                </p>

                <p>
                    Explore the <strong>Poor State, Rich State</strong> web app today and discover how data can illuminate the challenges and disparities faced by states across the nation.
                </p>
            </div>
        ),
    },
];
