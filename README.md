# SPRINT 3: LLM UPSKILLING - Transformer and BERT

# A summary of Attention is all you NEED!

- **Transformer Architecture**: Introduced a novel neural network architecture called the Transformer.
- **Self-Attention Mechanism**: Utilized self-attention mechanisms to capture relationships between input tokens.
- **Positional Encoding**: Incorporated positional information using sinusoidal functions in input embeddings.
- **Multi-Head Attention**: Employed multiple parallel self-attention heads for capturing diverse information.
- **Encoder-Decoder**: Demonstrated the Transformer's applicability in both encoder and decoder tasks, such as machine translation.
- **Scaled Dot-Product Attention**: Efficiently computed token relationships using scaled dot-product attention.
- **Position-wise Feed-Forward Networks**: Included position-wise feed-forward networks for added non-linearity.
- **Layer Normalization and Residual Connections**: Used for stabilization during training.
- **Training Objectives**: Discussed training objectives and introduced label smoothing.
- **State-of-the-Art Performance**: Achieved superior results in machine translation and other sequence-to-sequence tasks, with fewer parameters

# What is a transformer?

The Transformer is a neural network architecture introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. It has had a profound impact on the field of natural language processing (NLP) and has been widely adopted for various sequence-to-sequence tasks beyond NLP as well.

![the_transformer_3.png](SPRINT%203%20LLM%20UPSKILLING%20-%20Transformer%20and%20BERT%205306f0ead7ef46359ccf172f088c9149/the_transformer_3.png)

Figure: A high-level look at Transformer

![attention_research_1-727x1024.png](SPRINT%203%20LLM%20UPSKILLING%20-%20Transformer%20and%20BERT%205306f0ead7ef46359ccf172f088c9149/attention_research_1-727x1024.png)

Figure: Transformer “ Attention is all you need!”

# Transformer Architecture

The Transformer architecture is a neural network model introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. It has had a transformative impact on natural language processing and has been widely adopted in various sequence-to-sequence tasks. The core idea behind the Transformer is to use self-attention mechanisms to capture relationships between input tokens, enabling it to process sequential data efficiently. Here's an in-depth explanation of the key components and concepts of the Transformer architecture.

### 1. Input embeddings:

- The input to the Transformer consists of a sequence of tokens, such as words or subword units (e.g., subword pieces in BERT).
- Each token is initially embedded into a continuous vector representation, usually with an embedding layer.

### 2. Positional encoding:

- Since the Transformer does not inherently understand the order of tokens in a sequence, positional information is added to the token embeddings.
- Positional encoding is typically achieved using sinusoidal functions, allowing the model to differentiate tokens based on their positions in the sequence.

### 3. **Encoder and Decoder Stacks**:

- The Transformer architecture consists of stacks of encoders and decoders.
- The encoder stack processes the input sequence, while the decoder stack generates the output sequence in sequence-to-sequence tasks.

### 4. **Self-Attention Mechanism**:

- The self-attention mechanism is the key innovation of the Transformer. It allows the model to weigh the importance of different input tokens when generating an output token.
- Self-attention computes a weighted sum of all input tokens' representations, with the weights determined by their relevance to the current token being generated.
- Multi-head attention is used to capture different types of relationships in parallel, providing the model with more expressive power.

### 5. **Scaled Dot-Product Attention**:

- In self-attention, the similarity between two tokens' embeddings is computed as the scaled dot product of their vectors.
- The scaling factor prevents gradients from vanishing or exploding during training.

### 6. **Position-wise Feed-Forward Networks**:

- After the self-attention mechanism, a position-wise feed-forward network is applied to each token's representation.
- This network introduces non-linearity and helps the model capture complex functions.

### 7. **Residual Connections and Layer Normalization**:

- Residual connections, along with layer normalization, are used to stabilize training and enable the efficient flow of gradients through the network.

### 8. **Encoder-Decoder Attention**:

- In sequence-to-sequence tasks, the decoder stack includes an additional attention mechanism called encoder-decoder attention.
- It allows the decoder to focus on relevant parts of the input sequence during the generation of each output token.

### 9. Masking:

- In autoregressive tasks like language generation, a masking mechanism is applied to prevent tokens from attending to future tokens in the sequence.

### 10. Output layers:

- The final layer(s) in the decoder stack generates the output sequence, often with a soft-max activation for probability distribution over the vocabulary.

# Google BERT

BERT, which stands for "Bidirectional Encoder Representations from Transformers," is a natural language processing (NLP) model developed by Google in 2018. It represents a significant breakthrough in the field of NLP and has been widely adopted for various language understanding tasks.

## Features of Google BERT

1. **Transformer Architecture**: BERT is built upon the Transformer architecture, which is a neural network architecture designed for sequence-to-sequence tasks. Transformers have become the foundation for many state-of-the-art NLP models.
2. **Bidirectional Context**: Unlike earlier NLP models that processed text in a unidirectional manner (either left-to-right or right-to-left), BERT is bidirectional. It takes into account the context from both directions when understanding a word or token in a sentence. This helps it capture richer contextual information.
3. **Pretraining and Fine-Tuning**: BERT is pre-trained on a massive corpus of text, learning to predict missing words in a sentence. This pre-trained model can then be fine-tuned on specific NLP tasks like text classification, named entity recognition, question answering, and more. Fine-tuning is faster and requires less labeled data compared to training a model from scratch.
4. **Large-Scale Training**: BERT models are typically pre-trained on very large datasets containing billions of words. This extensive pretraining allows them to learn a broad range of language patterns and semantics.
5. **State-of-the-Art Results**: BERT achieved state-of-the-art results on a wide range of NLP tasks, including sentiment analysis, language translation, text summarization, and more. It has become a foundational model in NLP research and applications.
6. **Variants and Improvements**: Since the release of BERT, there have been various improvements and variants, such as RoBERTa, ALBERT, and more, which fine-tune the architecture and training techniques to achieve even better performance on specific tasks.

BERT and its variants have had a profound impact on NLP, leading to significant advancements in natural language understanding and applications like chatbots, language translation, search engines, and sentiment analysis. These models have also been instrumental in the development of conversational AI systems, like chatbots and virtual assistants, by providing them with a better understanding of context and semantics in human language.

## BERT works in two steps:

BERT, like many other transformer-based models used in natural language processing (NLP), operates in two main steps: pretraining and fine-tuning. Let's break down these two steps:

1. **Pretraining**:
    - **Architecture**: BERT is built upon a deep neural network architecture known as the Transformer.
    - **Massive Corpus**: BERT is pre-trained on a massive corpus of text, often containing billions of words, to learn the fundamental properties of language. The model is trained to predict the probability of a word or token in a sentence based on the surrounding words, taking into account both the words to the left and right of it. This bidirectional context modeling is a key feature of BERT.
    - **Masked Language Modeling (MLM)**: During pretraining, BERT learns by predicting masked words in sentences. Some words in the input text are randomly replaced with a special [MASK] token, and the model's objective is to predict what the masked words are based on the surrounding context. This process helps BERT capture a deep understanding of word meanings and context.
    - **Segment Embeddings**: BERT also incorporates segment embeddings to distinguish between different sentences or segments of text within a document. This is particularly useful for tasks that involve multiple sentences, such as question-answering.
    - **Positional Encodings**: Transformers, including BERT, do not have built-in knowledge of word order. Positional encodings are added to the input embeddings to provide information about the position of each word in the sequence.
    - **Layer Stacking**: BERT consists of multiple layers of transformers stacked on top of each other. Each layer refines the representation of the input text.
2. **Fine-Tuning**:
    - After the pretraining phase, BERT is a language model with a deep architecture and a strong understanding of language. However, it doesn't know specifics about any particular NLP task.
    - To make BERT useful for specific tasks like text classification, question answering, or sentiment analysis, it undergoes a fine-tuning process.
    - During fine-tuning, BERT's pre-trained weights are further trained on a smaller dataset related to the specific task at hand. This dataset contains labeled examples for the task.
    - The fine-tuning process adjusts the model's weights to make it perform well on the target task. The model learns task-specific patterns and features from the fine-tuning data.
    - Fine-tuning typically involves adding a task-specific output layer to the pre-trained BERT model, which transforms BERT's contextual embeddings into task-specific predictions. For example, in text classification, the output layer might consist of a softmax layer for classifying text into different categories.

# Model Architecture

The original BERT paper, titled "BERT: Bidirectional Encoder Representations from Transformers," introduced two main model sizes: BERT-base and BERT-large. These two sizes differ in terms of the number of layers, hidden units, and the total number of parameters. Here's an overview of both:

1. **BERT-base**:
    - **Number of Layers**: BERT-base consists of 12 layers of Transformer encoder blocks.
    - **Hidden Units**: Each layer has 768 hidden units (also referred to as the model's "dimension").
    - **Total Parameters**: BERT-base has approximately 110 million parameters.
2. **BERT-large**:
    - **Number of Layers**: BERT-large is a larger variant with 24 layers of Transformer encoder blocks.
    - **Hidden Units**: Each layer has 1,024 hidden units.
    - **Total Parameters**: BERT-large has approximately 340 million parameters.

Both BERT-base and BERT-large models are pre-trained on large text corpora and can be fine-tuned for specific downstream NLP tasks, such as text classification, named entity recognition and question answering. The larger BERT-large model generally provides improved performance over the BERT-base due to its increased model capacity, but it also requires more computational resources for training and inference.

# ****BERT: Input/Output Representation****

BERT (Bidirectional Encoder Representations from Transformers) operates on tokenized input text and produces contextualized embeddings as output. Here's a detailed explanation of the input and output representations in BERT:

**Input Representation**:

1. **Tokenization**: The input text is tokenized into subword or word-level tokens. BERT uses WordPiece tokenization, which breaks words into smaller units (subword tokens) when necessary. For example, "unhappiness" might be tokenized into ["un", "##happy", "##ness"].
2. **Special Tokens**:
    - [CLS] Token: At the beginning of each input sequence, a special [CLS] token is added. This token is used for various downstream tasks and carries information about the entire input sequence.
    - [SEP] Token: To separate different segments of text within a document (e.g., sentences or paragraphs), a [SEP] token is inserted between them. It helps BERT understand the structure of the text.
3. **Segment Embeddings**: BERT is designed to handle tasks involving multiple segments of text. To differentiate between different segments, such as sentences or paragraphs, segment embeddings are added to each token. For example, tokens in the first sentence might be assigned segment embedding 0, while tokens in the second sentence might be assigned segment embedding 1.
4. **Positional Embeddings**: Since the original Transformer architecture doesn't inherently understand the position of tokens in a sequence, positional embeddings are added to the input embeddings. These embeddings encode the position of each token within the sequence, allowing BERT to understand word order.
5. **Padding**: Input sequences are often padded to have a consistent length. Tokens are added to the end of shorter sequences to match the length of the longest sequence in a batch. Padding tokens do not contribute to the model's understanding of the text.

**Output Representation**:

1. **Contextualized Word Embeddings**:
    - BERT produces contextualized word embeddings for each token in the input sequence. These embeddings capture the meaning and context of each token based on the entire input text.
    - Contextualization is achieved through the multiple layers of Transformer encoder blocks in the model.
    - The final layer's hidden states represent the contextualized embeddings for each token.
2. **[CLS] Token Output**:
    - The hidden state corresponding to the [CLS] token at the beginning of the input sequence is used as a representation of the entire input sequence.
    - This [CLS] token representation is used for various downstream tasks, such as text classification. It carries information about the entire input.
3. **Additional Layers for Specific Tasks**:
    - For downstream NLP tasks like text classification, named entity recognition, or question answering, additional layers or output heads are added on top of BERT's contextualized embeddings.
    - These task-specific layers transform BERT's representations into task-specific predictions. For example, for text classification, a softmax layer may be added to predict class probabilities.

# ****BERT: Pretraining****

The pretraining phase of BERT (Bidirectional Encoder Representations from Transformers) is a crucial step in its architecture. During pretraining, BERT learns to understand and represent the structure and semantics of language using a massive corpus of text data. Here's how the pretraining phase of BERT works:

1. **Large Corpus of Text Data**: BERT is pre-trained on a vast amount of text data. The choice of this corpus is essential because it determines the breadth and quality of language knowledge that BERT acquires. Commonly used corpora include Wikipedia, BookCorpus, and other web text sources.
2. **Wordpiece Tokenization**: The input text data is tokenized into smaller units called wordpieces using a technique called WordPiece tokenization. This method allows BERT to handle a wide range of subword units and is particularly useful for languages with complex morphology or for breaking down long words into manageable pieces. For example, "unhappiness" might be tokenized into ["un", "##happy", "##ness"].
3. **Masked Language Modeling (MLM)**: BERT learns by predicting masked words in sentences. During pretraining, some of the tokens in the input sequence are randomly replaced with a special [MASK] token. The model's objective is to predict what these masked tokens are based on the surrounding context. This task is referred to as the "Masked Language Modeling" task.
    - For example, in the sentence: "The cat sat on the [MASK] and played with a ball," BERT might be tasked with predicting that the masked word is "mat."
    - Predicting masked tokens encourages BERT to learn the relationships between words and their contexts effectively.
4. **Bidirectional Context**: One of the key innovations of BERT is its bidirectional context modeling. Unlike earlier models that processed text in a unidirectional manner (either left-to-right or right-to-left), BERT takes into account the context from both directions when understanding a word or token in a sentence. This allows it to capture richer contextual information.
5. **Next Sentence Prediction (NSP)**: In addition to MLM, BERT is also pre-trained on a task called "Next Sentence Prediction." In this task, BERT learns to predict whether one sentence follows another in a given text pair. This helps BERT understand the relationships between sentences and paragraphs.
6. **Stacked Transformer Encoders**: BERT architecture consists of multiple layers of Transformer encoder blocks stacked on top of each other. Each encoder block refines the representation of the input text. The depth of the network (i.e., the number of layers) contributes to BERT's ability to capture complex language patterns.
7. **Training**: BERT is trained using a variant of the Transformer architecture called the "Transformer Encoder." It learns by minimizing the loss associated with the MLM and NSP tasks on the large corpus of text data. Training BERT typically requires powerful hardware and substantial computational resources.
8. **Knowledge Transfer**: After pretraining, BERT has acquired a deep understanding of language, including word meanings, syntax, and semantics, from the vast amount of text data. This knowledge can then be fine-tuned on specific downstream NLP tasks, which often require less labeled data compared to training a model from scratch.

# ****BERT: Fine-tuning****

Fine-tuning is the second phase of using BERT (Bidirectional Encoder Representations from Transformers) after the pretraining phase. During fine-tuning, the pre-trained BERT model is adapted or "fine-tuned" for specific downstream natural language processing (NLP) tasks. Here's how the fine-tuning process works:

1. **Task-Specific Data**:
    - For fine-tuning, you need a labeled dataset specific to your target NLP task. This dataset contains examples with input text and corresponding labels or annotations for the task.
    - Common NLP tasks for which BERT can be fine-tuned include text classification, named entity recognition, question answering, sentiment analysis, and more.
2. **Architecture Adaptation**:
    - The pre-trained BERT model, which has already learned a deep understanding of language during pretraining, serves as a powerful feature extractor.
    - Task-specific layers are added on top of the BERT model to adapt it for the specific task. These layers are often shallow compared to the original BERT model.
    - The architecture of these task-specific layers depends on the nature of the task. For example:
        - For text classification, a softmax layer is added for predicting class labels.
        - For named entity recognition, a conditional random field (CRF) layer can be added for sequence labeling.
        - For question answering, additional layers are designed to predict the answer span within the context.
3. **Loss Function and Training**:
    - The fine-tuning process involves training the adapted BERT model on the task-specific dataset.
    - A task-specific loss function is used to measure the difference between the model's predictions and the ground truth labels or annotations.
    - The model's weights are updated using backpropagation and gradient descent to minimize the loss.
4. **Hyperparameter Tuning**:
    - Fine-tuning may require adjusting hyperparameters such as learning rate, batch size, and the number of training epochs to achieve optimal performance on the specific task.
5. **Transfer Learning**:
    - Fine-tuning leverages the knowledge and language understanding acquired by the pre-trained BERT model during pretraining.
    - The model's lower layers, which contain general language understanding, remain relatively fixed, while the task-specific layers are trained to adapt to the new task.
    - This transfer of knowledge from pretraining to fine-tuning significantly reduces the need for large amounts of labeled data for each specific task.
6. **Evaluation and Deployment**:
    - After fine-tuning, the adapted BERT model is evaluated on a separate validation or test dataset to assess its performance.
    - Once satisfactory performance is achieved, the model can be deployed for inference on new, unlabeled data to make predictions or perform other NLP tasks.
7. **Iterative Process**:
    - Fine-tuning can be an iterative process, where hyperparameters are tuned and the model is refined until the desired performance is achieved.
    - The same pre-trained BERT model can be fine-tuned for multiple different tasks, making it a versatile tool for various NLP applications.

# ****BERT: Feature Extraction****

BERT (Bidirectional Encoder Representations from Transformers) can also be used for feature extraction, which is the process of extracting contextualized word embeddings or features from pretrained BERT models without fine-tuning them for specific tasks. Feature extraction with BERT is useful when you want to leverage the powerful language understanding capabilities of BERT to obtain contextualized representations for text data in order to use those features as input for other machine learning models. Here's how feature extraction with BERT typically works:

1. **Load Pretrained BERT Model**: Begin by loading a pretrained BERT model of your choice. You can choose from various BERT model sizes (e.g., "bert-base-uncased," "bert-large-uncased") based on your specific requirements and computational resources.
2. **Tokenization**: Tokenize the text data you want to extract features from using the same tokenization method and tokenizer that was used for pretraining BERT (usually WordPiece tokenization).
3. **Contextualized Embeddings**: Pass the tokenized input through the BERT model to obtain contextualized word embeddings for each token in the input text. These embeddings capture the semantics and context of the words in the text, taking into account bidirectional context.
4. **Feature Extraction**: Extract the contextualized embeddings or features from the BERT model for downstream use. You can extract embeddings from various layers of the BERT model, depending on the level of abstraction you require for your task. Common choices include:
    - **Token-Level Features**: You can extract embeddings for individual tokens in the input text. This is useful for tasks like text classification and sentiment analysis, where you might average or pool token-level embeddings to obtain a fixed-length feature vector for each text.
    - **Sentence-Level Features**: To obtain features at the sentence or document level, you can use the embedding of the [CLS] token (the first token in the input sequence), which is often used as a summary representation of the entire input.
    - **Pooling**: You can apply pooling operations, such as mean pooling or max pooling, over token-level embeddings to obtain aggregated features.
5. **Downstream Tasks**: Use the extracted features as input for downstream machine learning tasks. These tasks can include text classification, named entity recognition, sentiment analysis, and more. You can use traditional machine learning models or neural networks to build classifiers or predictors based on the contextualized features obtained from BERT.

### **Benefits of BERT Feature Extraction:**

- **Transfer Learning**: By using BERT for feature extraction, you benefit from the transfer of knowledge and language understanding that BERT acquired during its pretraining on large text corpora.
- **Improved Representations**: BERT provides rich and contextualized representations of text, which can improve the performance of downstream models, especially when dealing with complex and nuanced language understanding tasks.
- **Reduced Data Requirements**: Feature extraction allows you to use BERT for tasks with limited labeled data, as you don't need to fine-tune the entire model.
- **Versatility**: Extracted features can be used as input for a wide range of machine learning and NLP tasks, making BERT a versatile tool for text data analysis.

# References

1. [Attention is all you need!](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634) 
2. [The illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
3. [BERT Paper](https://www.notion.so/SPRINT-3-LLM-UPSKILLING-Transformer-and-BERT-5306f0ead7ef46359ccf172f088c9149?pvs=21)
4. [GOOGLE BERT](https://medium.com/@thapaliyanish123/google-bert-8e990b64f570)