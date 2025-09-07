# Learning Transformer GPT from Karpathy

This repository contains implementations of language models following Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out" tutorial series. The project demonstrates the evolution from a simple bigram model to a full transformer-based GPT model.

## 📁 Repository Structure

- `bigram.py` - Simple bigram language model implementation
- `gpt.py` - Full GPT transformer implementation with multi-head attention
- `input.txt` - Tiny Shakespeare dataset for training
- `ravindra_gpt_learning_from_karpathy.ipynb` - Jupyter notebook with step-by-step learning process

## 🚀 Features

### Bigram Model (`bigram.py`)
- **Simple Language Model**: Basic bigram model that predicts the next character
- **Character-level Tokenization**: Uses character-level vocabulary (65 unique characters)
- **Training Loop**: Complete training implementation with AdamW optimizer
- **Text Generation**: Generates new text by sampling from the learned distribution

**Key Parameters:**
- Batch size: 32
- Block size: 8 (context length)
- Learning rate: 1e-2
- Max iterations: 3000

### GPT Model (`gpt.py`)
- **Multi-Head Self-Attention**: Implements scaled dot-product attention
- **Transformer Architecture**: Complete transformer blocks with residual connections
- **Position Embeddings**: Learns positional information for tokens
- **Layer Normalization**: Pre-norm architecture for training stability
- **Dropout**: Regularization to prevent overfitting

**Key Parameters:**
- Batch size: 64
- Block size: 256 (context length)
- Embedding dimension: 384
- Number of heads: 6
- Number of layers: 6
- Dropout: 0.2
- Learning rate: 3e-4

## 🛠️ Technical Implementation

### Self-Attention Mechanism
The implementation includes a detailed explanation of the self-attention mechanism:

1. **Query, Key, Value Matrices**: Linear transformations of input embeddings
2. **Attention Scores**: Computed as `Q @ K^T / sqrt(head_size)`
3. **Causal Masking**: Lower triangular mask to prevent future token attention
4. **Softmax Normalization**: Converts scores to probabilities
5. **Weighted Aggregation**: Combines values based on attention weights

### Architecture Components

#### Head Class
- Single attention head implementation
- Scaled dot-product attention
- Causal masking for autoregressive generation

#### MultiHeadAttention Class
- Parallel processing of multiple attention heads
- Concatenation and projection of head outputs
- Dropout for regularization

#### FeedForward Class
- Two-layer MLP with ReLU activation
- Expansion factor of 4x (following original transformer paper)
- Dropout for regularization

#### Block Class
- Complete transformer block
- Self-attention + feed-forward with residual connections
- Layer normalization (pre-norm architecture)

## 📊 Dataset

The models are trained on the **Tiny Shakespeare** dataset:
- **Size**: ~1.1M characters
- **Content**: Complete works of Shakespeare
- **Split**: 90% training, 10% validation
- **Vocabulary**: 65 unique characters

## 🎯 Usage

### Running the Bigram Model
```bash
python bigram.py
```

### Running the GPT Model
```bash
python gpt.py
```

### Using the Jupyter Notebook
```bash
jupyter notebook "ravindra_gpt_learning_from_karpathy.ipynb"
```

## 📈 Training Progress

The notebook shows the learning progression:
1. **Data Preprocessing**: Character-level tokenization and batch creation
2. **Bigram Model**: Simple baseline with embedding lookup
3. **Self-Attention**: Mathematical foundation and implementation
4. **Multi-Head Attention**: Parallel attention heads
5. **Full Transformer**: Complete GPT architecture

## 🔧 Requirements

- Python 3.x
- PyTorch
- Jupyter Notebook (for interactive learning)

## 📚 Educational Value

This repository serves as an excellent learning resource for:
- Understanding transformer architecture from first principles
- Implementing self-attention mechanisms
- Building language models from scratch
- Learning PyTorch best practices
- Understanding the mathematical foundations of modern LLMs

## 🎓 Learning Path

The implementation follows a progressive learning approach:
1. Start with simple bigram model
2. Understand attention mechanisms through toy examples
3. Implement single-head attention
4. Scale to multi-head attention
5. Build complete transformer blocks
6. Assemble full GPT model

## 📝 Notes

- The code includes extensive comments explaining each component
- Mathematical operations are broken down step-by-step
- The implementation follows the original "Attention is All You Need" paper
- Pre-norm architecture is used for better training stability
- Character-level tokenization is used for simplicity (vs. subword tokenization in production models)

## 🤝 Contributing

This is a learning repository. Feel free to:
- Add more detailed explanations
- Implement additional features
- Experiment with different architectures
- Share your learning insights

## 📖 References

- [Let's build GPT: from scratch, in code, spelled out](https://karpathy.ai/zero-to-hero.html) - Andrej Karpathy
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
