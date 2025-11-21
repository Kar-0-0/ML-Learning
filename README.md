<!-- Deep Learning Study Repository -->
# Learning Machine Learning: From Basics to Deep Learning 📚

Welcome to my personal deep learning study repository! This space is dedicated to tracking my progress and sharing my learning journey in deep learning. The topics listed below are my end goals—areas I aim to explore and master over time. Not all topics are covered yet, but this repo will grow as I learn.

## Topics To Be Covered

- **Image Processing & Computer Vision**
  - Convolutional Neural Networks (CNNs)
  - Image classification, segmentation, and object detection
  - Data augmentation and preprocessing
- **Natural Language Processing (NLP)**
  - Language modeling (n-grams, RNNs, LSTMs, Transformers)
  - Text classification, sentiment analysis
  - Sequence-to-sequence models
- **Generative Models**
  - Variational Autoencoders (VAEs)
  - Generative Adversarial Networks (GANs)
- **Fundamentals**
  - Neural network basics
  - Activation functions, loss functions
  - Optimization techniques
- **Other Topics**
  - Transfer learning
  - Attention mechanisms
  - Reinforcement learning (basics)

## Repository Structure

```
├── data/           # Datasets and resources
│   └── names.txt   # Example data file
├── src/            # Source code and notebooks
│   └── bigram.ipynb# Example notebook
├── README.md       # This file
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dl_study.git
   ```
2. **Install dependencies**
   - Recommended: Use a Python virtual environment
   - Common packages: `numpy`, `torch`, `matplotlib`, `pandas`, `jupyter`
3. **Explore notebooks**
   - Open notebooks in Jupyter Lab or VS Code

## Contributing

This is a personal study space, but contributions, suggestions, and discussions are welcome! Feel free to open issues or submit pull requests. If you want to follow along or share advice, you're welcome to join my journey!

## License

This repository is licensed under the MIT License.

---

Happy learning! 🚀

---
# Output from Bigram.py

### Hyperparameters

epochs = 5000
n_layers = 6
context_size = 256
batch_size = 64
learning_rate = 3e-4
num_heads = 6
n_emb = 384
head_size = n_emb // num_heads
dropout = .2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

### Output (1,000 Characters)

WARWICK:
And, then, my gracious lord
And, all make recomfort of troops:
But as thou meane the king's body,
Proud the father that hinds--for that dance sleep,
Subjects to seel others, mount that is the dead;
For thou becausear'd of her home:
I, boy, her monumbering in barren, he must
Upon the short, much messen it shall
Scraign in kindle. Go, fie, look my help,
With I'rthly, which so far on thy uncle.

GLOUCESTER:
As I spain more dieted; call that I am.

CAMILLO:
God give, good my lord: say the tents to prove,
Or empty, threton as my point as mine:
I know in farewell to the notesty entrails,
Nor that country's pilot.

WARWIMONAMUS:
Agree; old fall our generaties are yourself good as
For what marks; I would I would say leave,
There far beauty to my sweet life for a country;
Then would give me in general be touch:
Might we will be mildine; take the look of a day.
I took your brother.

POMPEY:
For you know ever beloved yourself.
I have not at hangment, or else you may live
notine in me.
