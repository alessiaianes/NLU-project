# NLU Project: Technical Implementation Report

This project is organized into two main directories: **LM** (Language Modeling) and **NLU** (Natural Language Understanding). Each directory contains two sub-folders corresponding to the two tasks for that part, along with the comprehensive report summarizing the implementation details and key results.

---

## Part 1: Language Modeling (LM)

The objective of this part was to improve a baseline RNN model by incrementally incorporating advanced architectural and optimization techniques.

### Architectural Enhancements
* **LSTM Integration**: The initial vanilla RNN was replaced by a Long Short-Term Memory (LSTM) network to address the vanishing gradient problem and enhance the capacity to capture long-range dependencies.
* **Weight Tying**: This technique was implemented by sharing weights between the input embedding and output projection layers to reduce the number of parameters and improve regularization.

### Regularization and Optimization
* **Dropout Strategies**:
    * **Standard Dropout**: Two dropout layers were added: one positioned after the embedding layer and another prior to the final linear output layer.
    * **Variational Dropout**: This was applied by using the same dropout mask for recurrent connections across all time steps for a given sequence. 
* **Optimizers**:
    * **AdamW**: The standard SGD optimizer was replaced with AdamW to improve convergence dynamics. 
    * **NT-AvSGD**: The optimization process was further refined using Non-monotonically Triggered Averaged Stochastic Gradient Descent, where the averaging trigger is determined by a non-monotonic condition instead of a user-defined schedule. 

---

## Part 2: Natural Language Understanding (NLU)

This section focuses on joint **Intent Classification** and **Slot Filling** tasks using the ATIS dataset. 

### Enhanced LSTM 
* **Bidirectionality**: The baseline LSTM architecture was modified by introducing bidirectional layers, enabling the model to capture both past and future context for each token in a sequence. 
* **Multi-Layered Dropout**: Dropout layers were introduced after the embedding layer, the LSTM output, and the final hidden states to mitigate overfitting and improve generalization. 

### BERT Multi-task Learning 
* **Architecture**: A pre-trained BERT model was fine-tuned in a multi-task learning setup.
    * **Intent Classification**: Leverages the pooled output of the BERT architecture for sentence-level classification. 
    * **Slot Filling**: Utilizes token-level output for slot tagging.
* **Sub-tokenization Handling**: To address challenges associated with sub-tokenization, slot labels were aligned with BERT's tokenized outputs during the fine-tuning process. 
* **Model Comparison**: The implementation explored both **BERT-base-uncased** and **BERT-large-uncased**. While BERT-large offers a higher capacity to model complex relationships, BERT-base was prioritized for its lower resource requirements and faster training. 

---
