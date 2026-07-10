# RAG-against-the-machine

Build a Retrieval-Augmented Generation system that answers questions about codebases by retrieving relevant information and generating evidence-based responses, implementing intelligent chunking, efficient retrieval (TF-IDF/BM25)

# Resources

- [General explaination](https://www.mrlatte.net/en/research/2026/04/27/rag-complete-guide/)
- [RAG from scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=647s)
- [Data chunking strategies](https://developer.ibm.com/articles/awb-enhancing-rag-performance-chunking-strategies/)
- [Grounded generation](https://zeroentropy.dev/concepts/grounded-generation/)

**Model Overview**
*Qwen3-0.6B has the following features:*

    Type: Causal Language Models
    Training Stage: Pretraining & Post-training
    Number of Parameters: 0.6B
    Number of Paramaters (Non-Embedding): 0.44B
    Number of Layers: 28
    Number of Attention Heads (GQA): 16 for Q and 8 for KV
    Context Length: 32,768
