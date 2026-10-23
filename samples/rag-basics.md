# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) gives a language model access to documents it was not trained on. Instead of fine-tuning, relevant passages are fetched at query time and placed in the prompt.

## Ingestion

Documents are extracted to plain text, cleaned, and split into chunks small enough for the embedding model. Each chunk keeps metadata such as the source file, page number and section heading so answers can cite where they came from.

### Chunk size

Smaller chunks give precise matches but lose context. Larger chunks keep context but dilute the embedding. An overlap between neighbouring chunks prevents a sentence from being cut in half at a boundary.

## Embeddings

An embedding model maps text to a dense vector. all-MiniLM-L6-v2 produces 384-dimensional vectors and is small enough to run on a CPU. Vectors are L2-normalized so cosine similarity becomes a simple dot product.

## Retrieval

At query time the question is embedded with the same model and compared against every stored chunk. The top-k most similar chunks are returned and passed to the generator.
