# Grounded Document Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents. It retrieves relevant information using hybrid search and generates answers grounded in the uploaded documents, with source and page citations.

The goal is to reduce unsupported answers and make document-based question answering more reliable and transparent.

## Features

- **Multiple PDF Upload:** Upload and process multiple PDF documents.
- **Text Extraction:** Extract text from PDFs while preserving document names and page numbers.
- **Text Cleaning:** Clean extracted text to improve downstream processing.
- **Token-Based Chunking:** Split documents into manageable chunks with overlapping context.
- **Semantic Search:** Generate text embeddings using Sentence Transformers and retrieve semantically relevant chunks with FAISS.
- **Keyword Search:** Use BM25 to retrieve documents based on keyword relevance.
- **Hybrid Retrieval:** Combine semantic and keyword search results using Reciprocal Rank Fusion (RRF).
- **Grounded Answer Generation:** Use a Groq-hosted LLM to generate answers based on retrieved document content.
- **Source Citations:** Include document names and page numbers in generated answers.
- **Evidence-Based Refusal:** Respond when the uploaded documents do not contain sufficient information to answer a question.
- **Interactive Interface:** Use the application through a Streamlit interface.

## System Architecture

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

```text
          PDF Documents
                |
                v
         Text Extraction
                |
                v
          Text Cleaning
                |
                v
        Token-Based Chunking
                |
                v
       Generate Embeddings
                |
                v
       +------------------+
       |  FAISS Search    |
       |  BM25 Search     |
       +------------------+
                |
                v
     Reciprocal Rank Fusion
                |
                v
       Retrieve Relevant
             Chunks
                |
                v
        LLM Answer Generation
                |
                v
       Grounded Answer with
        Source Citations
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| PDF Text Extraction | pypdf |
| Text Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| LLM Inference | Groq |
| Environment Variables | python-dotenv |

## How It Works

### 1. Document Processing

Uploaded PDFs are processed to extract their text. The extracted content is cleaned and divided into smaller, overlapping chunks.

Each chunk retains metadata such as:
- Chunk ID
- Source document
- Page number

This metadata helps the application identify where retrieved information originated.

### 2. Embedding Generation and Indexing

The application uses the `sentence-transformers/all-MiniLM-L6-v2` model to convert text chunks into numerical embeddings.

These embeddings are normalized and stored in a FAISS index using inner-product similarity search.

### 3. Hybrid Retrieval

The application uses two retrieval methods:

- **FAISS:** Finds semantically similar text chunks.
- **BM25:** Finds text chunks based on keyword relevance.

The results from both retrieval methods are combined using Reciprocal Rank Fusion (RRF).

This helps retrieve relevant information even when the wording of a user's question differs from the wording in the document.

### 4. Grounded Answer Generation

The retrieved chunks are passed to the LLM through the Groq API.

The model is instructed to:
- Answer using the retrieved document content.
- Avoid introducing unsupported information.
- Include source document names and page numbers.
- Indicate when the available evidence is insufficient.

## Project Structure

```text
Grounded-Document-Assistant/
│
├── app.py
├── rag_pipeline.py
├── Grounded_Document_Assistant.ipynb
├── evaluation_dataset.py
├── evaluation_retrieval.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── screenshots/
│   ├── 01_home_page.png
│   ├── 02_document_processing.png
│   ├── 03_direct_question_answer.png
│   ├── 04_indirect_question_answer.png
│   ├── 05_no_evidence_response.png
│   └── 06_multiple_pdf_documents.png
│
└── data/
    └── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sanjuus/Grounded_Document_Assistant.git
cd Grounded_Document_Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key.

**Important:** Never commit your `.env` file or expose your API key in the repository.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

### 6. Upload Documents and Ask Questions

1. Upload one or more PDF documents.
2. Process the uploaded documents.
3. Enter a question related to the document contents.
4. Review the generated answer and its source citations.

## Evaluation

A retrieval evaluation dataset containing 10 questions was created to assess whether the retrieval pipeline could locate the expected source document and relevant page.

The evaluation used Recall@K.

| Metric | Result |
|---|---:|
| Recall@5 | 100% |
| Recall@10 | 100% |
| Evaluation Questions | 10 |

**Interpretation:** The expected source page was retrieved within the top 5 results for all 10 evaluation questions. It was also retrieved within the top 10 results for all 10 questions.

These results are an initial retrieval evaluation, not proof of perfect answer accuracy or production-level reliability. The evaluation is limited to the questions and document used in this test set.

## Limitations and Future Improvements

Potential improvements include:

- Expand the evaluation dataset with more documents and diverse questions.
- Evaluate answer correctness and citation accuracy, not only retrieval recall.
- Add automated checks for unsupported answers and hallucinations.
- Experiment with reranking models to improve retrieval relevance.
- Add document-level filtering and more advanced metadata search.
- Improve handling of scanned PDFs and image-based documents.
- Add conversation history for follow-up questions.

## Learning Outcomes

Through this project, I explored and implemented:

- Generative AI and Large Language Models.
- Retrieval-Augmented Generation (RAG).
- PDF text extraction and preprocessing.
- Token-based text chunking and overlap.
- Semantic search and text embeddings.
- Vector databases and FAISS similarity search.
- BM25 keyword retrieval.
- Hybrid search and Reciprocal Rank Fusion.
- Prompt design for evidence-grounded generation.
- Source attribution and refusal handling.
- Building an interactive application with Streamlit.
- Evaluating information retrieval using Recall@K.

## Acknowledgements

This project was developed as a hands-on learning and portfolio project to understand how document-grounded AI applications are designed and implemented.

