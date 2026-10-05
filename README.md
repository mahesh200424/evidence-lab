# Evidence Lab — local document research assistant

A local RAG portfolio project with page-preserving PDF search, evidence-linked answers, and transparent evaluation. Built with Streamlit, SQLite, NumPy, pypdf, and Ollama.

## Install from GitHub

```bash
git clone https://github.com/mahesh200424/evidence-lab.git
cd evidence-lab
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull phi4-mini
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Ollama must be installed and running separately. The portable Streamlit command works without the macOS launcher. Tested with Python 3.11 on Apple Silicon. No paid API credentials are needed.

Run `zsh /Users/mahesh/local-research-assistant/start.sh`, then open http://127.0.0.1:8501. Start Ollama and keep the model SSD connected. Stop the web server with Ctrl+C. Models are unloaded after each operation.

## Try it
1. In Library, load synthetic demo documents, or upload text-based PDFs and click Index.
2. In Research, ask “Does checking citation IDs prove factual correctness?” Inspect retrieval first, then generate an answer.
3. Expand cited sources to inspect exact text and page position. Uploaded source PDFs can be downloaded. Source expanders link answer citations to passages; there is no embedded PDF viewer yet.
4. In Evaluation, run the demo questions. Review answers and citations, save human reviews, and export JSON.
5. Switch the answer model and rerun the same questions. Keep corpus and top-k identical for a model comparison.

## Architecture
PDF → page-preserving extraction → overlapping 180-word passages → local nomic embeddings → SQLite → cosine similarity → top-k evidence → local structured answer → citation-ID validation → human review.

All model calls target loopback Ollama. No API key is required. SQLite stores uploaded files, extracted text, embedding vectors, evaluation outputs and reviews locally in data/. This folder is ignored by Git. No browsing or computer-control tools are provided to models. The app binds to loopback; do not publish it as a multi-user service without adding authentication and resource controls.

## Evaluation definitions
- Retrieval hit: at least one labelled document/page is present among top-k evidence. This is page-level hit rate, not complete evidence recall.
- Abstention behaviour: the answer's insufficient_evidence flag matches the question label, on both answerable and unanswerable cases.
- Citation-ID validation: rejects nonexistent IDs and uncited non-abstaining answers. It does not verify semantic support.
- Human correctness and support: percentages over reviewed cases only. Unreviewed cases remain empty.
- Timing: end-to-end retrieval and generation, including model loads. Per-run records contain model, top-k, corpus IDs and source excerpts. Failed cases are reported separately and excluded from quality percentages.

The demo contains six questions and fictional source documents. It is a smoke test, not a credible benchmark. Build at least 20 independently labelled questions from real papers; use held-out questions when tuning retrieval. Compare models sequentially on this 16 GB Mac.

## Limits
20 MB, 150 pages, 600 passages per uploaded PDF. No OCR, image interpretation, equation understanding or table reconstruction. PDF page positions may differ from printed labels. Chunking is word-based and can split sentences. The answer context is limited to 4096 tokens and five passages; long evidence may exceed some models' context. Local models can still hallucinate and misjudge abstention. Inspect sources before using answers. Duplicate uploads are detected by content hash. The index does not automatically migrate when the embedding model or chunking settings change.

## Portfolio next steps
Use three public papers, write a 20-question test set with supporting pages, compare two models, record a 60-second demo, and publish real results plus failures. Do not represent synthetic fixture numbers as benchmark results. Future improvements: OCR, reranking, streaming responses, evaluation dataset versioning, a PDF page viewer, and claim-level citation checks.

## Development
Python 3.11. Install requirements in `.venv`; run `.venv/bin/python -m unittest discover -s tests`. Test live services with `.venv/bin/python smoke.py`. The smoke check creates only fictional demo data and a recorded evaluation run. `requirements-lock.txt` records the installed versions.
