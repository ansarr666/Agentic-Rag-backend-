"""CLI entrypoint for Agentic RAG: Ingestion, Interactive Querying, and Evaluation."""

import sys
import argparse
import logging
from pathlib import Path

# Ensure UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from pipeline import IngestionPipeline, RAGPipeline
from evaluation.run_eval import run_evaluation

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG: Enterprise Ingestion, Hybrid Retrieval & Grounded Generation."
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "query", "interactive", "eval"],
        default="interactive",
        help="Operation mode: 'ingest' (index docs), 'query' (single Q&A), 'interactive' (REPL), 'eval' (benchmark)."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file.")
    parser.add_argument("--question", type=str, help="Question to ask (required for --mode query).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")

    args = parser.parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    if args.mode == "ingest":
        print("\n[+] Starting Document Ingestion Pipeline...")
        pipeline = IngestionPipeline(config, project_root=PROJECT_ROOT)
        summary = pipeline.run()
        print("\n" + "="*50)
        print("          INGESTION COMPLETED")
        print("="*50)
        print(f"Total Documents : {summary.total_documents}")
        print(f"Total Chunks    : {summary.total_chunks}")
        print(f"Total Tokens    : {summary.total_tokens}")
        print(f"Elapsed Time    : {summary.duration_seconds:.2f}s")
        print(f"Vector Store    : {summary.vector_index_path}")
        print(f"BM25 Index      : {summary.bm25_index_path}")
        print("="*50 + "\n")

    elif args.mode == "query":
        if not args.question:
            print("Error: --question is required when running in query mode.")
            sys.exit(1)
        rag = RAGPipeline(config, project_root=PROJECT_ROOT)
        response = rag.query(args.question)
        print("\n" + "="*60)
        print(f"QUESTION: {args.question}")
        print("="*60)
        print(f"\nANSWER:\n{response.answer}\n")
        print(f"Decision         : {response.decision.value} ({response.confidence_label})")
        print(f"Confidence Score : {response.confidence_score * 100:.1f}%")
        print(f"Latency          : {response.latency_ms} ms")
        if response.sources:
            print("Sources:")
            for s in response.sources:
                print(f"  - [{s.type}] {s.title} ({s.document_id}) (Score: {s.score:.4f})")
                print(f"    {s.snippet[:120]}...")
        elif response.citations:
            print("Sources:")
            for c in response.citations:
                doc_id = c.get("doc_id", "unknown") if isinstance(c, dict) else getattr(c, "doc_id", "unknown")
                chunk_id = c.get("chunk_id", "") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                score = c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0)
                snippet = c.get("snippet", "") if isinstance(c, dict) else getattr(c, "snippet", "")
                print(f"  - [{chunk_id}] {doc_id} (Score: {score:.4f})")
                print(f"    {snippet[:120]}...")
        print("="*60 + "\n")

    elif args.mode == "interactive":
        print("\n" + "="*60)
        print("  🧠 Welcome to Agentic RAG Interactive Console")
        print("  Type your question and press Enter. Type 'exit' to quit.")
        print("="*60 + "\n")
        rag = RAGPipeline(config, project_root=PROJECT_ROOT)

        while True:
            try:
                user_input = input("User >> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Exiting. Goodbye!")
                    break

                response = rag.query(user_input)
                print("\n" + "-"*60)
                print(f"Assistant >>\n{response.answer}")
                print(f"\n[Latency: {response.latency_ms}ms | Confidence: {response.confidence_score * 100:.1f}%]")
                if response.sources:
                    print("Sources:")
                    for s in response.sources:
                        print(f"  * [{s.type}] {s.title} -> \"{s.snippet[:90]}...\"")
                elif response.citations:
                    print("Citations:")
                    for c in response.citations:
                        doc_id = c.get("doc_id", "unknown") if isinstance(c, dict) else getattr(c, "doc_id", "unknown")
                        chunk_id = c.get("chunk_id", "") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                        snippet = c.get("snippet", "") if isinstance(c, dict) else getattr(c, "snippet", "")
                        print(f"  * {doc_id} [{chunk_id}] -> \"{snippet[:90]}...\"")
                print("-"*60 + "\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break

    elif args.mode == "eval":
        print("\n[+] Running Automated Benchmark Evaluation...")
        run_evaluation(args.config)

if __name__ == "__main__":
    main()
