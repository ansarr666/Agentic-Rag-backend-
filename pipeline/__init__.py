"""Pipeline package: Unified, sequential RAG pipeline stages 01 to 08."""

import importlib

# Dynamic import of sequential numbered pipeline stages
_m1 = importlib.import_module(".01_ingestion", package=__name__)
_m2 = importlib.import_module(".02_preprocessing", package=__name__)
_m3 = importlib.import_module(".03_chunking", package=__name__)
_m4 = importlib.import_module(".04_embeddings_and_vector_db", package=__name__)
_m5 = importlib.import_module(".05_bm25_index", package=__name__)
_m6 = importlib.import_module(".06_hybrid_retrieval", package=__name__)
_m7 = importlib.import_module(".07_llm_and_citations", package=__name__)
_m8 = importlib.import_module(".08_orchestrator", package=__name__)

# 01 Ingestion
DocumentConnector = _m1.DocumentConnector
PostgreSQLConnector = _m1.PostgreSQLConnector
LoadedDocument = _m1.LoadedDocument

# 02 Preprocessing
DocumentPreprocessor = _m2.DocumentPreprocessor
ParsedSection = _m2.ParsedSection

# 03 Chunking
RecursiveChunker = _m3.RecursiveChunker
DocumentChunk = _m3.DocumentChunk

# 04 Embeddings & Vector DB
EmbeddingProvider = _m4.EmbeddingProvider
VectorStore = _m4.VectorStore
VectorResult = _m4.VectorResult

# 05 BM25 Index
BM25Index = _m5.BM25Index
BM25Result = _m5.BM25Result

# 06 Hybrid Retrieval & Reranking
QueryEngine = _m6.QueryEngine
HybridRetriever = _m6.HybridRetriever
CrossEncoderReranker = _m6.CrossEncoderReranker
RetrievedCandidate = _m6.RetrievedCandidate

# 07 LLM & Citations
LLMProvider = _m7.LLMProvider
GroundedGenerator = _m7.GroundedGenerator
RAGResponse = _m7.RAGResponse
Citation = _m7.Citation

# 08 Orchestrator
IngestionPipeline = _m8.IngestionPipeline
IngestionSummary = _m8.IngestionSummary
RAGPipeline = _m8.RAGPipeline

# Agentic Capabilities & Structured Contracts
from .tools.calculator import CalculatorTool, CalculationResult
from .tools.web_search import WebSearchTool, WebSearchResult
from .tools.google_drive import GoogleDriveConnector, DriveSyncSummary
from .agent_router import AgentRouter, RoutingDecision
from .evidence_evaluator import EvidenceEvaluator, EvaluationGateResult
from .structured_contract import (
    AgentResponseContract,
    AgentDecision,
    EvidenceState,
    SourceItem,
    ExecutionStep,
    EvidenceSignals,
    validate_or_repair_response
)

__all__ = [
    "DocumentConnector",
    "PostgreSQLConnector",
    "LoadedDocument",
    "DocumentPreprocessor",
    "ParsedSection",
    "RecursiveChunker",
    "DocumentChunk",
    "EmbeddingProvider",
    "VectorStore",
    "VectorResult",
    "BM25Index",
    "BM25Result",
    "QueryEngine",
    "HybridRetriever",
    "CrossEncoderReranker",
    "RetrievedCandidate",
    "LLMProvider",
    "GroundedGenerator",
    "RAGResponse",
    "Citation",
    "IngestionPipeline",
    "IngestionSummary",
    "RAGPipeline",
    "CalculatorTool",
    "CalculationResult",
    "WebSearchTool",
    "WebSearchResult",
    "GoogleDriveConnector",
    "DriveSyncSummary",
    "AgentRouter",
    "RoutingDecision",
    "EvidenceEvaluator",
    "EvaluationGateResult",
    "AgentResponseContract",
    "AgentDecision",
    "EvidenceState",
    "SourceItem",
    "ExecutionStep",
    "EvidenceSignals",
    "validate_or_repair_response"
]
