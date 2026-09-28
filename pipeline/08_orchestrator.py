"""08_ORCHESTRATOR: Agentic RAG Pipeline with Tool Routing, Evidence Gating, and Structured Output."""

import time
import json
import importlib
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Dynamically import numbered pipeline stages
_ingest_mod = importlib.import_module(".01_ingestion", package=__package__)
DocumentConnector = _ingest_mod.DocumentConnector
PostgreSQLConnector = _ingest_mod.PostgreSQLConnector
LoadedDocument = _ingest_mod.LoadedDocument

_prep_mod = importlib.import_module(".02_preprocessing", package=__package__)
DocumentPreprocessor = _prep_mod.DocumentPreprocessor

_chunk_mod = importlib.import_module(".03_chunking", package=__package__)
RecursiveChunker = _chunk_mod.RecursiveChunker
DocumentChunk = _chunk_mod.DocumentChunk

_emb_mod = importlib.import_module(".04_embeddings_and_vector_db", package=__package__)
EmbeddingProvider = _emb_mod.EmbeddingProvider
VectorStore = _emb_mod.VectorStore

_bm_mod = importlib.import_module(".05_bm25_index", package=__package__)
BM25Index = _bm_mod.BM25Index

_ret_mod = importlib.import_module(".06_hybrid_retrieval", package=__package__)
QueryEngine = _ret_mod.QueryEngine
HybridRetriever = _ret_mod.HybridRetriever
CrossEncoderReranker = _ret_mod.CrossEncoderReranker

_gen_mod = importlib.import_module(".07_llm_and_citations", package=__package__)
LLMProvider = _gen_mod.LLMProvider
GroundedGenerator = _gen_mod.GroundedGenerator
RAGResponse = _gen_mod.RAGResponse
Citation = _gen_mod.Citation
generate_alternative_suggestions = _gen_mod.generate_alternative_suggestions

# Import core tools, router, evaluator, and structured contracts
from .tools.calculator import CalculatorTool
from .tools.web_search import WebSearchTool
from .tools.google_drive import GoogleDriveConnector
from .agent_router import AgentRouter, RoutingDecision, detect_small_talk, SMALL_TALK_REPLIES
from .evidence_evaluator import EvidenceEvaluator
from .structured_contract import (
    AgentResponseContract,
    AgentDecision,
    EvidenceState,
    SourceItem,
    ExecutionStep,
    EvidenceSignals,
    validate_or_repair_response
)

logger = logging.getLogger(__name__)

@dataclass
class IngestionSummary:
    """Summary of the document ingestion run."""
    total_documents: int
    total_chunks: int
    total_tokens: int
    duration_seconds: float
    vector_index_path: str
    bm25_index_path: str

class IngestionPipeline:
    """Complete offline pipeline: Load -> Preprocess -> Chunk -> Dual-Index."""

    def __init__(self, config: Dict[str, Any], project_root: Optional[Path] = None):
        self.config = config
        self.root = project_root or Path(__file__).parent.parent.resolve()

        paths = config.get("paths", {})
        knowledge_cfg = config.get("knowledge", {})
        self.raw_dirs = [
            (self.root / paths.get("raw_data_dir", "data/raw"), ["admin"]),
            (self.root / paths.get("documents_dir", "documents"), ["admin"]),
            (self.root / knowledge_cfg.get("public_dir", "data/public"), ["prospect", "admin"]),
        ]
        self.processed_dir = self.root / paths.get("processed_data_dir", "data/processed")
        self.vector_store_dir = self.root / paths.get("vector_store_dir", "data/indices/vector_store")
        self.bm25_index_file = self.root / paths.get("bm25_index_file", "data/indices/bm25_index.json")

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        prep_cfg = config.get("preprocessing", {})
        chunk_cfg = config.get("chunking", {})
        emb_cfg = config.get("models", {}).get("embedding", {})

        supported_exts = prep_cfg.get("supported_extensions", [".txt", ".pdf", ".md", ".docx", ".html", ".htm"])
        self.connector = DocumentConnector(supported_extensions=supported_exts)
        self.pg_connector = PostgreSQLConnector()
        self.preprocessor = DocumentPreprocessor(
            strip_html=prep_cfg.get("strip_html_tags", True),
            normalize_unicode=prep_cfg.get("normalize_unicode", True)
        )
        self.chunker = RecursiveChunker(
            chunk_size=chunk_cfg.get("chunk_size", 512),
            chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
            separators=chunk_cfg.get("separators", None)
        )
        self.embedder = EmbeddingProvider(
            provider=emb_cfg.get("provider", "sentence-transformers"),
            model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
            dimension=emb_cfg.get("dimension", 384),
            batch_size=emb_cfg.get("batch_size", 32)
        )
        self.vector_store = VectorStore(storage_dir=self.vector_store_dir)
        self.bm25_index = BM25Index()

    def run(self, include_postgres: bool = False, pg_table: str = "documents") -> IngestionSummary:
        """Run full document ingestion and index building."""
        start = time.perf_counter()

        # 1. Load files from local folders with deduplication
        raw_docs: List[LoadedDocument] = []
        seen_hashes = set()
        for d, audience in self.raw_dirs:
            if d.exists():
                for doc in self.connector.load_directory(d):
                    doc_hash = DocumentPreprocessor.compute_hash(doc.content)
                    if doc_hash not in seen_hashes:
                        seen_hashes.add(doc_hash)
                        doc.metadata["audience"] = audience
                        raw_docs.append(doc)

        # Optional PostgreSQL ingest
        if include_postgres:
            pg_docs = self.pg_connector.fetch_documents(table_name=pg_table)
            raw_docs.extend(pg_docs)

        logger.info(f"Loaded {len(raw_docs)} documents.")

        # 2. Preprocess, parse, chunk
        all_chunks: List[DocumentChunk] = []
        parent_docs: Dict[str, str] = {}

        for doc in raw_docs:
            sections = self.preprocessor.parse_sections(doc.content)
            clean_body = "\n\n".join(self.preprocessor.normalize_text(s.content) for s in sections)
            meta = self.preprocessor.enrich_metadata(clean_body, doc.metadata)
            meta["filename"] = doc.filename
            meta["source_path"] = doc.source_path
            meta["access_roles"] = ["all", "employee", "admin"] # Default public employee access
            meta.setdefault("audience", ["admin"])

            parent_docs[doc.filename] = clean_body

            doc_chunks = self.chunker.chunk_document(clean_body, doc_id=doc.filename, base_metadata=meta)
            all_chunks.extend(doc_chunks)

        logger.info(f"Generated {len(all_chunks)} chunks.")

        # Save processed chunks for inspection
        with open(self.processed_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "metadata": c.metadata} for c in all_chunks], f, indent=2)

        # 3. Build Vector & BM25 Indices
        if all_chunks:
            texts = [c.text for c in all_chunks]
            embeddings = self.embedder.embed_texts(texts)
            self.vector_store.build(all_chunks, embeddings, parent_docs=parent_docs)

            self.bm25_index.build(all_chunks)
            self.bm25_index.save(self.bm25_index_file)

        elapsed = time.perf_counter() - start
        total_tokens = sum(c.token_count for c in all_chunks)

        return IngestionSummary(
            total_documents=len(raw_docs),
            total_chunks=len(all_chunks),
            total_tokens=total_tokens,
            duration_seconds=round(elapsed, 3),
            vector_index_path=str(self.vector_store_dir),
            bm25_index_path=str(self.bm25_index_file)
        )

class RAGPipeline:
    """Complete Agentic Pipeline with Tool Routing, Evidence Gating, and Strict Structured Output."""

    def __init__(self, config: Dict[str, Any], project_root: Optional[Path] = None):
        self.config = config
        self.root = project_root or Path(__file__).parent.parent.resolve()

        paths = config.get("paths", {})
        self.vector_store_dir = self.root / paths.get("vector_store_dir", "data/indices/vector_store")
        self.bm25_index_file = self.root / paths.get("bm25_index_file", "data/indices/bm25_index.json")

        emb_cfg = config.get("models", {}).get("embedding", {})
        rerank_cfg = config.get("models", {}).get("reranker", {})
        llm_cfg = config.get("models", {}).get("llm", {})
        ret_cfg = config.get("retrieval", {})

        self.embedder = EmbeddingProvider(
            provider=emb_cfg.get("provider", "sentence-transformers"),
            model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
            dimension=emb_cfg.get("dimension", 384)
        )
        self.vector_store = VectorStore(storage_dir=self.vector_store_dir)
        self.bm25_index = BM25Index()

        vec_loaded = self.vector_store.load()
        bm_loaded = self.bm25_index.load(self.bm25_index_file)

        if not vec_loaded or not bm_loaded:
            logger.warning(
                "Indices not found or empty. Please run 'python main.py --mode ingest' first to index documents."
            )

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedder=self.embedder,
            fusion_method=ret_cfg.get("fusion_method", "rrf"),
            rrf_k=ret_cfg.get("rrf_k", 60),
            vector_weight=ret_cfg.get("vector_weight", 0.6),
            bm25_weight=ret_cfg.get("bm25_weight", 0.4)
        )
        self.reranker = CrossEncoderReranker(
            model_name=rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self.generator = GroundedGenerator(
            llm=LLMProvider(
                provider=llm_cfg.get("provider", "mock"),
                model_name=llm_cfg.get("model_name", "gemini-flash-lite-latest"),
                temperature=llm_cfg.get("temperature", 0.1),
                timeout_seconds=llm_cfg.get("timeout_seconds", 10)
            ),
            embedder=self.embedder
        )
        self.top_k = ret_cfg.get("top_k_vector", 10)
        self.top_n = rerank_cfg.get("top_n", 3)

        # Initialize First-Class Agent Tools
        self.calculator = CalculatorTool
        self.web_search = WebSearchTool
        self.google_drive = GoogleDriveConnector()

    def query(
        self,
        user_question: str,
        user_role: str = "employee",
        user_groups: Optional[List[str]] = None,
        persona: str = "admin"
    ) -> AgentResponseContract:
        """
        Agentic Decision Flow:
        User Question -> Intent Understanding -> Router -> Evidence Evaluation -> Fallback -> Output Schema Validation
        """
        start_time = time.perf_counter()
        trace: List[ExecutionStep] = []
        user_groups = user_groups or []

        # ── 0. Small-talk gate: exact-match greetings/acks skip routing, retrieval and LLM ──
        small_talk = detect_small_talk(user_question)
        if small_talk:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            trace.append(ExecutionStep(stage="small_talk_gate", status="completed", details=f"Intent: {small_talk}", latency_ms=round(elapsed_ms, 2)))
            return AgentResponseContract(
                status="answered",
                answer=SMALL_TALK_REPLIES[small_talk],
                decision=AgentDecision.SMALL_TALK,
                evidence_state=EvidenceState.SUFFICIENT,
                sources=[],
                tools_used=[],
                confidence_label="Direct response",
                confidence_score=1.0,
                execution_trace=trace,
                latency_ms=round(elapsed_ms, 2)
            )

        # ── 1. Understand Intent & Initial Routing Decision ──
        step_start = time.perf_counter()
        initial_route = AgentRouter.route_initial(user_question)
        step_lat = (time.perf_counter() - step_start) * 1000.0
        trace.append(ExecutionStep(
            stage="intent_routing",
            status="completed",
            details=f"Decision: {initial_route.decision.value} | Reason: {initial_route.reasoning}",
            latency_ms=round(step_lat, 2)
        ))

        # ── Branch A: Pure Deterministic Calculator ──
        if initial_route.decision == AgentDecision.CALCULATOR:
            calc_start = time.perf_counter()
            calc_res = self.calculator.extract_and_evaluate(user_question)
            calc_lat = (time.perf_counter() - calc_start) * 1000.0
            trace.append(ExecutionStep(
                stage="calculator_execution",
                status="completed" if calc_res.success else "failed",
                details=f"Expression: {calc_res.expression} -> {calc_res.formatted_result}",
                latency_ms=round(calc_lat, 2)
            ))

            total_lat = (time.perf_counter() - start_time) * 1000.0
            ans_text = calc_res.formatted_result
            if calc_res.steps:
                ans_text = f"**{calc_res.formatted_result}**\n\n*(Calculation: {calc_res.steps})*"

            return AgentResponseContract(
                status="answered" if calc_res.success else "error",
                answer=ans_text,
                decision=AgentDecision.CALCULATOR,
                evidence_state=EvidenceState.SUFFICIENT if calc_res.success else EvidenceState.INSUFFICIENT,
                sources=[],
                tools_used=["calculator"],
                confidence_label="Verified calculation performed",
                confidence_score=1.0 if calc_res.success else 0.0,
                execution_trace=trace,
                latency_ms=round(total_lat, 2)
            )

        # ── Branch B: Direct Authorized Web Search ──
        if initial_route.decision == AgentDecision.WEB_SEARCH:
            web_start = time.perf_counter()
            web_results = self.web_search.search(user_question)
            web_lat = (time.perf_counter() - web_start) * 1000.0
            trace.append(ExecutionStep(
                stage="web_search_execution",
                status="completed" if web_results else "failed",
                details=f"Retrieved {len(web_results)} external sources",
                latency_ms=round(web_lat, 2)
            ))

            if web_results:
                top_r = web_results[0]
                sources = [
                    SourceItem(
                        type="web_search",
                        document_id=r.domain,
                        title=r.title,
                        url=r.url,
                        snippet=r.snippet,
                        score=round(r.authority_score, 4),
                        authority=f"Domain: {r.domain} (Authority: {r.authority_score:.2f})"
                    )
                    for r in web_results
                ]

                # Synthesize verified external answer
                web_context = "\n\n".join([f"Source: {r.title} ({r.url})\nSnippet: {r.snippet}" for r in web_results])
                ans_prompt = f"EXTERNAL TECHNICAL SOURCES:\n{web_context}\n\nUSER QUESTION:\n{user_question}\n\nProvide an authoritative answer directly citing the external source."
                web_answer = self.generator.llm.generate(ans_prompt)
                if not web_answer or "do not specify" in web_answer.lower():
                    web_answer = f"According to verified documentation ({top_r.domain}):\n\n{top_r.snippet}"

                total_lat = (time.perf_counter() - start_time) * 1000.0
                return AgentResponseContract(
                    status="answered",
                    answer=web_answer,
                    decision=AgentDecision.WEB_SEARCH,
                    evidence_state=EvidenceState.SUFFICIENT,
                    sources=sources,
                    tools_used=["web_search"],
                    confidence_label="Additional source used (Public Documentation)",
                    confidence_score=round(top_r.authority_score, 4),
                    citations=[{"chunk_id": s.document_id, "doc_id": s.title, "snippet": s.snippet, "score": s.score} for s in sources],
                    execution_trace=trace,
                    latency_ms=round(total_lat, 2)
                )

        # ── 2. Search Internal Knowledge (Vector + BM25 + Cross-Encoder) ──
        ret_start = time.perf_counter()
        q_info = QueryEngine.process_query(user_question)
        corrected = q_info.get("corrected_query", q_info["cleaned_query"])
        rerank_q = q_info.get("rerank_query", corrected)
        did_you_mean = q_info.get("did_you_mean")

        candidates = self.retriever.retrieve(corrected, top_k=self.top_k)

        # Permission Filtering: restrict chunks by audience (prospects) or role (staff)
        permitted_candidates = []
        for c in candidates:
            c_meta = getattr(c, "metadata", {})
            if persona == "prospect":
                audience = c_meta.get("audience", ["admin"])
                if "prospect" not in audience:
                    continue
                permitted_candidates.append(c)
            else:
                allowed_roles = c_meta.get("access_roles", ["all", "employee"])
                if user_role.lower() == "admin" or "all" in allowed_roles or user_role.lower() in allowed_roles:
                    permitted_candidates.append(c)
                elif any(g.lower() in allowed_roles for g in user_groups):
                    permitted_candidates.append(c)

        top_chunks = self.reranker.rerank(rerank_q, permitted_candidates, top_n=self.top_n)
        ret_lat = (time.perf_counter() - ret_start) * 1000.0
        trace.append(ExecutionStep(
            stage="internal_knowledge_search",
            status="completed",
            details=f"Retrieved {len(candidates)} raw candidates -> {len(permitted_candidates)} RBAC permitted -> {len(top_chunks)} reranked",
            latency_ms=round(ret_lat, 2)
        ))

        # ── 3. Evaluate Retrieved Evidence (Answerability Gate) ──
        eval_start = time.perf_counter()
        ev_eval = EvidenceEvaluator.evaluate(user_question, top_chunks, query_info=q_info)
        eval_lat = (time.perf_counter() - eval_start) * 1000.0
        trace.append(ExecutionStep(
            stage="evidence_evaluation",
            status="completed",
            details=f"State: {ev_eval.state.value.upper()} (Top score: {ev_eval.signals.reranker_top_score:.2f})",
            latency_ms=round(eval_lat, 2)
        ))

        # ── 4. Determine Fallback if Evidence is Insufficient ──
        if ev_eval.state in [EvidenceState.INSUFFICIENT, EvidenceState.PARTIAL]:
            google_drive_available = (persona != "prospect") and self.google_drive.state.get("connected", False)
            fallback_decision, fallback_reason = AgentRouter.evaluate_fallback(
                initial_route=initial_route,
                evidence_state=ev_eval.state.value,
                query=user_question,
                google_drive_available=google_drive_available
            )
            trace.append(ExecutionStep(
                stage="fallback_routing",
                status="completed",
                details=f"Evaluated fallback: {fallback_decision.value} | {fallback_reason}"
            ))

            # Option A: Query Connected Google Drive for unindexed/live documents
            if fallback_decision == AgentDecision.GOOGLE_DRIVE:
                drive_start = time.perf_counter()
                drive_hits = self.google_drive.live_search(user_question, user_role=user_role, max_results=2)
                drive_lat = (time.perf_counter() - drive_start) * 1000.0

                if drive_hits:
                    # Validate Drive evidence before accepting
                    q_kw = q_info.get("keywords", [])
                    has_kw_match = any(any(k in d['content'].lower() for k in q_kw) for d in drive_hits)
                    drive_candidates = [
                        type('DriveChunk', (), {
                            'chunk_id': d['file_id'],
                            'doc_id': d['name'],
                            'text': d['content'],
                            'rerank_score': 1.2 if has_kw_match else -2.5,
                            'fusion_score': 0.8 if has_kw_match else 0.1,
                            'metadata': {'doc_id': d['name'], 'source_type': 'google_drive'}
                        })()
                        for d in drive_hits
                    ]
                    drive_eval = EvidenceEvaluator.evaluate(user_question, drive_candidates, query_info=q_info)

                    if drive_eval.state == EvidenceState.SUFFICIENT and has_kw_match:
                        trace.append(ExecutionStep(
                            stage="google_drive_search",
                            status="completed",
                            details=f"Found {len(drive_hits)} matching Google Drive documents with SUFFICIENT evidence",
                            latency_ms=round(drive_lat, 2)
                        ))
                        # Ground answer using Google Drive documents
                        drive_doc = drive_hits[0]
                        drive_prompt = (
                            f"RETRIEVED CONTEXT\n"
                            f"-----------------\n"
                            f"--- Google Drive Document: {drive_doc['name']} ---\n"
                            f"[File ID: {drive_doc['file_id']} | Location: {drive_doc['drive_location']}]\n"
                            f"{drive_doc['content']}\n"
                            f"-----------------\n\n"
                            f"USER QUESTION\n"
                            f"{user_question}\n\n"
                            f"Now provide the most accurate answer supported by the retrieved context."
                        )
                        drive_ans = self.generator.llm.generate(drive_prompt)
                        sources = [
                            SourceItem(
                                type="google_drive",
                                document_id=d["file_id"],
                                title=d["name"],
                                url=d["source_url"],
                                snippet=d["snippet"],
                                score=0.92,
                                authority=d.get("drive_location")
                            )
                            for d in drive_hits
                        ]

                        if "do not specify" in drive_ans.lower() or "does not contain" in drive_ans.lower() or "not explicitly detail" in drive_ans.lower():
                            total_lat = (time.perf_counter() - start_time) * 1000.0
                            return AgentResponseContract(
                                status="unanswerable",
                                answer="The available company knowledge does not contain enough information to answer this.",
                                decision=AgentDecision.ABSTAIN,
                                evidence_state=EvidenceState.INSUFFICIENT,
                                sources=[],
                                tools_used=["knowledge_search", "google_drive"],
                                confidence_label="Limited information available",
                                confidence_score=0.10,
                                execution_trace=trace,
                                latency_ms=round(total_lat, 2)
                            )

                        total_lat = (time.perf_counter() - start_time) * 1000.0
                        return AgentResponseContract(
                            status="answered",
                            answer=drive_ans,
                            decision=AgentDecision.GOOGLE_DRIVE,
                            evidence_state=EvidenceState.SUFFICIENT,
                            sources=sources,
                            tools_used=["knowledge_search", "google_drive"],
                            confidence_label="Answer based on company Google Drive",
                            confidence_score=0.92,
                            citations=[{"chunk_id": s.document_id, "doc_id": s.title, "snippet": s.snippet, "score": s.score} for s in sources],
                            execution_trace=trace,
                            latency_ms=round(total_lat, 2)
                        )
                    else:
                        trace.append(ExecutionStep(
                            stage="google_drive_search",
                            status="completed",
                            details="Google Drive queried but evidence was INSUFFICIENT",
                            latency_ms=round(drive_lat, 2)
                        ))

            # Option B: Fallback to Authoritative Web Search (ONLY if permitted and not confidential)
            if fallback_decision == AgentDecision.WEB_SEARCH and initial_route.web_search_allowed:
                web_start = time.perf_counter()
                web_hits = self.web_search.search(user_question)
                web_lat = (time.perf_counter() - web_start) * 1000.0

                if web_hits:
                    trace.append(ExecutionStep(
                        stage="web_search_fallback",
                        status="completed",
                        details=f"Retrieved {len(web_hits)} external sources",
                        latency_ms=round(web_lat, 2)
                    ))
                    sources = [
                        SourceItem(
                            type="web_search",
                            document_id=r.domain,
                            title=r.title,
                            url=r.url,
                            snippet=r.snippet,
                            score=r.authority_score,
                            authority=r.domain
                        )
                        for r in web_hits
                    ]
                    top_web = web_hits[0]
                    total_lat = (time.perf_counter() - start_time) * 1000.0
                    return AgentResponseContract(
                        status="answered",
                        answer=f"Verified external reference ({top_web.domain}):\n\n{top_web.snippet}",
                        decision=AgentDecision.WEB_SEARCH,
                        evidence_state=EvidenceState.PARTIAL,
                        sources=sources,
                        tools_used=["knowledge_search", "web_search"],
                        confidence_label="Additional source used",
                        confidence_score=round(top_web.authority_score, 4),
                        citations=[{"chunk_id": s.document_id, "doc_id": s.title, "snippet": s.snippet, "score": s.score} for s in sources],
                        execution_trace=trace,
                        latency_ms=round(total_lat, 2)
                    )

            # Option C: Clean Abstention (Anti-Hallucination)
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return AgentResponseContract(
                status="unanswerable",
                answer="The available company knowledge does not contain enough information to answer this.",
                decision=AgentDecision.ABSTAIN,
                evidence_state=EvidenceState.INSUFFICIENT,
                sources=[],
                tools_used=["knowledge_search"],
                confidence_label="Limited information available",
                confidence_score=0.10,
                execution_trace=trace,
                latency_ms=round(total_lat, 2)
            )

        # ── 5. Handle Conflicting Evidence ──
        if ev_eval.state == EvidenceState.CONFLICTING and ev_eval.conflict_notes:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            sources = [
                SourceItem(
                    type="internal_document",
                    document_id=c.doc_id,
                    title=c.doc_id,
                    snippet=c.text[:140] + "...",
                    score=getattr(c, "rerank_score", 0.0)
                )
                for c in ev_eval.resolved_chunks[:2]
            ]
            ans = f"**Conflicting Company Documentation Detected**\n\n{ev_eval.conflict_notes}\n\nPlease consult HR or legal compliance for the definitive effective version."
            return AgentResponseContract(
                status="answered",
                answer=ans,
                decision=AgentDecision.INTERNAL_RAG,
                evidence_state=EvidenceState.CONFLICTING,
                sources=sources,
                tools_used=["knowledge_search"],
                confidence_label="Conflicting company sources",
                confidence_score=0.50,
                citations=[{"chunk_id": s.document_id, "doc_id": s.title, "snippet": s.snippet, "score": s.score} for s in sources],
                execution_trace=trace,
                latency_ms=round(total_lat, 2)
            )

        # ── 6. Grounded Generation for Sufficient Internal Evidence ──
        gen_start = time.perf_counter()
        raw_resp = self.generator.generate_response(user_question, ev_eval.resolved_chunks, effective_query=rerank_q, persona=persona)
        gen_lat = (time.perf_counter() - gen_start) * 1000.0

        if did_you_mean and not raw_resp.answer.startswith("The available company documents do not specify"):
            raw_resp.answer = f"Did you mean '{did_you_mean}'?\n\n{raw_resp.answer}"

        sources = [
            SourceItem(
                type="internal_document",
                document_id=c.doc_id,
                title=c.doc_id,
                snippet=c.snippet,
                score=c.score,
                authority="Authoritative Company Knowledge"
            )
            for c in raw_resp.citations
        ]

        trace.append(ExecutionStep(
            stage="structured_generation",
            status="completed",
            details="Grounding completed with strict citations",
            latency_ms=round(gen_lat, 2)
        ))

        total_lat = (time.perf_counter() - start_time) * 1000.0

        is_unanswerable = (
            raw_resp.answer.startswith("The available company documents do not specify") or
            raw_resp.answer.startswith("While our available documentation does not explicitly detail")
        )

        final_decision = AgentDecision.ABSTAIN if is_unanswerable else AgentDecision.INTERNAL_RAG
        final_state = EvidenceState.INSUFFICIENT if is_unanswerable else ev_eval.state
        final_status = "unanswerable" if is_unanswerable else "answered"
        final_label = "Limited information available" if is_unanswerable else ev_eval.confidence_label
        final_score = 0.10 if is_unanswerable else ev_eval.confidence_score

        # Build schema-validated response
        contract_data = {
            "status": final_status,
            "answer": raw_resp.answer,
            "decision": final_decision,
            "evidence_state": final_state,
            "sources": sources if not is_unanswerable else [],
            "tools_used": ["knowledge_search"],
            "requires_followup": False,
            "confidence_label": final_label,
            "confidence_score": final_score,
            "citations": [{"chunk_id": s.document_id, "doc_id": s.title, "snippet": s.snippet, "score": s.score} for s in sources] if not is_unanswerable else [],
            "num_context_chunks": len(ev_eval.resolved_chunks) if not is_unanswerable else 0,
            "context_texts": raw_resp.context_texts if not is_unanswerable else [],
            "evidence_signals": ev_eval.signals,
            "execution_trace": trace,
            "latency_ms": round(total_lat, 2),
            "suggested_queries": raw_resp.suggested_queries
        }

        return validate_or_repair_response(contract_data)

    def _web_search_conversational(self, user_question: str) -> Optional[tuple]:
        """Run authoritative web search and format a conversational answer for the public widget."""
        web_hits = self.web_search.search(user_question)
        if not web_hits:
            return None

        top = web_hits[0]
        context = "\n\n".join(f"Source: {r.title} ({r.url})\n{r.snippet}" for r in web_hits[:3])
        prompt = (
            f"EXTERNAL PUBLIC SOURCES\n"
            f"-----------------------\n{context}\n"
            f"-----------------------\n\n"
            f"VISITOR'S QUESTION\n{user_question}\n\n"
            f"Answer the visitor's question directly and concisely using only these public sources. "
            f"Do not invent facts. If OrionSoft's software services are relevant, you may close with one short offer to help."
        )
        answer = self.generator.llm.generate(prompt)
        if not answer or answer.strip() == "" or answer.startswith("The available company documents do not specify"):
            answer = f"Based on {top.domain}:\n\n{top.snippet}"

        suggestions = generate_alternative_suggestions(user_question, embedder=self.embedder)[:4]
        return answer, suggestions

    def conversational_query(
        self,
        user_question: str,
        history: Optional[List[Dict[str, str]]] = None,
        visitor_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Natural conversational path for the public website chatbot.

        Retrieves prospect-safe knowledge, then asks the LLM to produce a friendly
        grounded answer plus four dynamic, context-aware suggested questions.
        Returns a plain dict: {answer, suggested_questions, retrieved_chunks, latency_ms}.
        """
        start_time = time.perf_counter()
        history = history or []

        small_talk = detect_small_talk(user_question)
        if small_talk:
            suggestions = []
            if small_talk in ("rephrase", "greeting", "filler"):
                suggestions = [
                    "What software development services do you offer?",
                    "Can you build custom AI chatbots and web apps?",
                    "Do you work with clients outside India?",
                    "How much does it cost to build a mobile app?",
                ]
            return {
                "answer": SMALL_TALK_REPLIES[small_talk],
                "suggested_questions": suggestions,
                "retrieved_chunks": 0,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            }

        prospect_web_search = bool(self.config.get("knowledge", {}).get("prospect_web_search", True))
        initial_route = AgentRouter.route_initial(user_question)

        # Direct external/public questions (e.g. latest tech releases) go to web search.
        if prospect_web_search and initial_route.decision == AgentDecision.WEB_SEARCH:
            web_answer = self._web_search_conversational(user_question)
            if web_answer is not None:
                answer, suggestions = web_answer
                return {
                    "answer": answer,
                    "suggested_questions": suggestions,
                    "retrieved_chunks": 0,
                    "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                }

        q_info = QueryEngine.process_query(user_question)
        corrected = q_info.get("corrected_query", q_info["cleaned_query"])
        rerank_q = q_info.get("rerank_query", corrected)

        candidates = self.retriever.retrieve(corrected, top_k=self.top_k)

        # Restrict to prospect-safe knowledge (never expose internal documents)
        permitted = [c for c in candidates if "prospect" in c.metadata.get("audience", ["admin"])]
        if not permitted:
            permitted = candidates[: self.top_n]

        top_chunks = self.reranker.rerank(rerank_q, permitted, top_n=self.top_n) if permitted else []

        # Non-confidential questions with no relevant company knowledge fall back to web search.
        if prospect_web_search and not initial_route.is_internal_confidential:
            ev = EvidenceEvaluator.evaluate(user_question, top_chunks, query_info=q_info)
            if ev.state == EvidenceState.INSUFFICIENT:
                web_answer = self._web_search_conversational(user_question)
                if web_answer is not None:
                    answer, suggestions = web_answer
                    return {
                        "answer": answer,
                        "suggested_questions": suggestions,
                        "retrieved_chunks": 0,
                        "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                    }

        answer, suggestions, _gen_ms = self.generator.generate_conversational_response(
            user_question, top_chunks, history=history, visitor_name=visitor_name
        )

        total_lat = (time.perf_counter() - start_time) * 1000.0

        return {
            "answer": answer,
            "suggested_questions": suggestions,
            "retrieved_chunks": len(top_chunks),
            "latency_ms": round(total_lat, 2),
        }
