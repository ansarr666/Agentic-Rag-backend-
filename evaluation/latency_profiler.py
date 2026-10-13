"""Latency Profiler and Bottleneck Diagnostic Tool.

Measures response times across:
1. Webhook & request handling overhead
2. Website-to-lead / handoff submission (/api/public/lead, /api/public/intent)
3. AI call latency (Gemini LLM API)
4. Retrieval & Reranking latency (Hybrid BM25 + Vector + Cross-Encoder)
5. Database persistence latency (PostgreSQL / SQLite / JSON dual-write)
6. Reply delivery serialization
"""

import time
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from pipeline import RAGPipeline
import db
logger = logging.getLogger(__name__)


def profile_components(num_runs: int = 3):
    config = load_config()
    db.configure_from_env()
    rag = RAGPipeline(config, project_root=PROJECT_ROOT)

    test_queries = [
        "What software development services does OrionSoft provide?",
        "Can you build a mobile app for iOS and Android?",
        "How much does a custom CRM or website cost?",
    ]

    results = {
        "hybrid_retrieval_ms": [],
        "cross_encoder_rerank_ms": [],
        "ai_llm_call_ms": [],
        "db_persist_turn_ms": [],
        "lead_submission_ms": [],
        "handoff_intent_ms": [],
        "total_conversational_ms": [],
    }

    print("=" * 70)
    print("PROFILING AGENTIC RAG SYSTEM LATENCY")
    print("=" * 70)

    # 1. Profile Retrieval & AI Call in isolation
    for q in test_queries:
        print(f"\nEvaluating query: '{q}'")
        
        # Retrieval stage
        t0 = time.perf_counter()
        candidates = rag.retriever.retrieve(q, top_k=rag.top_k)
        ret_ms = (time.perf_counter() - t0) * 1000.0
        results["hybrid_retrieval_ms"].append(ret_ms)
        print(f"  ├── Hybrid Retrieval:       {ret_ms:.2f} ms ({len(candidates)} candidates)")

        # Rerank stage
        t0 = time.perf_counter()
        top_chunks = rag.reranker.rerank(q, candidates[:10], top_n=rag.top_n)
        rerank_ms = (time.perf_counter() - t0) * 1000.0
        results["cross_encoder_rerank_ms"].append(rerank_ms)
        print(f"  ├── Cross-Encoder Rerank:   {rerank_ms:.2f} ms ({len(top_chunks)} chunks)")

        # LLM Call stage
        t0 = time.perf_counter()
        ans, sugs, gen_ms = rag.generator.generate_conversational_response(q, top_chunks)
        results["ai_llm_call_ms"].append(gen_ms)
        print(f"  ├── AI (Gemini) Call:       {gen_ms:.2f} ms")

        # Database Persistence stage (measuring dual write: JSON + DB)
        t0 = time.perf_counter()
        cid = f"prof_conv_{int(time.time()*1000)}"
        user_id = f"prof_user_{int(time.time()*1000)}"
        try:
            db.upsert_user(user_id=user_id, email=None, role="guest", provider="anonymous")
            db.ensure_conversation(conversation_id=cid, user_id=user_id, title="Profile Chat")
            db.add_message(cid, role="user", content=q, metadata={})
            db.add_message(cid, role="assistant", content=ans, metadata={})
        except Exception as e:
            print(f"  ├── DB Error: {e}")
        db_ms = (time.perf_counter() - t0) * 1000.0
        results["db_persist_turn_ms"].append(db_ms)
        print(f"  ├── DB Persistence (4 ops): {db_ms:.2f} ms")

        # Full Conversational Query (Uncached - First Run)
        t0 = time.perf_counter()
        full_resp = rag.conversational_query(q)
        total_ms = (time.perf_counter() - t0) * 1000.0
        results["total_conversational_ms"].append(total_ms)
        print(f"  ├── Full Pipeline (Uncached): {total_ms:.2f} ms")

        # Full Conversational Query (Cached - Second Run)
        t0 = time.perf_counter()
        cached_resp = rag.conversational_query(q)
        cached_ms = (time.perf_counter() - t0) * 1000.0
        results.setdefault("cached_conversational_ms", []).append(cached_ms)
        is_hit = cached_resp.get("cached", False)
        print(f"  └── Full Pipeline (Cached):   {cached_ms:.2f} ms (Cache Hit: {is_hit})")

    # 2. Profile Lead Submission & Handoff Intent
    print("\nProfiling Form Submissions & Handoff:")
    from app import app
    client = app.test_client()

    for i in range(num_runs):
        lead_payload = {
            "name": f"Test Lead {i}",
            "email": f"test.lead{i}@gmail.com",
            "phone": "9876543210",
            "company": "Test Enterprise",
            "project_requirement": "Custom AI & Mobile App development"
        }
        t0 = time.perf_counter()
        res = client.post("/api/public/lead", json=lead_payload)
        lead_ms = (time.perf_counter() - t0) * 1000.0
        results["lead_submission_ms"].append(lead_ms)
        lead_id = res.get_json().get("lead_id")

        handoff_payload = {
            "lead_id": lead_id,
            "conversation_id": f"conv_prof_{i}",
            "phone": "9876543210",
            "requested_action": "Book a Discovery Call",
            "summary": "Visitor interested in mobile app"
        }
        t0 = time.perf_counter()
        res_h = client.post("/api/public/intent", json=handoff_payload)
        handoff_ms = (time.perf_counter() - t0) * 1000.0
        results["handoff_intent_ms"].append(handoff_ms)

    # 3. Print Summary & Component Breakdown
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    uncached_avg = avg(results['total_conversational_ms'])
    cached_avg = avg(results.get('cached_conversational_ms', []))
    speedup = (uncached_avg / cached_avg) if cached_avg > 0 else 1.0

    print("\n" + "=" * 70)
    print("COMPONENT LATENCY SUMMARY & BOTTLENECK ANALYSIS")
    print("=" * 70)
    print(f"  1. Hybrid Retrieval:             {avg(results['hybrid_retrieval_ms']):.2f} ms")
    print(f"  2. Cross-Encoder Reranking:       {avg(results['cross_encoder_rerank_ms']):.2f} ms")
    print(f"  3. AI LLM API Call (Gemini):      {avg(results['ai_llm_call_ms']):.2f} ms")
    print(f"  4. DB Persistence (per turn):     {avg(results['db_persist_turn_ms']):.2f} ms")
    print(f"  5. Website-to-Lead submission:    {avg(results['lead_submission_ms']):.2f} ms")
    print(f"  6. Discovery/Handoff Booking:     {avg(results['handoff_intent_ms']):.2f} ms")
    print(f"  7. Total Conversational (Cold):   {uncached_avg:.2f} ms")
    print(f"  8. Total Conversational (Cached): {cached_avg:.2f} ms ({speedup:.1f}x faster)")
    print("=" * 70)

    ai_pct = (avg(results['ai_llm_call_ms']) / uncached_avg) * 100 if uncached_avg > 0 else 0
    db_pct = (avg(results['db_persist_turn_ms']) / uncached_avg) * 100 if uncached_avg > 0 else 0

    print(f"\nIDENTIFIED BOTTLENECKS & IMPACT:")
    print(f"  • AI LLM Call accounts for:       {ai_pct:.1f}% of cold conversational response time.")
    print(f"  • Database persistence:           {avg(results['db_persist_turn_ms']):.2f} ms if synchronous.")
    print(f"  • Latency reduction with Cache:   {uncached_avg - cached_avg:.2f} ms saved per cached response.")

    return results


if __name__ == "__main__":
    profile_components(num_runs=3)
