"""06_HYBRID_RETRIEVAL: Query Understanding, Reciprocal Rank Fusion (RRF) & Reranking."""

import re
import difflib
import importlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# Dynamic import of sibling numbered modules
_emb_mod = importlib.import_module(".04_embeddings_and_vector_db", package=__package__)
VectorStore = _emb_mod.VectorStore
EmbeddingProvider = _emb_mod.EmbeddingProvider
VectorResult = _emb_mod.VectorResult

_bm_mod = importlib.import_module(".05_bm25_index", package=__package__)
BM25Index = _bm_mod.BM25Index
BM25Result = _bm_mod.BM25Result

logger = logging.getLogger(__name__)

@dataclass
class RetrievedCandidate:
    """Unified retrieved chunk candidate with fusion and rerank scores."""
    chunk_id: str
    doc_id: str
    text: str
    fusion_score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryEngine:
    """Classifies user intent and generates keyword/semantic query variants."""

    INTENT_MAP = {
        "service_inquiry": ["service", "offer", "provide", "do you", "what do you", "help", "work with"],
        "app_development": ["app", "mobile", "android", "iphone", "zomato", "uber", "on demand", "tracking", "login"],
        "software_development": ["software", "custom", "booking", "inventory", "claims", "internal tool", "build"],
        "cloud_services": ["cloud", "aws", "google cloud", "migrate", "hosting", "server"],
        "cybersecurity": ["security", "cybersecurity", "hack", "threat", "protect", "secure", "vulnerability"],
        "ai_and_chatbot": ["ai", "chatbot", "generative", "automate", "recommend", "personalize", "machine learning"],
        "staff_augmentation": ["hire", "developer", "augmentation", "dedicated", "team member", "in house"],
        "seo_services": ["seo", "search engine", "ranking", "traffic", "google ranking", "organic"],
        "paid_advertising": ["ads", "google ads", "facebook ads", "paid", "campaign", "budget", "ppc"],
        "social_media": ["social media", "instagram", "facebook", "linkedin", "posting", "content creation"],
        "branding_design": ["logo", "brand", "branding", "design", "identity", "colors", "fonts"],
        "pricing_engagement": ["price", "cost", "package", "free call", "discovery call", "quote", "budget"],
        "technical_explanation": ["how", "architecture", "component", "difference", "compare", "vs"],
        "general_factual": ["what", "who", "where", "about", "company", "experience", "portfolio", "clients"]
    }

    GREETINGS = {
        "hi", "hello", "hey", "hola", "howdy", "greetings", "good morning",
        "good afternoon", "good evening", "good day", "sup", "yo", "wassup", "hiya"
    }
    FAREWELLS = {
        "bye", "goodbye", "see you", "see ya", "cya", "farewell", "take care",
        "have a good day", "have a nice day", "good night", "tata"
    }
    GRATITUDE = {
        "thanks", "thank you", "thx", "ty", "appreciate it", "much appreciated",
        "thank you so much", "thanks a lot"
    }
    IDENTITY_HELP = {
        "who are you", "what are you", "what can you do", "help", "help me", "how to use"
    }
    WELL_BEING = {
        "how are you", "how are you doing", "how are you today", "how are u",
        "how r u", "hows it going", "how is it going", "how do you do",
        "whats up", "what is up", "how is everything", "how are things"
    }

    @classmethod
    def detect_conversational(cls, query: str) -> Optional[str]:
        """Detect if query is purely conversational (greeting, farewell, gratitude, help, well-being)."""
        q = re.sub(r"[^\w\s]", " ", query.lower())
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            return None

        # 1. Well-being patterns (e.g., "how are you", "hi how are you", "hello how are you doing", "how r u")
        well_being_patterns = [
            r"\bhow\s+(are|r)\s+(you|u)\b",
            r"\bhow\s+are\s+things\b",
            r"\bhow\s*s\s+it\s+going\b",
            r"\bhow\s+is\s+it\s+going\b",
            r"\bhow\s+do\s+you\s+do\b",
            r"\bwhat\s*s\s+up\b",
            r"\bwhat\s+is\s+up\b"
        ]
        for pat in well_being_patterns:
            if re.search(pat, q):
                return "well_being"

        # 2. Identity / capability help
        identity_patterns = [
            r"\bwho\s+are\s+you\b",
            r"\bwhat\s+are\s+you\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bwho\s+made\s+you\b",
            r"\bwho\s+created\s+you\b"
        ]
        for pat in identity_patterns:
            if re.search(pat, q):
                return "identity_help"

        # 3. Gratitude
        if any(w in q.split() for w in ["thanks", "thank", "thx", "ty", "appreciate"]):
            return "gratitude"

        # 4. Farewell
        farewell_phrases = ["bye", "goodbye", "see you", "see ya", "cya", "farewell", "take care", "good night", "tata"]
        if any(f in q for f in farewell_phrases):
            return "farewell"

        # 5. Greetings (e.g., "hi", "hello", "hey", "good morning", "hi there")
        greeting_phrases = ["good morning", "good afternoon", "good evening", "good day"]
        greeting_words = {"hi", "hello", "hey", "hola", "howdy", "greetings", "sup", "yo", "wassup", "hiya"}
        
        # Check if entire query or leading phrase is a greeting without a business query attached
        words = q.split()
        if len(words) <= 4:
            if any(gp in q for gp in greeting_phrases):
                return "greeting"
            if any(w in greeting_words for w in words):
                return "greeting"

        return None

    SPELL_AND_SYNONYM_MAP = {
        # E-commerce / Online Stores
        "ecommes": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "e-commes": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "ecomm": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "ecom": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "e-comm": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "e-com": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "ecomerce": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "ecommerece": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "eccomerce": {"display": "e-commerce", "search": "ecommerce e-commerce online store"},
        "onlinestore": {"display": "online store", "search": "online store ecommerce"},
        "shopping site": {"display": "online store", "search": "online store ecommerce"},
        "shopping website": {"display": "online store", "search": "online store ecommerce"},

        # Websites
        "websit": {"display": "website", "search": "website websites"},
        "webiste": {"display": "website", "search": "website websites"},
        "websits": {"display": "websites", "search": "websites"},
        "webstie": {"display": "website", "search": "website websites"},
        "wbsite": {"display": "website", "search": "website websites"},

        # Mobile / Apps
        "androyd": {"display": "Android", "search": "android mobile app"},
        "andriod": {"display": "Android", "search": "android mobile app"},
        "andorid": {"display": "Android", "search": "android mobile app"},
        "iphne": {"display": "iPhone", "search": "iPhone iOS mobile app"},
        "iphn": {"display": "iPhone", "search": "iPhone iOS mobile app"},
        "moblie": {"display": "mobile", "search": "mobile app"},
        "mobl": {"display": "mobile", "search": "mobile app"},
        "appp": {"display": "app", "search": "app mobile"},
        "apss": {"display": "apps", "search": "apps mobile"},

        # Developers & Staffing
        "developper": {"display": "developer", "search": "developer developers hire staff"},
        "develoepr": {"display": "developer", "search": "developer developers hire staff"},
        "developr": {"display": "developer", "search": "developer developers hire staff"},
        "devloper": {"display": "developer", "search": "developer developers hire staff"},
        "programer": {"display": "developer", "search": "developer programmers"},
        "progammer": {"display": "developer", "search": "developer programmers"},
        "enginer": {"display": "engineer", "search": "engineer developer"},

        # Software
        "softare": {"display": "software", "search": "software custom application"},
        "softwar": {"display": "software", "search": "software custom application"},
        "softwaer": {"display": "software", "search": "software custom application"},
        "softwer": {"display": "software", "search": "software custom application"},

        # Pricing & Costs
        "pricng": {"display": "pricing", "search": "pricing cost budget estimate"},
        "prizing": {"display": "pricing", "search": "pricing cost budget estimate"},
        "priec": {"display": "pricing", "search": "pricing cost budget estimate"},
        "budgt": {"display": "budget", "search": "budget cost pricing estimate"},
        "expens": {"display": "expense", "search": "expense cost pricing"},

        # Security
        "secuirty": {"display": "security", "search": "security cybersecurity protect"},
        "securty": {"display": "security", "search": "security cybersecurity protect"},
        "secuity": {"display": "security", "search": "security cybersecurity protect"},
        "cybr": {"display": "cybersecurity", "search": "cybersecurity security protect"},

        # Chatbots & AI
        "chatbt": {"display": "chatbot", "search": "chatbot AI assistant WhatsApp"},
        "chabot": {"display": "chatbot", "search": "chatbot AI assistant WhatsApp"},
        "chatbto": {"display": "chatbot", "search": "chatbot AI assistant WhatsApp"},
        "whatapp": {"display": "WhatsApp", "search": "WhatsApp chatbot"},
        "watsapp": {"display": "WhatsApp", "search": "WhatsApp chatbot"},
        "whatsap": {"display": "WhatsApp", "search": "WhatsApp chatbot"},

        # Marketing & SEO
        "marketng": {"display": "marketing", "search": "marketing digital ads SEO"},
        "markting": {"display": "marketing", "search": "marketing digital ads SEO"},
        "markteing": {"display": "marketing", "search": "marketing digital ads SEO"},

        # Cloud & Servers
        "clowd": {"display": "cloud", "search": "cloud AWS Google Cloud hosting"},
        "cluod": {"display": "cloud", "search": "cloud AWS Google Cloud hosting"},
        "serivr": {"display": "server", "search": "server hosting cloud"},

        # Custom & Payments
        "custm": {"display": "custom", "search": "custom software development"},
        "custome": {"display": "custom", "search": "custom software development"},
        "paymnt": {"display": "payment", "search": "payment gateway Stripe Razorpay"},
        "pyament": {"display": "payment", "search": "payment gateway Stripe Razorpay"}
    }

    CANONICAL_DOMAIN_TERMS = [
        "ecommerce", "commerce", "website", "websites", "software",
        "mobile", "android", "iphone", "ios", "application",
        "developer", "developers", "hiring", "augmentation", "cloud",
        "aws", "cybersecurity", "security", "pricing", "estimate",
        "chatbot", "ai", "whatsapp", "seo", "marketing", "ads",
        "payment", "custom", "store", "online", "discovery"
    ]

    @classmethod
    def correct_spelling_and_synonyms(cls, query_text: str) -> Tuple[str, Optional[str]]:
        """Correct minor typos, expand search synonyms, and return (search_query, did_you_mean)."""
        tokens = query_text.split()
        search_tokens = []
        corrections = []
        for t in tokens:
            clean_t = re.sub(r"[^\w\-]", "", t.lower())
            if not clean_t:
                search_tokens.append(t)
                continue

            # 1. Direct dictionary match
            if clean_t in cls.SPELL_AND_SYNONYM_MAP:
                entry = cls.SPELL_AND_SYNONYM_MAP[clean_t]
                if isinstance(entry, dict):
                    search_tokens.append(entry["search"])
                    if clean_t.lower() != entry["display"].lower():
                        corrections.append(entry["display"])
                else:
                    search_tokens.append(entry)
                    if clean_t.lower() != entry.lower():
                        corrections.append(entry)
            # 2. Fuzzy match against canonical domain vocabulary
            elif len(clean_t) >= 4 and clean_t not in {"what", "when", "where", "with", "from", "have", "that", "this"}:
                close = difflib.get_close_matches(clean_t, cls.CANONICAL_DOMAIN_TERMS, n=1, cutoff=0.72)
                if close and close[0].lower() != clean_t.lower():
                    search_tokens.append(close[0])
                    corrections.append(close[0])
                else:
                    search_tokens.append(t)
            else:
                search_tokens.append(t)

        did_you_mean = corrections[0] if corrections else None
        return " ".join(search_tokens), did_you_mean

    @classmethod
    def expand_short_query(cls, query: str) -> str:
        """Expand keyword queries into natural questions for optimal semantic reranking."""
        q_lower = query.lower()
        words = q_lower.split()
        # If user already asked a natural sentence or question, check for specific on-demand verticals
        if any(w in words for w in ["do", "can", "how", "what", "is", "are", "will", "why", "i", "we", "you", "build", "create", "need"]):
            if any(k in q_lower for k in ["food", "driver", "courier", "dispatch"]):
                return f"{query} Can you build on-demand service apps like Zomato or Uber with live tracking and order matching?"
            return query

        if any(w in q_lower for w in ["ecomm", "store", "shop"]):
            return "Do you build ecommerce websites and online stores for products?"
        if any(w in q_lower for w in ["websit", "web"]):
            return "Do you build custom websites and web applications?"
        if any(w in q_lower for w in ["app", "mobil", "andro", "iphon"]):
            return "Do you build mobile apps for iPhone and Android?"
        if any(w in q_lower for w in ["softwar", "custom"]):
            return "Do you build custom software and tools for businesses?"
        if any(w in q_lower for w in ["pric", "cost", "budg"]):
            return "How much does custom software, website or mobile app cost?"
        if any(w in q_lower for w in ["hire", "dev", "staff"]):
            return "Can I hire dedicated developers from your team?"
        if any(w in q_lower for w in ["cloud", "aws", "host"]):
            return "Do you help businesses move to the cloud like AWS or Google Cloud?"
        if any(w in q_lower for w in ["secu", "hack", "cyb"]):
            return "Do you offer cybersecurity and website protection services?"
        if any(w in q_lower for w in ["chat", "bot", "ai", "whatsap"]):
            return "Can you build an AI chatbot for website or WhatsApp?"
        if any(w in q_lower for w in ["seo", "rank"]):
            return "Do you offer SEO services to improve search ranking?"
        if any(w in q_lower for w in ["ad", "market", "social"]):
            return "Do you manage paid ads and digital marketing campaigns?"
        if any(w in q_lower for w in ["logo", "brand"]):
            return "Can you design a logo and build a complete brand identity?"

        return query

    @classmethod
    def process_query(cls, raw_query: str) -> Dict[str, Any]:
        """Analyze, spell-correct, clean, and classify user query."""
        cleaned = re.sub(r"[^\w\s\-\?\$\:\.]", " ", raw_query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # If query starts with greeting followed by question (e.g., 'Hi, can you build...'), strip greeting for search
        search_query = re.sub(r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b[\s\,]*", "", cleaned, flags=re.IGNORECASE).strip()
        if not search_query:
            search_query = cleaned

        # Intelligent spelling & domain synonym correction
        corrected_query, did_you_mean = cls.correct_spelling_and_synonyms(search_query)

        # Semantic natural expansion for short/keyword queries
        rerank_query = cls.expand_short_query(corrected_query)

        words = re.findall(r"\b[a-zA-Z0-9_\-\$]{2,}\b", corrected_query.lower())
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "why", "how",
            "the", "and", "for", "with", "about", "is", "are", "can", "does"
        }
        keywords = [w for w in words if w not in stop_words and not w.isdigit()]

        q_lower = corrected_query.lower()
        detected_intent = "general_factual"
        for intent, triggers in cls.INTENT_MAP.items():
            if any(t in q_lower for t in triggers):
                detected_intent = intent
                break

        return {
            "raw_query": raw_query,
            "cleaned_query": search_query,
            "corrected_query": corrected_query,
            "rerank_query": rerank_query,
            "did_you_mean": did_you_mean,
            "keywords": keywords,
            "intent": detected_intent
        }

class HybridRetriever:
    """
    Combines dense vector search and sparse BM25 search using Reciprocal Rank Fusion (RRF)
    or Weighted Normalized Score Fusion.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: EmbeddingProvider,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.fusion_method = fusion_method.lower()
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedCandidate]:
        """Execute dense vector + sparse BM25 search and fuse candidate results."""
        # 1. Dense Vector Search
        q_vec = self.embedder.embed_query(query)
        v_results = self.vector_store.search(q_vec, top_k=top_k)

        # 2. Sparse BM25 Search
        b_results = self.bm25_index.search(query, top_k=top_k)

        if not v_results and not b_results:
            return []

        fusion_scores: Dict[str, float] = {}
        v_scores: Dict[str, float] = {}
        b_scores: Dict[str, float] = {}
        chunk_info: Dict[str, Dict[str, Any]] = {}

        # Collect chunk info and raw scores
        for rank, res in enumerate(v_results, start=1):
            v_scores[res.chunk_id] = res.score
            chunk_info[res.chunk_id] = {
                "doc_id": res.doc_id,
                "text": res.text,
                "metadata": res.metadata
            }

        for rank, res in enumerate(b_results, start=1):
            b_scores[res.chunk_id] = res.score
            if res.chunk_id not in chunk_info:
                chunk_info[res.chunk_id] = {
                    "doc_id": res.doc_id,
                    "text": res.text,
                    "metadata": res.metadata
                }

        # 3. Apply Score Fusion Strategy
        all_cids = list(chunk_info.keys())

        if self.fusion_method == "weighted":
            # Min-Max Normalization for scores
            max_v = max(v_scores.values()) if v_scores else 1.0
            min_v = min(v_scores.values()) if v_scores else 0.0
            range_v = (max_v - min_v) if (max_v - min_v) > 1e-6 else 1.0

            max_b = max(b_scores.values()) if b_scores else 1.0
            min_b = min(b_scores.values()) if b_scores else 0.0
            range_b = (max_b - min_b) if (max_b - min_b) > 1e-6 else 1.0

            for cid in all_cids:
                norm_v = (v_scores.get(cid, min_v) - min_v) / range_v if cid in v_scores else 0.0
                norm_b = (b_scores.get(cid, min_b) - min_b) / range_b if cid in b_scores else 0.0
                fusion_scores[cid] = (self.vector_weight * norm_v) + (self.bm25_weight * norm_b)
        else:
            # Reciprocal Rank Fusion (RRF)
            for rank, res in enumerate(v_results, start=1):
                fusion_scores[res.chunk_id] = fusion_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            for rank, res in enumerate(b_results, start=1):
                fusion_scores[res.chunk_id] = fusion_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_cids = sorted(fusion_scores.keys(), key=lambda cid: fusion_scores[cid], reverse=True)
        top_k = min(top_k, len(sorted_cids))

        return [
            RetrievedCandidate(
                chunk_id=cid,
                doc_id=chunk_info[cid]["doc_id"],
                text=chunk_info[cid]["text"],
                fusion_score=float(fusion_scores[cid]),
                vector_score=float(v_scores.get(cid, 0.0)),
                bm25_score=float(b_scores.get(cid, 0.0)),
                metadata=chunk_info[cid]["metadata"]
            )
            for cid in sorted_cids[:top_k]
        ]

class CrossEncoderReranker:
    """
    Reranks candidate query-document pairs using deep cross-attention
    or enhanced lexical-coverage scoring with phrase matching.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            try:
                self._model = CrossEncoder(model_name, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(model_name)
            logger.info(f"Loaded CrossEncoder model: {model_name}")
        except Exception:
            self._model = None

    def rerank(self, query: str, candidates: List[RetrievedCandidate], top_n: int = 3) -> List[RetrievedCandidate]:
        """Rerank candidates and select top N chunks."""
        if not candidates:
            return []

        if self._model is not None:
            try:
                pairs = [[query, c.text] for c in candidates]
                scores = self._model.predict(pairs)
                for c, s in zip(candidates, scores):
                    c.rerank_score = float(s)
            except Exception as e:
                logger.warning(f"CrossEncoder inference failed ({e}). Falling back to lexical reranker.")
                self._lexical_rerank(query, candidates)
        else:
            self._lexical_rerank(query, candidates)

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_n]

    @staticmethod
    def _lexical_rerank(query: str, candidates: List[RetrievedCandidate]):
        """
        Enhanced lexical & semantic overlap reranker:
        Combines exact phrase bonus, keyword coverage ratio, and Jaccard similarity.
        """
        q_lower = query.lower()
        q_terms = [t for t in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", q_lower)]
        q_term_set = set(q_terms)

        for c in candidates:
            doc_lower = c.text.lower()
            doc_term_set = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", doc_lower))

            # 1. Term coverage (how many query terms exist in doc)
            overlap = len(q_term_set.intersection(doc_term_set))
            coverage = (overlap / len(q_term_set)) if q_term_set else 0.0

            # 2. Jaccard similarity
            union_len = len(q_term_set.union(doc_term_set))
            jaccard = (overlap / union_len) if union_len > 0 else 0.0

            # 3. Exact phrase match bonus
            phrase_bonus = 0.4 if q_lower in doc_lower else 0.0

            # 4. Partial 2-word phrase matching
            bigram_bonus = 0.0
            if len(q_terms) >= 2:
                for i in range(len(q_terms) - 1):
                    bigram = f"{q_terms[i]} {q_terms[i+1]}"
                    if bigram in doc_lower:
                        bigram_bonus += 0.15

            # Blend with fusion score for stability
            c.rerank_score = round(coverage * 0.45 + jaccard * 0.15 + phrase_bonus + bigram_bonus + (c.fusion_score * 0.25), 4)
