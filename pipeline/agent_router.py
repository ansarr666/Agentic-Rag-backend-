"""Intelligent Agent Router & Decision Layer: Intent Classification and Fallback Routing."""

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from .structured_contract import AgentDecision
from .tools.calculator import CalculatorTool

logger = logging.getLogger(__name__)

_SMALL_TALK_INTENTS = {
    "greeting": {"hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "yo", "sup", "hiya", "howdy",
                 "hi there", "hello there", "hey there",
                 "good morning", "good afternoon", "good evening", "good night",
                 "how are you", "how are you doing", "hi how are you", "hello how are you",
                 "hi there how are you", "hey how are you", "who are you", "what can you do"},
    "thanks": {"thanks", "thank you", "thx", "ty", "thanks a lot", "thanks so much", "thank you so much",
               "much appreciated"},
    "bye": {"bye", "goodbye", "see you"},
    "ack": {"ok", "okay", "k", "cool", "great", "nice"},
    "affirmation": {"yes", "yea", "yep", "yeah", "yup", "sure", "alright", "ok ok", "okok", "fine", "cool cool"},
    "negation": {"no", "nope", "nah", "nop", "no no", "nope nope", "nah nah"},
    "filler": {"maybe", "idk", "i dont know", "i don't know", "dunno", "hmm", "hmmm",
               "um", "uh", "ah", "oh", "ohh", "aha", "whatever"},
}

# Category B safety net: short non-questions without any of these are asked to rephrase
_INTENT_KEYWORDS = (
    "policy", "notice", "salary", "budget", "drive", "calculate", "how", "what", "when", "where",
    "why", "which", "who", "cost", "rate", "period", "leave", "handbook", "compliance", "gst",
    "tax", "percent", "%",
    # public-widget service topics
    "app", "web", "software", "seo", "marketing", "pric", "service", "develop", "mobile",
    "android", "ios", "iphone", "cloud", "security", "commerce", "store", "chatbot", "whatsapp",
    "payment", "server", "custom", "design",
)

# When several small-talk phrases are combined, the most meaningful one picks the reply
_COMBO_PRIORITY = ("bye", "thanks", "greeting", "negation", "affirmation", "ack", "filler")

SMALL_TALK_REPLIES = {
    "greeting": (
        "Hi! I'm OrionSoft AI. I can help you explore our software development, mobile apps, "
        "AI solutions, cloud services, and company information. What would you like to know?"
    ),
    "thanks": "You're welcome! Anything else I can help with?",
    "bye": "Goodbye! Come back anytime you need help.",
    "ack": "Got it. What else can I help you with?",
    "affirmation": "Got it! What would you like to know?",
    "negation": "Understood. Is there something specific I can help with?",
    "filler": (
        "I'm here! Feel free to ask about OrionSoft's services, software development, "
        "mobile apps, AI solutions, or project pricing."
    ),
    "rephrase": (
        "Could you please rephrase that as a complete question or tell me what you're looking for?\n\n"
        "OrionSoft Technologies provides full-cycle digital and software services, including:\n"
        "• **Custom Software & Web Development** (web portals, internal business tools, SaaS)\n"
        "• **Mobile App Development** (native iOS & Android, on-demand platforms, cross-platform apps)\n"
        "• **AI & Generative AI** (smart AI chatbots, process automation, predictive systems)\n"
        "• **Cloud & Cybersecurity** (AWS/Google Cloud migrations, security audits, threat protection)\n"
        "• **Digital Marketing & SEO** (organic search ranking, PPC advertising, brand design)\n"
        "• **Staff Augmentation** (dedicated vetted developers and engineers)\n\n"
        "For example, you could ask:\n"
        "• *'What software development or mobile app services does OrionSoft provide?'*\n"
        "• *'Can you build custom AI solutions or chatbots for my website?'*\n"
        "• *'Do you work with international clients outside India?'*\n"
        "• *'How much does it cost to build a custom application or website?'*\n"
        "• *'Can I hire dedicated developers from your team?'*"
    ),
}

def detect_small_talk(query: str) -> Optional[str]:
    """Return the small-talk intent for exact-match fillers, or 'rephrase' for short keyword-less non-questions."""
    q = query.strip().lower().rstrip("!.,?").strip()
    for intent, phrases in _SMALL_TALK_INTENTS.items():
        if q in phrases:
            return intent

    # Combined small talk ("okay thanks", "ok, bye"): every word must belong to a known phrase
    tokens = re.sub(r"[!.,?]+", " ", q).split()
    found, i = [], 0
    while i < len(tokens):
        for n in range(min(5, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i:i + n])
            intent = next((k for k, v in _SMALL_TALK_INTENTS.items() if phrase in v), None)
            if intent:
                found.append(intent)
                i += n
                break
        else:
            found = []
            break
    if found:
        return min(found, key=_COMBO_PRIORITY.index)

    words = q.split()
    keywords = _INTENT_KEYWORDS + tuple(AgentRouter.CONFIDENTIAL_INTERNAL_TERMS) + tuple(AgentRouter.PUBLIC_EXTERNAL_TRIGGERS)
    if (0 < len(words) <= 3
            and not query.strip().endswith("?")
            and not any(ch.isdigit() for ch in q)
            and not any(k in q for k in keywords)):
        return "rephrase"
    return None

@dataclass
class RoutingDecision:
    """Agent router outcome determining the optimal source and tool execution flow."""
    decision: AgentDecision
    primary_tool: str
    is_internal_confidential: bool
    web_search_allowed: bool
    requires_calculator: bool
    query_intent: str
    reasoning: str

class AgentRouter:
    """
    Agentic Decision Layer routing user questions based on intent,
    policy boundaries, and observable evidence requirements.
    """

    # Internal confidential concepts that must NEVER trigger external web search
    CONFIDENTIAL_INTERNAL_TERMS = [
        "policy", "probation", "leave", "notice period", "handbook", "salary",
        "compensation", "stipend", "reimbursement", "orionsoft", "company policy",
        "client list", "internal tool", "confidential", "hr rule", "director",
        "contract", "employment", "nda", "appraisal"
    ]

    # External / Public tech triggers where Web Search is authorized
    PUBLIC_EXTERNAL_TRIGGERS = [
        "latest", "release", "current version", "lts", "python version",
        "node version", "react 19", "public docs", "documentation for",
        "framework", "external api", "market trends", "news"
    ]

    @classmethod
    def route_initial(cls, query: str) -> RoutingDecision:
        """
        Classify query and make initial routing determination
        before expensive model or retrieval calls.
        """
        q_lower = query.lower().strip()

        # 1. Deterministic Calculator Check (e.g., "What is 17% of ₹184,500?", "cloud bill reduced by 17%")
        if CalculatorTool.can_handle(query):
            return RoutingDecision(
                decision=AgentDecision.CALCULATOR,
                primary_tool="calculator",
                is_internal_confidential=False,
                web_search_allowed=False,
                requires_calculator=True,
                query_intent="deterministic_calculation",
                reasoning="Identified mathematical or financial percentage operation suitable for safe AST calculator."
            )

        # 2. Check if query is confidential internal company policy
        is_confidential = any(term in q_lower for term in cls.CONFIDENTIAL_INTERNAL_TERMS)

        # 3. Check if query explicitly concerns external public technology
        is_public_tech = any(trigger in q_lower for trigger in cls.PUBLIC_EXTERNAL_TRIGGERS)

        if is_public_tech and not is_confidential:
            # Query about public libraries, latest releases, external tools
            return RoutingDecision(
                decision=AgentDecision.WEB_SEARCH,
                primary_tool="web_search",
                is_internal_confidential=False,
                web_search_allowed=True,
                requires_calculator=False,
                query_intent="external_public_knowledge",
                reasoning="Query requests current public technology information not maintained in internal company knowledge."
            )

        # Default: Route to Internal Knowledge Search first
        return RoutingDecision(
            decision=AgentDecision.INTERNAL_RAG,
            primary_tool="knowledge_search",
            is_internal_confidential=is_confidential,
            web_search_allowed=not is_confidential,
            requires_calculator=False,
            query_intent="company_policy" if is_confidential else "company_services",
            reasoning="Routed to Internal RAG hybrid index to check authoritative company documents."
        )

    @classmethod
    def evaluate_fallback(
        cls,
        initial_route: RoutingDecision,
        evidence_state: str,
        query: str,
        google_drive_available: bool = True
    ) -> Tuple[AgentDecision, str]:
        """
        Determine appropriate fallback when internal evidence is insufficient or partial:
        ├── SUFFICIENT -> internal_rag
        └── INSUFFICIENT / PARTIAL
             ├── Google Drive Search (if connected and authorized)
             ├── Web Search (if question is externally answerable AND NOT confidential)
             └── Abstain / Clarification (if confidential or unanswerable)
        """
        if evidence_state == "sufficient":
            return AgentDecision.INTERNAL_RAG, "Sufficient verified evidence found in internal company documents."

        if evidence_state == "conflicting":
            return AgentDecision.INTERNAL_RAG, "Conflicting evidence detected across retrieved company documents."

        # Evidence is INSUFFICIENT or PARTIAL
        # Check 1: Can Google Drive answer?
        if google_drive_available:
            return AgentDecision.GOOGLE_DRIVE, "Internal local index insufficient; querying connected Google Drive sources."

        # Check 2: If company confidential, Web Search is strictly prohibited!
        if initial_route.is_internal_confidential:
            return AgentDecision.ABSTAIN, "Internal company knowledge contains insufficient information; external web search is prohibited for internal company policies."

        # Check 3: If externally answerable and allowed, route to Web Search
        if initial_route.web_search_allowed:
            return AgentDecision.WEB_SEARCH, "Insufficient internal evidence; routing to authoritative web search fallback."

        # Final fallback: Abstain
        return AgentDecision.ABSTAIN, "The available company knowledge does not contain enough information to answer this."
