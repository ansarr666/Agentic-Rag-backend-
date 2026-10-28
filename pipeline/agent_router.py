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
               "much appreciated", "okay thanks", "ok thanks", "ok thank you", "okay thank you"},
    "bye": {"bye", "goodbye", "see you", "see ya", "cya", "bye bye", "byebye", "have a good day", "have a great day", "take care"},
    "ack": {"ok", "okay", "k", "cool", "great", "nice", "got it", "understood"},
    "affirmation": {"yes", "yea", "yep", "yeah", "yup", "sure", "alright", "ok ok", "okok", "fine", "cool cool"},
    "negation": {
        "no", "nope", "nah", "nop", "no no", "nope nope", "nah nah", "no thanks", "no thank you",
        "no thx", "no ty", "no need", "i said no", "i told you no", "already said no", "said no",
        "nothing", "nothing else", "nothing more", "nothing right now", "not really",
        "i am good", "im good", "i'm good", "all good", "we are good", "we're good",
        "that's all", "thats all", "that is all", "that's it", "thats it", "that is it",
        "dont need anything", "don't need anything", "not now", "not right now",
        "leave it", "never mind", "nevermind", "no more"
    },
    "filler": {"maybe", "idk", "i dont know", "i don't know", "dunno", "hmm", "hmmm",
               "um", "uh", "ah", "oh", "ohh", "aha", "whatever"},
    "booking": {
        "book a call", "book call", "booking a call", "schedule a call", "schedule call",
        "schedule a discovery call", "book a discovery call", "discovery call",
        "book consultation", "schedule consultation", "book a consultation", "schedule a meeting",
        "book meeting", "book a meeting", "i want to book a call", "can i book a call",
        "can we book a call", "can we schedule a call", "how to book a call",
        "how do i book a call", "setup a call", "set up a call", "talk to sales",
        "talk to an expert", "talk to expert", "contact sales", "call me", "request a call",
        "schedule demo", "book demo", "book a demo", "call", "book", "schedule"
    },
}

# Category B safety net: short non-questions without any of these are asked to rephrase
_INTENT_KEYWORDS = (
    "policy", "notice", "salary", "budget", "drive", "calculate", "how", "what", "when", "where",
    "why", "which", "who", "cost", "rate", "period", "leave", "handbook", "compliance", "gst",
    "tax", "percent", "%",
    # public-widget service topics
    "app", "web", "software", "seo", "marketing", "pric", "service", "develop", "mobile",
    "android", "ios", "iphone", "cloud", "security", "commerce", "store", "chatbot",
    "payment", "server", "custom", "design",
    # enterprise & specialized software systems (case-insensitive)
    "crm", "erp", "saas", "api", "pos", "cms", "lms", "mvp", "portal", "dashboard",
    "database", "backend", "frontend", "fullstack", "devops", "automation", "build", "create", "make", "need", "want",
    # booking and consultation topics
    "book", "call", "schedule", "meeting", "consult", "consultation", "demo", "expert", "talk", "hire", "quote", "phone",
)

# When several small-talk phrases are combined, the most meaningful one picks the reply
_COMBO_PRIORITY = ("booking", "bye", "thanks", "greeting", "negation", "affirmation", "ack", "filler")

SMALL_TALK_REPLIES = {
    "greeting": (
        "Hi! I'm OrionSoft AI. I can help you explore our software development, mobile apps, "
        "AI solutions, cloud services, and company information. What would you like to know?"
    ),
    "thanks": "You're welcome! Let me know if you need anything else, or feel free to reach out whenever you're ready.",
    "bye": "Goodbye! Feel free to reach out anytime if you need help with a software or digital project. Have a wonderful day!",
    "ack": "Got it. Feel free to ask if there's anything you'd like to discuss.",
    "affirmation": "Got it! What would you like to know?",
    "negation": "No problem! Feel free to reach out anytime if you have questions or want to discuss a project in the future. Have a great day!",
    "filler": (
        "I'm here! Feel free to ask about OrionSoft's services, software development, "
        "mobile apps, AI solutions, or project pricing whenever you're ready."
    ),
    "booking": (
        "We would love to connect with you on a call! You can book a free discovery call directly using "
        "the 'Book a Call' option, or simply share your phone number and preferred time right here. "
        "Our solutions engineering team will get in touch to discuss your project."
    ),
    "rephrase": (
        "Could you please share a bit more detail about what you're looking for? "
        "I can help with custom software development, mobile apps, AI solutions, web development, cloud services, and digital marketing."
    ),
}

def detect_small_talk(query: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """Return the small-talk intent for exact-match fillers, closure phrases, or 'rephrase' for short keyword-less non-questions."""
    # Strip quotes, punctuation, and extra whitespace
    clean_q = re.sub(r"[\'\"`!.,?]+", " ", query).strip().lower()
    q = re.sub(r"\s+", " ", clean_q).strip()
    if not q:
        return None

    # 0. High-priority booking and call requests (handles typos like 'ook me call', 'book me call', 'give me a call', etc.)
    booking_patterns = [
        r"\b(?:b?ook|schedule|arrange|set\s*up|request|want|need|give|get)\s+(?:me\s+)?(?:a\s+)?(?:discovery\s+|consultation\s+|phone\s+|sales\s+)?(?:call|meeting|appointment|demo)\b",
        r"\b(?:b?ook|schedule|set\s*up)\s+(?:me\s+)?(?:call|meeting)\b",
        r"\b(?:call|phone)\s+(?:me|us|back)(?:\s+please|\s+now)?\b",
        r"\b(?:can|could|would)\s+(?:you|someone|we)\s+(?:call|reach|contact)\s+(?:me|us)\b",
        r"\b(?:talk|speak)\s+(?:to|with)\s+(?:a\s+|an\s+)?(?:human|expert|agent|sales|consultant|team|someone)\b",
        r"\b(?:i\s+want\s+to|i\s+would\s+like\s+to|can\s+i|how\s+do\s+i)\s+(?:talk|call|speak|meet|connect)\b",
        r"\b(?:discovery\s+call|schedule\s+call|book\s+call|meeting\s+request|consultation\s+call)\b",
    ]
    for bp in booking_patterns:
        if re.search(bp, q):
            return "booking"

    # 1. Exact match in intents
    for intent, phrases in _SMALL_TALK_INTENTS.items():
        if q in phrases:
            return intent

    # 2. Closure and negation pattern match (e.g. "i said no", "i told you no", "no thanks", "nothing else", "all good")
    closure_patterns = [
        r"\b(?:i\s+said|i\s+told\s+you|already\s+said|i\s+say)\s+no\b",
        r"\bno\s+(?:thanks|thank\s+you|thx|ty|need|more|problem)\b",
        r"\b(?:nothing|not\s+much)\s*(?:else|more|right\s+now)?\b",
        r"\b(?:that\s*s|thats|that\s+is)\s+(?:all|it)\b",
        r"\b(?:i\s*m|im|i\s+am|we\s*re|were|we\s+are)\s+(?:all\s+)?good\b",
        r"\b(?:not|no)\s+(?:really|now|right\s+now|today)\b",
    ]
    for cp in closure_patterns:
        if re.search(cp, q):
            return "negation"

    # 3. Contextual dialogue check: if assistant asked if they need anything else and user declined
    if history:
        last_bot = next((t.get("content", "").lower() for t in reversed(history) if t.get("role") in ("assistant", "bot")), "")
        if any(h in last_bot for h in ["anything else", "what else", "help with", "something specific"]):
            if q in {"no", "nope", "nah", "nothing", "all good", "i said no", "no need", "we are good"}:
                return "negation"

    # 4. Combined small talk ("okay thanks", "ok, bye"): every word must belong to a known phrase
    tokens = q.split()
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

    # 5. Short keyword-less non-questions safety net
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
