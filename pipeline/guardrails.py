"""Guardrails and Scope Enforcement Layer for OrionSoft Conversational AI.

Enforces business boundaries, detects prompt injection / role hijacking,
filters out-of-scope requests (coding help, creative writing, trivia),
and manages conversational flow controls (repetition, ambiguity, restart/resume).
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GuardrailAction(str, Enum):
    PASS = "pass"
    REFUSE_OFF_TOPIC = "refuse_off_topic"
    REFUSE_PROMPT_INJECTION = "refuse_prompt_injection"
    HANDLE_RESTART = "handle_restart"
    HANDLE_RESUME = "handle_resume"
    HANDLE_AMBIGUOUS = "handle_ambiguous"
    HANDLE_REPEATED = "handle_repeated"


@dataclass
class GuardrailResult:
    action: GuardrailAction
    is_safe: bool
    category: str
    response: Optional[str] = None
    suggested_questions: Optional[List[str]] = None
    reason: Optional[str] = None


# ── Prompt Injection & Role Hijack Patterns ───────────────────
_INJECTION_PATTERNS = [
    r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|rules|guidelines)\b",
    r"\b(?:show|reveal|print|display|tell\s+me|output|dump)\s+(?:your\s+)?(?:system\s+prompt|initial\s+prompt|instructions|system\s+instructions|secret\s+key)\b",
    r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(?:an?\s+)?(?:dan|jailbreak|unrestricted|uncensored|evil|hacker|a\s+terminal|terminal|bash|linux\s+console)\b",
    r"\b(?:act\s+as|roleplay\s+as|pretend\s+to\s+be)\b.*?\b(?:terminal|hacker|dan|jailbreak|unrestricted|no\s+filters|no\s+safety)\b",
    r"\b(?:developer\s+mode|god\s+mode|jailbreak\s+mode)\s+(?:enable|on|activate)\b",
    r"\b(?:forget\s+you\s+are|you\s+are\s+no\s+longer)\s+orionsoft\b",
    r"\b(?:bypass|disable|turn\s+off)\s+(?:all\s+)?(?:safety|guardrails|filters|restrictions)\b",
    r"\bwhat\s+(?:is|was)\s+your\s+(?:system\s+prompt|exact\s+instruction)\b",
]

# ── Off-topic: Direct Coding & Programming Tasks ─────────────
# Distinguish asking OrionSoft to build software from asking the bot to write code snippets / solve homework
_CODING_REQUEST_PATTERNS = [
    r"\b(?:write|generate|give\s+me|create|show\s+me)\s+(?:a\s+)?(?:python|javascript|typescript|c\+\+|java|c#|rust|golang|php|ruby|sql|bash|html|css)\s+(?:code|script|program|snippet|function|class|query)\b",
    r"\b(?:write|code|create|print)\s+(?:a\s+)?(?:python\s+)?[\"']?hello[\s,-]*world[\"']?\b",
    r"\b(?:debug|fix|find)\s+(?:(?:this|the|a)\s+)?(?:bug\s+in\s+)?(?:my\s+|this\s+)?(?:[a-zA-Z]+\s+)?(?:code|function|script|error|traceback|program)\b",
    r"\b(?:typeerror|syntaxerror|referenceerror|valueerror|nullpointerexception|undefined\s+is\s+not\s+a\s+function)\b",
    r"\b(?:solve|implement)\s+(?:this\s+)?(?:leetcode|binary\s+search|fibonacci|two\s+sum|sorting\s+algorithm|regex)\b",
    r"\b(?:write|implement|create)\s+(?:a\s+)?(?:function|algorithm|method|class|code)\s+to\s+.*?(?:reverse|sort|search|traverse|linked\s+list|binary\s+tree|palindrome|fibonacci)\b",
    r"\b(?:write\s+a\s+sql\s+query|select\s+\*\s+from|insert\s+into\s+\w+|create\s+table\s+\w+)\b",
    r"\bhow\s+to\s+(?:code|program|write\s+a\s+function\s+to)\b",
]

# ── Off-topic: Creative Writing & Entertainment ──────────────
_CREATIVE_WRITING_PATTERNS = [
    r"\b(?:write|compose|generate)\s+(?:a\s+)?(?:poem|poetry|rhyme|haiku|sonnet|limerick)\b",
    r"\b(?:write|tell\s+me)\s+(?:a\s+)?(?:bedtime\s+story|fairy\s+tale|fictional\s+story|novel|fanfic)\b",
    r"\b(?:tell\s+me\s+a\s+joke|make\s+me\s+laugh|write\s+a\s+pun)\b",
    r"\b(?:write|compose)\s+(?:a\s+)?(?:song|lyrics|rap|movie\s+script)\b",
]

# ── Off-topic: General Knowledge & Trivia ────────────────────
_TRIVIA_PATTERNS = [
    r"\b(?:what\s+is\s+the\s+capital\s+of|capital\s+city\s+of)\s+[a-zA-Z\s]+\b",
    r"\brecipe\s+for\b",
    r"\bhow\s+to\s+(?:bake|cook|make)\s+(?:a\s+)?.*?(?:pizza|cake|cookies|pasta|curry|bread|pie)\b",
    r"\b(?:who\s+won\s+the\s+(?:fifa|world\s+cup|super\s+bowl|olympics|ipl))\b",
    r"\b(?:who\s+is\s+the\s+president\s+of|prime\s+minister\s+of)\b",
    r"\b(?:what\s+is\s+the\s+distance\s+between|distance\s+from\s+earth\s+to)\b",
    r"\b(?:explain\s+quantum\s+physics|theory\s+of\s+relativity|schrodinger)\b",
    r"\btranslate\s+.*?\b(?:to|into)\s+(?:french|spanish|german|chinese|japanese|russian)\b",
]

# ── Conversational Flow: Ambiguous Answers ───────────────────
_AMBIGUOUS_PHRASES = {
    "maybe", "not sure", "depends", "it depends", "could be", "perhaps",
    "i dont know", "i don't know", "dunno", "idk", "possibly", "hard to say",
    "not decided", "unsure", "i'm not sure", "im not sure"
}

# ── Conversational Flow: Restart & Resume ────────────────────
_RESTART_PATTERNS = [
    r"^(?:restart|start\s+over|reset|reset\s+chat|clear\s+chat|begin\s+again|new\s+conversation|start\s+fresh|clear\s+history)$"
]

_RESUME_PATTERNS = [
    r"^(?:resume|continue|pick\s+up(?:\s+where\s+we\s+left\s+off)?|go\s+on|proceed|back\s+to\s+where\s+we\s+were)$"
]

# ── Standard Safe Response Templates ─────────────────────────
REFUSAL_PROMPT_INJECTION = (
    "I am OrionSoft AI, and my role is strictly to assist with OrionSoft Technologies' "
    "custom software development, mobile apps, AI solutions, cloud services, and digital consulting. "
    "I cannot alter my instructions, assume a different persona, or reveal system prompts. "
    "How can I assist you with your software or digital project today?"
)

REFUSAL_CODING_ASSISTANCE = (
    "I'm OrionSoft AI, specialized in helping you discover OrionSoft's engineering capabilities, "
    "custom software development, mobile apps, and enterprise AI solutions. "
    "I don't provide general code generation or programming homework assistance. "
    "If you're looking to build a custom application, integrate an AI system, or hire dedicated engineers, "
    "I'd be glad to discuss how our development team can help!"
)

REFUSAL_CREATIVE_WRITING = (
    "I'm here as OrionSoft's digital solutions consultant to discuss software engineering, "
    "mobile app development, AI solutions, and digital transformation services. "
    "I'm unable to help with creative writing, stories, or poetry. "
    "Feel free to ask about our development services, tech stack, or booking a discovery call!"
)

REFUSAL_TRIVIA = (
    "I'm OrionSoft AI, focused on software development, mobile apps, AI integrations, "
    "and digital consulting for OrionSoft Technologies. "
    "I don't answer general trivia, recipes, or unrelated general knowledge questions. "
    "Can I help you with any software or app development requirements?"
)

REPLY_AMBIGUOUS = (
    "I understand there are various factors to evaluate. "
    "To help you find the best solution, could you share a bit more about your project goals, "
    "or would you like to schedule a free discovery call so our engineering consultants can review your options?"
)

REPLY_RESTART = (
    "Let's start fresh! I'm OrionSoft AI, ready to assist you with custom software, "
    "mobile app development, AI chatbots, cloud architecture, and marketing services. "
    "What would you like to explore?"
)

REPLY_RESUME = (
    "Welcome back! We can pick up right where we left off. "
    "What would you like to discuss next regarding your project requirements or our services?"
)

REPLY_REPEATED = (
    "It looks like you've asked about this previously. "
    "To discuss your specific requirements in greater depth than our chatbot overview, "
    "would you like to schedule a quick, free discovery call with one of our software consultants?"
)

STANDARD_ORIONSOFT_SUGGESTIONS = [
    "What software development services do you offer?",
    "Can you build custom mobile apps for iOS and Android?",
    "How much does custom software development cost?",
    "How do I book a free discovery call?",
]


class GuardrailEvaluator:
    """Evaluates user input against safety, prompt injection, and business scope boundaries."""

    @classmethod
    def evaluate(
        cls,
        user_question: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> GuardrailResult:
        """
        Run multi-layer guardrail inspection on the incoming message.
        Returns a GuardrailResult with action, safety flag, and standardized response if blocked.
        """
        raw_text = (user_question or "").strip()
        q_lower = raw_text.lower()

        if not raw_text:
            return GuardrailResult(
                action=GuardrailAction.HANDLE_AMBIGUOUS,
                is_safe=True,
                category="empty",
                response="Please enter a message to begin our conversation.",
                suggested_questions=STANDARD_ORIONSOFT_SUGGESTIONS
            )

        # 1. Prompt Injection & Role Hijack Gate (Highest Priority)
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, q_lower):
                logger.warning(f"Guardrail triggered: Prompt injection attempt detected: '{raw_text[:80]}'")
                return GuardrailResult(
                    action=GuardrailAction.REFUSE_PROMPT_INJECTION,
                    is_safe=False,
                    category="prompt_injection",
                    response=REFUSAL_PROMPT_INJECTION,
                    suggested_questions=STANDARD_ORIONSOFT_SUGGESTIONS,
                    reason="Matched prompt injection / role bypass pattern"
                )

        # 2. Conversation Flow: Restart / Reset
        for pattern in _RESTART_PATTERNS:
            if re.match(pattern, q_lower.strip("!.,? ")):
                return GuardrailResult(
                    action=GuardrailAction.HANDLE_RESTART,
                    is_safe=True,
                    category="conversation_flow_restart",
                    response=REPLY_RESTART,
                    suggested_questions=STANDARD_ORIONSOFT_SUGGESTIONS,
                    reason="User requested conversation restart"
                )

        # 3. Conversation Flow: Resume / Continue
        for pattern in _RESUME_PATTERNS:
            if re.match(pattern, q_lower.strip("!.,? ")):
                return GuardrailResult(
                    action=GuardrailAction.HANDLE_RESUME,
                    is_safe=True,
                    category="conversation_flow_resume",
                    response=REPLY_RESUME,
                    suggested_questions=STANDARD_ORIONSOFT_SUGGESTIONS,
                    reason="User requested conversation resumption"
                )

        # 4. Conversation Flow: Ambiguous Answers (e.g. "maybe", "not sure", "depends")
        clean_ambiguous = q_lower.rstrip("!.,? ").strip()
        if clean_ambiguous in _AMBIGUOUS_PHRASES:
            return GuardrailResult(
                action=GuardrailAction.HANDLE_AMBIGUOUS,
                is_safe=True,
                category="conversation_flow_ambiguous",
                response=REPLY_AMBIGUOUS,
                suggested_questions=[
                    "What software development services do you offer?",
                    "How does OrionSoft estimate project costs?",
                    "Can we schedule a discovery call?",
                    "What is your typical project timeline?"
                ],
                reason="User provided an ambiguous or uncertain response"
            )

        # 5. Conversation Flow: Repeated Question Check
        if history and len(history) >= 2:
            repeated = cls._check_repeated_query(q_lower, history)
            if repeated:
                logger.info(f"Guardrail triggered: Repeated question detected: '{raw_text[:80]}'")
                return GuardrailResult(
                    action=GuardrailAction.HANDLE_REPEATED,
                    is_safe=True,
                    category="conversation_flow_repeated",
                    response=REPLY_REPEATED,
                    suggested_questions=[
                        "How do I schedule a discovery call?",
                        "What information should I prepare for the call?",
                        "Can you explain your pricing model?",
                        "What is your development process?"
                    ],
                    reason="User asked the same question multiple times"
                )

        # 6. Off-Topic: Direct Coding / Programming Request
        for pattern in _CODING_REQUEST_PATTERNS:
            if re.search(pattern, q_lower):
                # Ensure we don't block legitimate inquiries like "Do you write Python software?"
                if not cls._is_legitimate_software_service_inquiry(q_lower):
                    logger.info(f"Guardrail triggered: Off-topic coding request: '{raw_text[:80]}'")
                    return GuardrailResult(
                        action=GuardrailAction.REFUSE_OFF_TOPIC,
                        is_safe=False,
                        category="off_topic_coding",
                        response=REFUSAL_CODING_ASSISTANCE,
                        suggested_questions=[
                            "What programming languages and frameworks do you build with?",
                            "Can you build a custom web app or mobile app for my business?",
                            "Can I hire dedicated developers from OrionSoft?",
                            "How do I get a project estimate?"
                        ],
                        reason="Direct code generation or debugging request is out of business scope"
                    )

        # 7. Off-Topic: Creative Writing & Entertainment
        for pattern in _CREATIVE_WRITING_PATTERNS:
            if re.search(pattern, q_lower):
                logger.info(f"Guardrail triggered: Creative writing request: '{raw_text[:80]}'")
                return GuardrailResult(
                    action=GuardrailAction.REFUSE_OFF_TOPIC,
                    is_safe=False,
                    category="off_topic_creative",
                    response=REFUSAL_CREATIVE_WRITING,
                    suggested_questions=STANDARD_ORIONSOFT_SUGGESTIONS,
                    reason="Creative writing is outside business scope"
                )

        # 8. Off-Topic: General Trivia & Unrelated Knowledge
        for pattern in _TRIVIA_PATTERNS:
            if re.search(pattern, q_lower):
                logger.info(f"Guardrail triggered: General trivia request: '{raw_text[:80]}'")
                return GuardrailResult(
                    action=GuardrailAction.REFUSE_OFF_TOPIC,
                    is_safe=False,
                    category="off_topic_trivia",
                    response=REFUSAL_TRIVIA,
                    suggested_questions=STANDARD_ORIONSOFT_SUGGESTIONS,
                    reason="General trivia is outside business scope"
                )

        # 9. All guardrails passed
        return GuardrailResult(
            action=GuardrailAction.PASS,
            is_safe=True,
            category="in_scope"
        )

    @classmethod
    def _is_legitimate_software_service_inquiry(cls, q_lower: str) -> bool:
        """
        Verify if a query mentioning programming languages is asking about OrionSoft's
        services, capabilities, or experience (which is IN SCOPE) rather than asking the bot
        to write code snippets or solve programming problems (which is OUT OF SCOPE).
        """
        service_indicators = [
            "do you", "can you", "does orionsoft", "do you guys", "are you able to",
            "team", "hire", "experience with", "work with", "services", "offer",
            "develop for", "cost to build", "project with", "build an app in", "build software in"
        ]
        # If it asks "Do you build apps in Python?" or "Can you develop software using Python?", that's IN SCOPE
        if any(ind in q_lower for ind in service_indicators):
            if not any(act in q_lower for act in ["write a code", "write the code", "write a python hello", "solve this", "debug this", "write a function"]):
                return True
        return False

    @classmethod
    def _check_repeated_query(cls, current_query: str, history: List[Dict[str, str]]) -> bool:
        """Check if user has asked the exact same or very similar question in the recent turns."""
        q_norm = re.sub(r"[^\w\s]", "", current_query).strip()
        if len(q_norm) < 4:
            return False

        user_messages = [
            re.sub(r"[^\w\s]", "", turn.get("content", "").lower()).strip()
            for turn in history
            if turn.get("role") == "user"
        ]

        # Count how many times identical/near-identical message appeared
        match_count = 0
        for msg in user_messages[-4:]:
            if not msg:
                continue
            if msg == q_norm:
                match_count += 1
            elif len(msg) > 10 and len(q_norm) > 10:
                # Jaccard word overlap
                w1 = set(msg.split())
                w2 = set(q_norm.split())
                overlap = len(w1 & w2) / max(len(w1 | w2), 1)
                if overlap >= 0.85:
                    match_count += 1

        return match_count >= 2
