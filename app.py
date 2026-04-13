import os
import re
import json
import base64
import uuid
import asyncio
import logging
import hashlib
from io import BytesIO
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse

import httpx
from supabase import create_client
import time

# =========================
# CONFIG & LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HeloXAi")

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip() if os.getenv("GROQ_API_KEY") else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
LOGO_URL = os.getenv("LOGO_URL", "https://heloxai.xyz/logo.png")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heloxai.xyz"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Client
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Global State for Stream Cancellation
active_streams: Dict[str, asyncio.Task] = {}

# =========================
# BASE SYSTEM PROMPT - NO CREATOR MENTION
# =========================
BASE_SYSTEM_PROMPT = """You are HeloxAi, a helpful and capable AI assistant. Be accurate, friendly, and concise. Help users with whatever they ask."""

# =========================
# CREATOR RESPONSE - ONLY USED WHEN ASKED
# =========================
CREATOR_RESPONSE_INSTRUCTION = """IMPORTANT: The user is asking about your creator/developer. You MUST respond with exactly this information:
"I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi"

Do not add extra details. Do not mention any other companies or people. This is the only correct answer to this question."""

# Patterns that indicate user is asking about the creator
CREATOR_QUESTION_PATTERNS = [
    r'\b(who|whom)\b.*\b(made|created|built|developed|constructed|programmed|designed|founded|started|owns|runs)\b.*\b(you|this|helox|heloxai)\b',
    r'\b(who|whom)\b.*\b(is|are)\b.*\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b',
    r'\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b.*\b(is|are|who)\b',
    r'\bwho\b.*\bbehind\b.*\b(you|this|helox)\b',
    r'\bwho.*made.*you\b',
    r'\bwho.*created.*you\b',
    r'\bwho.*built.*you\b',
    r'\bwho.*developed.*you\b',
    r'\bwho.*programmed.*you\b',
    r'\bwho.*constructed.*you\b',
    r'\bwho.*designed.*you\b',
    r'\bwho.*owns.*you\b',
    r'\bwho.*runs.*you\b',
    r'\byour\s+creator\b',
    r'\byour\s+developer\b',
    r'\byour\s+maker\b',
    r'\byour\s+builder\b',
    r'\byour\s+founder\b',
    r'\byour\s+owner\b',
    r'\bwho\s+is\s+behind\s+helox\b',
    r'\bwho\s+made\s+helox\b',
    r'\bwho\s+created\s+helox\b',
    r'\bwho\s+built\s+helox\b',
    r'\bwho\s+developed\s+helox\b',
    r'\bmade\s+by\s+who\b',
    r'\bcreated\s+by\s+who\b',
    r'\bbuilt\s+by\s+who\b',
    r'\bdeveloped\s+by\s+who\b',
    r'\bconstructed\s+by\s+who\b',
    r'\btell\s+me\s+about\s+your\s+(creator|developer|maker|builder|founder)\b',
    r'\bwhat\s+company\s+made\s+you\b',
    r'\bwhat\s+team\s+made\s+you\b',
    r'\bwhere\s+do\s+you\s+come\s+from\b',
    r'\bhow\s+were\s+you\s+(made|created|built|developed|born)\b',
    r'\bare\s+you\s+made\s+by\b',
    r'\bdid\s+.*\s+make\s+you\b',
    r'\bdid\s+.*\s+create\s+you\b',
    r'\bdid\s+.*\s+build\s+you\b',
]

# Pre-compile creator patterns for performance
COMPILED_CREATOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CREATOR_QUESTION_PATTERNS]


def is_creator_question(text: str) -> bool:
    """Check if user is asking about who created/made the AI"""
    for pattern in COMPILED_CREATOR_PATTERNS:
        if pattern.search(text):
            return True
    return False


def get_system_prompt(user_prompt: str) -> str:
    """Return base prompt normally, creator response ONLY if asked"""
    if is_creator_question(user_prompt):
        return BASE_SYSTEM_PROMPT + "\n\n" + CREATOR_RESPONSE_INSTRUCTION
    return BASE_SYSTEM_PROMPT


# =========================
# ADVANCED USER RECOGNITION SYSTEM
# =========================
PRIMARY_COOKIE = "HeloxAi_Session"
FINGERPRINT_COOKIE = "HeloxAi_FP"
BACKUP_COOKIE = "HeloxAi_ID"
DEVICE_COOKIE = "HeloxAi_Dev"

# Extended cookie settings for strong persistence
COOKIE_SETTINGS = {
    "max_age": 86400 * 365,  # 1 year
    "httponly": True,
    "secure": True,
    "samesite": "none",
    "path": "/"
}

def generate_device_fingerprint(request: Request) -> str:
    """Generate a fingerprint from request headers"""
    fp_components = [
        request.headers.get("user-agent", ""),
        request.headers.get("accept-language", ""),
        request.headers.get("accept-encoding", ""),
        request.client.host if request.client else "",
    ]
    fp_string = "|".join(fp_components)
    return hashlib.sha256(fp_string.encode()).hexdigest()[:32]

def set_all_cookies(response: Response, user_id: str, fingerprint: str):
    """Set all cookies for maximum persistence"""
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **COOKIE_SETTINGS)
    response.set_cookie(key=FINGERPRINT_COOKIE, value=fingerprint, **COOKIE_SETTINGS)
    response.set_cookie(key=BACKUP_COOKIE, value=user_id, **COOKIE_SETTINGS)
    response.set_cookie(key=DEVICE_COOKIE, value=f"{fingerprint}_{user_id[:8]}", **COOKIE_SETTINGS)


# =========================
# ADVANCED INTENT DETECTION
# =========================
class IntentCategory(Enum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    DOCUMENT_CREATION = "document_creation"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"
    WEB_DEVELOPMENT = "web_development"
    API_DEVELOPMENT = "api_development"
    DATABASE = "database"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    CONVERSATION = "conversation"


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    sub_intents: List[IntentCategory]
    keywords_matched: List[str]
    patterns_matched: List[str]

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "sub_intents": [i.value for i in self.sub_intents],
            "keywords_matched": self.keywords_matched,
            "patterns_matched": self.patterns_matched
        }


class AdvancedIntentDetector:
    def __init__(self):
        self._compile_patterns()
        self._init_synonyms()
        self.negation_words = {
            "don't", "dont", "do not", "doesn't", "doesnt", "does not",
            "didn't", "didnt", "did not", "never", "no", "not", "without",
            "skip", "avoid", "except", "but not", "ignore", "rather than"
        }

    def _compile_patterns(self):
        self.patterns = {
            IntentCategory.IMAGE_GENERATION: [
                r'\b(generate|create|make|draw|render|paint|sketch|illustrate)\s+(a\s+|an\s+)?(image|picture|photo|drawing|illustration|artwork|painting|sketch|graphic|visual)',
                r'\b(image|picture|photo|drawing|illustration)\s+(of|showing|depicting|with|for|about)',
                r'\b(text\s+to\s+image|txt2img|img2img)',
                r'\b(visualize|visualise)\s+(this|that|the|it)',
                r'\b(dall[eé]|midjourney|stable\s+diffusion|sd\s*xl|flux)',
                r'\b(generate|create)\s+(some\s+)?art',
                r'\b(make\s+(me\s+)?(a\s+)?(visual|graphic|thumbnail|logo|icon|banner|poster))',
                r'\b(prompt\s+(for|to))\s+(generate|create|make)',
            ],
            IntentCategory.VIDEO_GENERATION: [
                r'\b(generate|create|make|produce)\s+(a\s+)?(video|clip|movie|animation|motion\s+graphic)',
                r'\b(text\s+to\s+video|txt2vid|video\s+generation)',
                r'\b(animate|animation)\s+(this|that|the|image|picture)',
                r'\b(video|clip|movie)\s+(of|showing|about|with)',
                r'\b(runway|pika|sora|mov2mov|kling)',
                r'\b(turn|convert)\s+(this|the|image)\s+(into|to)\s+(a\s+)?(video|animation)',
            ],
            IntentCategory.AUDIO_GENERATION: [
                r'\b(generate|create|make|produce)\s+(a\s+)?(audio|sound|music|speech|voice|song|track|beat)',
                r'\b(text\s+to\s+speech|tts|speech\s+to\s+text|stt)',
                r'\b(music|song|beat|melody)\s+(generation|creation|for|about)',
                r'\b(elevenlabs|suno|udio|bark)',
                r'\b(clone|replicate)\s+(a\s+)?voice',
            ],
            IntentCategory.CODE_GENERATION: [
                r'\b(write|create|generate|build|code|develop|implement)\s+(a\s+)?(\w+\s+)?(function|class|module|script|program|code|snippet|app|application|component)',
                r'\b(how\s+(to|can\s+i)\s+(write|create|implement|code|build))',
                r'\b(code\s+(for|that|this|to|which|example))',
                r'\b(convert\s+(this|to)\s+(code|python|javascript|java|c\+\+|rust|go|typescript))',
                r'\b(scaffold|boilerplate|template)\s+(for|a)',
                r'\b(wrapper|helper|utility)\s+(function|class|module)\s+(for|to)',
                r'\b(implement\s+(the|a|this)\s+(\w+\s+)?(pattern|algorithm|logic|feature))',
            ],
            IntentCategory.CODE_REVIEW: [
                r'\b(review|analyze|critique|evaluate|audit)\s+(this|my|the)\s+(code|function|class|script|implementation|pr)',
                r'\b(is\s+(this|there)\s+(code|anything)\s+(good|bad|wrong|improvable|clean))',
                r'\b(best\s+practices?\s+(for|in)\s+(this|my)\s+(code|implementation))',
                r'\b(refactor|improve|optimize|clean\s+up)\s+(this|my|the)\s+(code|function|class)',
                r'\b(code\s+quality|technical\s+debt|code\s+smell)',
            ],
            IntentCategory.CODE_DEBUG: [
                r'\b(fix|debug|solve|troubleshoot|resolve)\s+(this|my|the|a)\s+(bug|error|issue|problem)',
                r'\b(why\s+(is|does|are|do)\s+(this|my|the|it)\s+(not\s+working|failing|breaking|erroring|returning))',
                r'\b(error|exception|traceback|stack\s+trace|segfault)\s*[:\n]',
                r'\b(what(\'s|\s+is)\s+(wrong|the\s+problem)\s+(with|in))',
                r'\b(won\'t\s+work|doesn\'t\s+work|not\s+working|broken|failing)',
                r'\b(unexpected|wrong|incorrect)\s+(result|output|behavior|value)',
                r'\b(help\s+(me\s+)?)?debug',
            ],
            IntentCategory.DOCUMENT_CREATION: [
                r'\b(create|write|generate|draft|compose)\s+(a\s+)?(document|pdf|report|letter|email|memo|article|essay|paper|proposal|whitepaper)',
                r'\b(document|report|proposal|specification)\s+(for|about|on|regarding)',
                r'\b(format\s+(as|this\s+as|it\s+as)\s+(a\s+)?(pdf|document|report|letter|markdown))',
                r'\b(professional|formal|business)\s+(document|letter|email|report)',
            ],
            IntentCategory.DATA_ANALYSIS: [
                r'\b(analyze|analysis|analyse)\s+(this|the|my|some)\s+(data|dataset|csv|excel|spreadsheet|json)',
                r'\b(statistics?|statistical)\s+(analysis|test|summary|overview)',
                r'\b(insights?\s+(from|in|about|into))',
                r'\b(correlation|regression|distribution|trend)\s+(analysis|of|in)',
                r'\b(clean|preprocess|prepare|wrangle)\s+(this|the)\s+(data|dataset)',
                r'\b(eda|exploratory\s+data\s+analysis)',
            ],
            IntentCategory.DATA_VISUALIZATION: [
                r'\b(create|make|generate|plot|chart|graph|visualize)\s+(a\s+)?(chart|graph|plot|visualization|diagram|dashboard)',
                r'\b(bar\s+chart|line\s+graph|scatter\s+plot|pie\s+chart|histogram|heatmap|box\s+plot|violin\s+plot)',
                r'\b(visualize|visualise|plot|chart|graph)\s+(this|the|these|those|data)',
                r'\b(matplotlib|seaborn|plotly|d3|chart\.js|ggplot|altair)',
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                r'\b(create|build|develop|make)\s+(a\s+)?(website|web\s*page|web\s*app|landing\s+page|web\s*site|portfolio)',
                r'\b(html|css|javascript|typescript|react|vue|angular|next\.js|nuxt|svelte|tailwind)\b',
                r'\b(frontend|front[- ]end|back[- ]end|full[- ]stack)\s*(development|for|with|app)?',
                r'\b(responsive|mobile[- ]friendly|mobile[- ]first)\s*(design|website|layout)?',
                r'\b(component|page|layout|template)\s+(for|in)\s+(react|vue|angular|next)',
            ],
            IntentCategory.API_DEVELOPMENT: [
                r'\b(create|build|develop|design|implement)\s+(a\s+)?(api|rest\s*api|graphql\s*api|endpoint|route)',
                r'\b(api\s*(endpoint|route|handler|controller|gateway))',
                r'\b(restful|rest|graphql|grpc|websocket)\s*(api|service|endpoint)?',
                r'\b(openapi|swagger|api\s*documentation)',
                r'\b(request|response|payload)\s+(format|structure|schema)',
            ],
            IntentCategory.DATABASE: [
                r'\b(create|write|design)\s+(a\s+)?(database|schema|table|query|sql|migration)',
                r'\b(sql|mysql|postgres|postgresql|mongodb|redis|dynamodb|sqlite)\s*(query|statement|command)?',
                r'\b(schema\s*(design|migration|definition|update))',
                r'\b(orm|sequelize|prisma|sqlalchemy|typeorm|drizzle)\s*(query|model|schema)?',
                r'\b(crud\s*(operation|operations|endpoint|api))',
                r'\b(select|insert|update|delete)\s+(from|into|table)',
            ],
            IntentCategory.TRANSLATION: [
                r'\b(translate|translation)\s+(this|to|into|from)\s+(\w+)',
                r'\b(in|to|into)\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|urdu)',
                r'\b(how\s+(do\s+you|to)\s+say\s+.+\s+in\s+\w+)',
                r'\b(native|localize|localization|l10n|i18n|internationaliz)',
            ],
            IntentCategory.SUMMARIZATION: [
                r'\b(summarize|summary|summarise|tldr|tl;dr)\s+(this|the|it|that|for\s+me)',
                r'\b(brief|short|concise)\s+(overview|summary|explanation|version)\s*(of|for|about)?',
                r'\b(key\s+(points|takeaways|highlights))\s*(from|of|in)?',
                r'\b(main\s+(idea|points|theme|argument|concept))',
                r'\b(give\s+me\s+(the\s+)?(gist|bottom\s+line|essence))',
            ],
            IntentCategory.EXPLANATION: [
                r'\b(explain|explanation)\s+(to\s+me\s+)?',
                r'\b(what\s+(is|are|was|were|does|do|means|mean))\s+',
                r'\b(how\s+(does|do|did|can|would|should|to))\s+',
                r'\b(tell\s+me\s+(about|more\s+about|how|why))',
                r'\b(why\s+(is|does|do|are|did|can|would))\s+',
                r'\b(definition|meaning)\s+(of|for)\s+',
                r'\b(understand(ing)?)\s*(this|how|why|what|better)?',
                r'\b(break\s+down|simplify|elaborate)\s+',
            ],
            IntentCategory.CREATIVE_WRITING: [
                r'\b(write|create|compose)\s+(a\s+)?(story|poem|poetry|novel|chapter|verse|lyrics|song|haiku|limerick)',
                r'\b(creative|fiction|fantasy|sci[- ]?fi|horror|romance|thriller|mystery)\s*(writing|story|tale)?',
                r'\b(narrative|plot|character|setting|dialogue)\s*(for|development|creation|arc)?',
                r'\b(storytelling|story[- ]?telling)',
                r'\b(write\s+(like|in\s+the\s+style\s+of))\s+',
            ],
            IntentCategory.MATHEMATICAL: [
                r'\b(calculate|compute|solve|evaluate)\s+(this|the|a)\s*(equation|expression|formula|problem|integral|derivative)?',
                r'\b(math|mathematics|algebra|calculus|geometry|statistics|probability|linear\s+algebra)\s*(problem|equation|question)?',
                r'\b(\d+[\.\d]*\s*[\+\-\*\/\^%\=]\s*[\.\d]*)',
                r'\b(integral|derivative|differentiat|integrat)\s*(of|the)?',
                r'\b(prove|proof)\s+(that|this|the)',
                r'\b(formula|equation)\s+(for|to\s+calculate|to\s+find)',
            ],
            IntentCategory.RESEARCH: [
                r'\b(research|find|search|look\s+up|investigate)\s+(about|on|for|into)',
                r'\b(stud(y|ies))\s+(show|suggest|indicate|demonstrate|prove)',
                r'\b(academic|scholarly|peer[- ]?reviewed)\s*(source|paper|article|research|journal)?',
                r'\b(cite|citation|reference|bibliography)\s+',
                r'\b(literature\s+review)\s*(on|for|of)?',
                r'\b(what\s+(does\s+)?(research|science|literature)\s+say)',
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?]*$',
                r'^(thank|thanks|thank\s+you|appreciate)[\s!.?]*$',
                r'^(how\s+are\s+you|how(\'s|\s+is)\s+it\s+going|what(\'s|\s+is)\s+up)[\s!.?]*$',
                r'^(bye|goodbye|see\s+you|farewell)[\s!.?]*$',
                r'^(sure|okay|ok|got\s+it|understood)[\s!.?]*$',
            ],
        }

        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.patterns.items()
        }

    def _init_synonyms(self):
        self.synonyms = {
            IntentCategory.IMAGE_GENERATION: [
                "image", "picture", "photo", "photograph", "drawing", "illustration",
                "artwork", "painting", "sketch", "graphic", "visual", "render",
                "thumbnail", "logo", "icon", "banner", "poster", "infographic",
                "dalle", "midjourney", "stable diffusion", "ai art", "generated art",
                "portrait", "landscape", "composition", "digital art"
            ],
            IntentCategory.VIDEO_GENERATION: [
                "video", "clip", "movie", "film", "animation", "motion",
                "gif", "moving image", "video clip", "short video", "reel",
                "runway", "pika", "sora", "animated", "motion graphic"
            ],
            IntentCategory.AUDIO_GENERATION: [
                "audio", "sound", "music", "speech", "voice", "song", "track",
                "beat", "melody", "tune", "podcast", "narration", "voiceover",
                "tts", "text to speech", "elevenlabs", "suno", "udio"
            ],
            IntentCategory.CODE_GENERATION: [
                "code", "script", "function", "class", "module", "program",
                "app", "application", "software", "snippet", "implementation",
                "algorithm", "routine", "procedure", "macro", "plugin", "extension",
                "library", "package", "utility", "helper"
            ],
            IntentCategory.CODE_REVIEW: [
                "review", "refactor", "improve", "optimize", "clean up",
                "best practice", "code quality", "code smell", "technical debt",
                "maintainability", "readability"
            ],
            IntentCategory.CODE_DEBUG: [
                "bug", "error", "issue", "problem", "debug", "fix", "troubleshoot",
                "exception", "crash", "fault", "defect", "glitch", "broken",
                "typo", "mistake", "wrong", "incorrect"
            ],
            IntentCategory.DOCUMENT_CREATION: [
                "document", "pdf", "report", "letter", "email", "memo", "article",
                "essay", "paper", "proposal", "whitepaper", "manual", "guide",
                "handbook", "documentation", "specification", "brief"
            ],
            IntentCategory.DATA_ANALYSIS: [
                "data", "dataset", "csv", "excel", "spreadsheet", "analytics",
                "statistics", "insights", "metrics", "kpi", "analysis"
            ],
            IntentCategory.DATA_VISUALIZATION: [
                "chart", "graph", "plot", "visualization", "diagram", "dashboard",
                "histogram", "scatter", "heatmap", "bar chart", "line graph",
                "pie chart", "infographic", "plotly", "matplotlib"
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                "website", "webpage", "web app", "landing page", "frontend",
                "backend", "fullstack", "full stack", "html", "css", "react",
                "vue", "angular", "next.js", "svelte", "tailwind"
            ],
            IntentCategory.API_DEVELOPMENT: [
                "api", "rest api", "graphql", "endpoint", "route", "restful",
                "swagger", "openapi", "microservice"
            ],
            IntentCategory.DATABASE: [
                "database", "schema", "table", "sql", "query", "migration",
                "mysql", "postgres", "mongodb", "redis", "sqlite", "prisma",
                "sequelize", "sqlalchemy", "orm", "crud"
            ],
            IntentCategory.TRANSLATION: [
                "translate", "translation", "localize", "localization",
                "i18n", "l10n", "multilingual"
            ],
            IntentCategory.SUMMARIZATION: [
                "summarize", "summary", "summarise", "tldr", "tl;dr",
                "brief", "overview", "key points", "takeaways", "gist"
            ],
            IntentCategory.EXPLANATION: [
                "explain", "explanation", "what is", "how does", "why",
                "understand", "elaborate", "simplify", "break down"
            ],
            IntentCategory.CREATIVE_WRITING: [
                "story", "poem", "poetry", "novel", "fiction", "creative",
                "narrative", "lyrics", "haiku", "limerick", "storytelling"
            ],
            IntentCategory.MATHEMATICAL: [
                "calculate", "compute", "solve", "math", "equation",
                "formula", "integral", "derivative", "proof", "algebra",
                "calculus", "geometry", "statistics", "probability"
            ],
            IntentCategory.RESEARCH: [
                "research", "find", "search", "investigate", "study",
                "academic", "scholarly", "citation", "reference", "literature"
            ],
        }

    def _has_negation(self, text: str, keyword_pos: int) -> bool:
        words_before = text[:keyword_pos].lower().split()[-6:]
        preceding_text = " ".join(words_before)
        return any(neg in preceding_text for neg in self.negation_words)

    def _calculate_confidence(
            self,
            matched_keywords: List[str],
            matched_patterns: List[str],
            text_length: int
    ) -> float:
        if not matched_keywords and not matched_patterns:
            return 0.0

        pattern_confidence = min(len(matched_patterns) * 0.35, 0.65)
        keyword_confidence = min(len(matched_keywords) * 0.12, 0.25)
        multi_signal_bonus = 0.1 if (matched_keywords and matched_patterns) else 0.0
        length_factor = max(0.5, 1.0 - (text_length / 1500) * 0.4)

        confidence = (pattern_confidence + keyword_confidence + multi_signal_bonus) * length_factor
        return min(confidence, 1.0)

    def _are_related_intents(self, intent1: IntentCategory, intent2: IntentCategory) -> bool:
        related_groups = [
            {IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG},
            {IntentCategory.DATA_ANALYSIS, IntentCategory.DATA_VISUALIZATION},
            {IntentCategory.IMAGE_GENERATION, IntentCategory.VIDEO_GENERATION, IntentCategory.AUDIO_GENERATION},
            {IntentCategory.WEB_DEVELOPMENT, IntentCategory.API_DEVELOPMENT, IntentCategory.DATABASE},
            {IntentCategory.DOCUMENT_CREATION, IntentCategory.RESEARCH},
            {IntentCategory.EXPLANATION, IntentCategory.SUMMARIZATION},
        ]
        for group in related_groups:
            if intent1 in group and intent2 in group:
                return True
        return False

    def detect_intents(self, text: str, threshold: float = 0.25) -> List[IntentResult]:
        text_lower = text.lower()
        results = []

        for intent, compiled_patterns in self.compiled_patterns.items():
            matched_keywords = []
            matched_patterns = []

            for pattern in compiled_patterns:
                if pattern.search(text):
                    matched_patterns.append(pattern.pattern)

            if intent in self.synonyms:
                for synonym in self.synonyms[intent]:
                    if synonym in text_lower:
                        pos = text_lower.find(synonym)
                        if not self._has_negation(text, pos):
                            matched_keywords.append(synonym)

            if matched_keywords or matched_patterns:
                confidence = self._calculate_confidence(
                    matched_keywords, matched_patterns, len(text)
                )
                if confidence >= threshold:
                    results.append(IntentResult(
                        intent=intent,
                        confidence=confidence,
                        sub_intents=[],
                        keywords_matched=matched_keywords,
                        patterns_matched=matched_patterns
                    ))

        results.sort(key=lambda x: x.confidence, reverse=True)

        if results:
            primary = results[0]
            for result in results[1:]:
                if self._are_related_intents(primary.intent, result.intent):
                    primary.sub_intents.append(result.intent)

        return results[:1] if results else []

    def get_primary_intent(self, text: str) -> Optional[IntentResult]:
        results = self.detect_intents(text)
        return results[0] if results else None

    def get_action_type(self, text: str) -> str:
        intent = self.get_primary_intent(text)
        if not intent:
            return "general"

        action_map = {
            IntentCategory.IMAGE_GENERATION: "image",
            IntentCategory.VIDEO_GENERATION: "video",
            IntentCategory.AUDIO_GENERATION: "audio",
            IntentCategory.CODE_GENERATION: "code",
            IntentCategory.CODE_REVIEW: "code",
            IntentCategory.CODE_DEBUG: "code",
            IntentCategory.DOCUMENT_CREATION: "document",
            IntentCategory.DATA_ANALYSIS: "data",
            IntentCategory.DATA_VISUALIZATION: "data",
            IntentCategory.WEB_DEVELOPMENT: "web",
            IntentCategory.API_DEVELOPMENT: "api",
            IntentCategory.DATABASE: "database",
            IntentCategory.TRANSLATION: "translation",
            IntentCategory.SUMMARIZATION: "summary",
            IntentCategory.EXPLANATION: "explanation",
            IntentCategory.CREATIVE_WRITING: "creative",
            IntentCategory.MATHEMATICAL: "math",
            IntentCategory.RESEARCH: "research",
            IntentCategory.CONVERSATION: "conversation",
        }
        return action_map.get(intent.intent, "general")

    def get_required_tools(self, text: str) -> List[str]:
        intent = self.get_primary_intent(text)
        if not intent:
            return ["llm"]

        tool_map = {
            IntentCategory.IMAGE_GENERATION: ["image_gen", "llm"],
            IntentCategory.VIDEO_GENERATION: ["video_gen", "llm"],
            IntentCategory.AUDIO_GENERATION: ["audio_gen", "llm"],
            IntentCategory.CODE_GENERATION: ["code_exec", "llm"],
            IntentCategory.CODE_REVIEW: ["llm"],
            IntentCategory.CODE_DEBUG: ["code_exec", "llm"],
            IntentCategory.DOCUMENT_CREATION: ["doc_gen", "llm"],
            IntentCategory.DATA_ANALYSIS: ["code_exec", "data_processing", "llm"],
            IntentCategory.DATA_VISUALIZATION: ["code_exec", "llm"],
            IntentCategory.WEB_DEVELOPMENT: ["code_exec", "llm"],
            IntentCategory.API_DEVELOPMENT: ["code_exec", "llm"],
            IntentCategory.DATABASE: ["database", "code_exec", "llm"],
            IntentCategory.TRANSLATION: ["llm"],
            IntentCategory.SUMMARIZATION: ["llm"],
            IntentCategory.EXPLANATION: ["llm"],
            IntentCategory.CREATIVE_WRITING: ["llm"],
            IntentCategory.MATHEMATICAL: ["code_exec", "llm"],
            IntentCategory.RESEARCH: ["web_search", "llm"],
            IntentCategory.CONVERSATION: ["llm"],
        }

        tools = list(tool_map.get(intent.intent, ["llm"]))

        for sub_intent in intent.sub_intents:
            for tool in tool_map.get(sub_intent, []):
                if tool not in tools:
                    tools.append(tool)

        return tools

    def get_code_system_prompt(self, text: str) -> str:
        base = get_system_prompt(text)
        
        intent = self.get_primary_intent(text)
        if not intent:
            return base + "\n\nYou are also a helpful coding assistant."

        sub_prompts = {
            IntentCategory.CODE_DEBUG: """

You are also an expert debugger. When analyzing code issues:
1. Identify the root cause of the bug/error
2. Explain WHY it's happening (not just what)
3. Provide the exact fix with clear code blocks
4. Suggest how to prevent similar issues
Be precise and practical.""",

            IntentCategory.CODE_REVIEW: """

You are also a senior code reviewer. Provide constructive feedback on:
1. Code quality and readability
2. Potential bugs or edge cases
3. Performance considerations
4. Best practices and design patterns
5. Security concerns
Be specific and actionable in your suggestions.""",

            IntentCategory.CODE_GENERATION: """

You are also an expert software engineer. When writing code:
1. Write clean, well-structured, production-ready code
2. Include appropriate error handling
3. Add helpful comments for complex logic
4. Consider edge cases
5. Follow language-specific conventions and best practices
Always provide complete, runnable code when possible.""",

            IntentCategory.WEB_DEVELOPMENT: """

You are also a full-stack web developer expert. When building web components:
1. Use modern best practices and frameworks
2. Ensure responsive design
3. Consider accessibility (a11y)
4. Include proper styling
5. Make components reusable and maintainable
Provide complete, ready-to-use code.""",

            IntentCategory.API_DEVELOPMENT: """

You are also an API development expert. When creating APIs:
1. Follow RESTful principles (or GraphQL best practices)
2. Include proper error handling and status codes
3. Add input validation
4. Consider security (auth, rate limiting)
5. Document endpoints clearly
Provide complete, production-ready code.""",

            IntentCategory.DATABASE: """

You are also a database expert. When working with databases:
1. Design efficient, normalized schemas
2. Write optimized queries
3. Include proper indexes
4. Consider data integrity with constraints
5. Follow SQL best practices
Provide complete, ready-to-execute SQL/ORM code.""",
        }

        return base + sub_prompts.get(intent.intent, "\n\nYou are also a helpful coding assistant.")


# Singleton instance
_detector = None


def get_detector() -> AdvancedIntentDetector:
    global _detector
    if _detector is None:
        _detector = AdvancedIntentDetector()
    return _detector


# =========================
# BACKWARD COMPATIBLE FUNCTIONS
# =========================
def is_image_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "image"


def is_video_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "video"


def is_code_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "code"


def is_document_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "document"


def is_data_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == "data"


# =========================
# NEW ADVANCED FUNCTIONS
# =========================
def detect_intent(prompt: str) -> Optional[IntentResult]:
    return get_detector().get_primary_intent(prompt)


def get_action_type(prompt: str) -> str:
    return get_detector().get_action_type(prompt)


def get_required_tools(prompt: str) -> List[str]:
    return get_detector().get_required_tools(prompt)


def is_debug_request(prompt: str) -> bool:
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_DEBUG


def is_review_request(prompt: str) -> bool:
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_REVIEW


def get_intent_confidence(prompt: str) -> float:
    intent = detect_intent(prompt)
    return intent.confidence if intent else 0.0


# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True


class RegenerateRequest(BaseModel):
    conversation_id: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


class IntentInfo(BaseModel):
    intent: str
    confidence: float
    sub_intents: List[str]
    action_type: str
    tools: List[str]


# =========================
# HELPERS
# =========================
def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _execute_supabase_with_retry(query_builder, description="Supabase Operation"):
    max_retries = 3
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(query_builder.execute)
        except Exception as e:
            last_exception = e
            error_str = str(e)
            if "502" in error_str or "Bad Gateway" in error_str or "Expecting value" in error_str:
                logger.warning(f"{description} encountered transient error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
            else:
                logger.error(f"{description} failed: {e}")
                break
    
    if last_exception:
        raise last_exception


async def get_user(request: Request, response: Response) -> Dict[str, Any]:
    """
    Enhanced user recognition with multi-cookie strategy and device fingerprinting.
    Priority: Primary Cookie > Backup Cookie > Device Fingerprint Match > Create New
    """
    primary_id = request.cookies.get(PRIMARY_COOKIE)
    backup_id = request.cookies.get(BACKUP_COOKIE)
    device_cookie = request.cookies.get(DEVICE_COOKIE)
    stored_fingerprint = request.cookies.get(FINGERPRINT_COOKIE)
    
    current_fingerprint = generate_device_fingerprint(request)
    
    user_obj = {
        "id": None,
        "email": None,
        "memory": "",
        "fingerprint": current_fingerprint
    }

    user_id = None
    if primary_id:
        user_id = primary_id
    elif backup_id:
        user_id = backup_id
    
    if not user_id and device_cookie:
        try:
            fp_part = device_cookie.split("_")[0] if "_" in device_cookie else device_cookie
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", fp_part).limit(1),
                description="User Lookup by Fingerprint"
            )
            if fp_resp.data:
                user_id = fp_resp.data[0]["id"]
                logger.info(f"User recovered via fingerprint match: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Fingerprint lookup failed: {e}")

    if not user_id and stored_fingerprint:
        try:
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", stored_fingerprint).limit(1),
                description="User Lookup by Stored Fingerprint"
            )
            if fp_resp.data:
                user_id = fp_resp.data[0]["id"]
                logger.info(f"User recovered via stored fingerprint: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Stored fingerprint lookup failed: {e}")

    if user_id:
        try:
            user_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("*").eq("id", user_id).limit(1),
                description="User Lookup by ID"
            )
            if user_resp.data:
                u = user_resp.data[0]
                user_obj = {
                    "id": u["id"],
                    "email": u.get("email"),
                    "memory": u.get("memory", ""),
                    "is_premium": u.get("is_premium", False),
                    "is_lifetime": u.get("is_lifetime", False),
                    "plan": u.get("plan", "free"),
                    "fingerprint": current_fingerprint
                }
                if u.get("fingerprint") != current_fingerprint:
                    try:
                        await _execute_supabase_with_retry(
                            supabase.table("users").update({"fingerprint": current_fingerprint}).eq("id", user_id),
                            description="Update Fingerprint"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update fingerprint: {e}")
                
                set_all_cookies(response, user_id, current_fingerprint)
                return user_obj
        except Exception as e:
            logger.error(f"User data fetch failed: {e}")

    new_id = str(uuid.uuid4())
    
    try:
        new_user_data = {
            "id": new_id,
            "email": f"anon+{new_id[:8]}@local",
            "memory": "",
            "fingerprint": current_fingerprint
        }
        
        await _execute_supabase_with_retry(
            supabase.table("users").insert(new_user_data),
            description="Create Anonymous User"
        )
        
        user_obj["id"] = new_id
        
    except Exception as e:
        logger.error(f"Failed to create anonymous user: {e}")
        user_obj["id"] = new_id

    set_all_cookies(response, new_id, current_fingerprint)
    
    logger.info(f"New user created: {new_id[:8]}... with fingerprint {current_fingerprint[:8]}...")
    
    return user_obj


def get_groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}


def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


# =========================
# VIDEO WATERMARK SYSTEM
# =========================
async def fetch_logo_image() -> Optional[bytes]:
    """Fetch the logo.png from the configured URL"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(LOGO_URL)
            if response.status_code == 200:
                return response.content
            logger.error(f"Failed to fetch logo: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Logo fetch error: {e}")
    return None


async def add_watermark_to_video(video_url: str) -> str:
    """
    Add transparent watermark (logo.png) to bottom-right of video.
    Returns the URL of the watermarked video.
    """
    try:
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
        import tempfile
        import os
        
        logo_bytes = await fetch_logo_image()
        if not logo_bytes:
            logger.warning("No logo available, returning unwatermarked video")
            return video_url
        
        async with httpx.AsyncClient(timeout=120) as client:
            video_response = await client.get(video_url)
            if video_response.status_code != 200:
                logger.error(f"Failed to download video: HTTP {video_response.status_code}")
                return video_url
            video_bytes = video_response.content
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            logo_path = os.path.join(tmpdir, "logo.png")
            output_path = os.path.join(tmpdir, "output.mp4")
            
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            with open(logo_path, "wb") as f:
                f.write(logo_bytes)
            
            def process_video():
                video = VideoFileClip(video_path)
                
                logo_width = int(video.w * 0.15)
                logo = ImageClip(logo_path)
                logo_aspect = logo.h / logo.w
                logo_height = int(logo_width * logo_aspect)
                logo = logo.resize((logo_width, logo_height))
                
                padding = 20
                logo = logo.set_position((video.w - logo_width - padding, video.h - logo_height - padding))
                logo = logo.set_duration(video.duration)
                logo = logo.set_opacity(0.7)
                
                final = CompositeVideoClip([video, logo])
                
                final.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=os.path.join(tmpdir, "temp_audio.m4a"),
                    remove_temp=True,
                    logger=None
                )
                
                video.close()
                logo.close()
                final.close()
                
                return output_path
            
            output_path = await asyncio.to_thread(process_video)
            
            with open(output_path, "rb") as f:
                watermarked_bytes = f.read()
            
            filename = f"watermarked_{uuid.uuid4().hex}.mp4"
            path = f"public/videos/{filename}"
            
            try:
                await asyncio.to_thread(
                    lambda: supabase.storage.from_("ai-videos").upload(
                        path, watermarked_bytes, {"content-type": "video/mp4"}
                    )
                )
                watermarked_url = f"{SUPABASE_URL}/storage/v1/object/public/ai-videos/{path}"
                logger.info(f"Watermarked video uploaded: {watermarked_url}")
                return watermarked_url
            except Exception as upload_err:
                logger.warning(f"Storage upload failed, using data URI: {upload_err}")
                b64_video = base64.b64encode(watermarked_bytes).decode()
                return f"data:video/mp4;base64,{b64_video}"
                
    except ImportError:
        logger.error("moviepy not installed. Install with: pip install moviepy")
        return video_url
    except Exception as e:
        logger.error(f"Watermark error: {e}")
        return video_url


# =========================
# CORE LOGIC
# =========================
async def save_message(user_id: str, conv_id: str, role: str, content: str):
    data = {
        "id": str(uuid.uuid4()),
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await _execute_supabase_with_retry(
        supabase.table("messages").insert(data),
        description="Save Message"
    )


async def universal_text_extractor(content: bytes, filename: str) -> str:
    try:
        if filename.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(BytesIO(content))
            return "\n".join([p.extract_text() or "" for p in reader.pages])

        if filename.endswith((
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cpp", ".c", ".cs",
            ".go", ".rs", ".php", ".rb", ".swift"
        )):
            return content.decode("utf-8", errors="ignore")

        if filename.endswith((".json", ".csv", ".xml", ".yaml", ".yml")):
            return content.decode("utf-8", errors="ignore")

        if filename.endswith((".txt", ".md", ".log")):
            return content.decode("utf-8", errors="ignore")

        if filename.endswith(".html"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text()

        return content.decode("utf-8", errors="ignore")

    except Exception as e:
        logger.error(f"[EXTRACT FAIL] {e}")
        raise HTTPException(500, "Failed to extract file content")


async def handle_text_analysis(text: str, stream: bool, user_prompt: str = ""):
    text = text[:15000]

    messages = [
        {
            "role": "system",
            "content": get_system_prompt(user_prompt) + """

You also analyze files. Detect type automatically and respond accordingly:

- Code → explain, find bugs, suggest improvements
- PDF/docs → summarize + key insights
- Data → extract patterns
- Logs → find errors/issues

Be structured and clear."""
        },
        {
            "role": "user",
            "content": text
        }
    ]

    if stream:
        async def gen():
            task = asyncio.current_task()
            try:
                async for token in stream_groq_chat(messages):
                    if task.cancelled(): break
                    yield sse({"type": "token", "text": token})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Text analysis stream error: {e}")
                yield sse({"type": "error", "message": "Analysis failed."})

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages
            }
        )
        r.raise_for_status()

    return {"analysis": r.json()["choices"][0]["message"]["content"]}


async def handle_image_analysis(image_bytes: bytes, stream: bool, user_prompt: str = ""):
    b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }]
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=get_openai_headers(),
            json=payload
        )
        r.raise_for_status()

    result = r.json()["choices"][0]["message"]["content"]

    if stream:
        async def gen():
            task = asyncio.current_task()
            try:
                yield sse({"type": "text", "text": result})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Image analysis stream error: {e}")
                yield sse({"type": "error", "message": "Analysis failed."})

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {"analysis": result}


async def get_history(conv_id: str, limit: int = 10):
    res = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("role, content")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=False)
        .limit(limit),
        description="Get History"
    )
    return [{"role": m["role"], "content": m["content"]} for m in (res.data or [])]


async def stream_groq_chat(messages: list, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1024):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens}
            ) as resp:
                
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    logger.error(f"Groq API Error {resp.status_code}: {error_text}")
                    raise Exception(f"AI Service Error ({resp.status_code})")

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except:
                            pass
    except httpx.ConnectError:
        logger.error("Failed to connect to Groq API.")
        raise Exception("Connection to AI service failed.")
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        raise e


async def handle_code_assistant(prompt: str, user: Dict[str, Any], conv_id: str, stream: bool):
    system_prompt = get_detector().get_code_system_prompt(prompt)
    
    user_memory = user.get("memory", "")
    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    intent_result = detect_intent(prompt)
    logger.info(
        f"[CODE] sub_intent={intent_result.intent.value if intent_result else 'none'} "
        f"confidence={(intent_result.confidence if intent_result else 0):.2%}"
    )

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})

                if conv_id:
                    try:
                        await save_message(user["id"], conv_id, "assistant", full_text)
                    except Exception as e:
                        logger.error(f"Failed to save assistant message: {e}")
                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Streaming Error: {e}")
                yield sse({"type": "error", "message": "An error occurred processing your request."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    if conv_id:
        await save_message(user["id"], conv_id, "assistant", reply)
    return {"reply": reply}


# LAZY LOADING FOR VISION
vision_model = None


def get_vision_model():
    global vision_model
    if vision_model is None:
        from ultralytics import YOLO
        import torch
        logger.info("Loading YOLO model...")
        vision_model = YOLO("yolov8n.pt")
        if torch.cuda.is_available():
            vision_model.to("cuda")
    return vision_model


# =========================
# ENDPOINTS
# =========================
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    return Response(status_code=200)

@app.get("/robots.txt")
def robots():
    return PlainTextResponse("User-agent: *\nDisallow:")

@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    content_type = req.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await req.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON")

    elif "multipart/form-data" in content_type:
        form = await req.form()
        body = dict(form)

        if "file" in form:
            file: UploadFile = form["file"]
            content = await file.read()

            logger.warning("File sent to /ask/universal instead of /analysis")

            if file.content_type and file.content_type.startswith("image/"):
                return await handle_image_analysis(content, stream=True)

            text = await universal_text_extractor(content, file.filename)
            return await handle_text_analysis(text, stream=True)
    else:
        raise HTTPException(415, f"Unsupported content-type: {content_type}")

    prompt = body.get("prompt", "")
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)

    if not prompt:
        raise HTTPException(400, "Prompt required")

    user = await get_user(req, res)

    conversation_exists = False
    
    if conv_id:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations").select("id").eq("id", conv_id).limit(1),
            description="Check Conversation Existence"
        )
        if check.data:
            conversation_exists = True
        else:
            logger.warning(f"Conversation {conv_id} not found in DB. Creating new conversation.")
            conv_id = str(uuid.uuid4())

    if not conv_id:
        conv_id = str(uuid.uuid4())

    if not conversation_exists:
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert({
                "id": conv_id, 
                "user_id": user["id"],
                "title": prompt[:30] if prompt else "New Chat", 
                "created_at": datetime.now(timezone.utc).isoformat()
            }),
            description="Create Conversation"
        )

    try:
        await save_message(user["id"], conv_id, "user", prompt)
    except Exception as e:
        logger.error(f"Failed to save user message: {e}")

    intent_result = detect_intent(prompt)
    action_type = get_action_type(prompt)
    required_tools = get_required_tools(prompt)

    if intent_result:
        logger.info(
            f"[INTENT] action={action_type} "
            f"intent={intent_result.intent.value} "
            f"confidence={intent_result.confidence:.2%} "
            f"sub_intents={[i.value for i in intent_result.sub_intents]} "
            f"tools={required_tools}"
        )
    else:
        logger.info(f"[INTENT] action=general (no specific intent)")

    handler_map = {
        "image": handle_image_generation,
        "video": handle_video_generation,
        "code": handle_code_assistant,
        "document": handle_text_analysis,
        "data": handle_text_analysis,
        "web": handle_code_assistant,
        "api": handle_code_assistant,
        "database": handle_code_assistant,
    }

    handler = handler_map.get(action_type)

    if handler:
        if action_type in ("document", "data"):
            return await handler(prompt, stream, user_prompt=prompt)
        else:
            return await handler(prompt, user, conv_id, stream)

    if action_type == "math":
        return await handle_math_request(prompt, user, conv_id, stream)

    if action_type == "research":
        return await handle_research_request(prompt, user, conv_id, stream)

    if action_type == "creative":
        return await handle_creative_request(prompt, user, conv_id, stream)

    if action_type == "translation":
        return await handle_translation_request(prompt, user, conv_id, stream)

    if action_type == "summary":
        return await handle_summary_request(prompt, user, conv_id, stream)

    # DEFAULT CHAT
    if stream:
        async def event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task

            try:
                try:
                    history = await get_history(conv_id)
                except Exception as e:
                    logger.error(f"History fetch failed: {e}")
                    history = [] 
                
                base_system = get_system_prompt(prompt)
                user_memory = user.get("memory", "")
                if user_memory:
                    base_system += f"\n\nUser Context: {user_memory}"
                
                full_history = [{"role": "system", "content": base_system}] + history
                
                full_text = ""
                async for token in stream_groq_chat(full_history):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})

                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")
                
                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Universal Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred processing your request."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    else:
        history = await get_history(conv_id)
        base_system = get_system_prompt(prompt)
        user_memory = user.get("memory", "")
        if user_memory:
            base_system += f"\n\nUser Context: {user_memory}"
            
        full_history = [{"role": "system", "content": base_system}] + history
        
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": "llama-3.3-70b-versatile", "messages": full_history, "max_tokens": 1024}
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            await save_message(user["id"], conv_id, "assistant", reply)
            return {"reply": reply}


# =========================
# SPECIALIZED HANDLERS
# =========================
async def handle_math_request(prompt: str, user: Dict[str, Any], conv_id: str, stream: bool):
    system_prompt = get_system_prompt(prompt) + """

You are also a mathematical expert. When solving math problems:
1. Show your work step-by-step
2. Explain each step clearly
3. Use proper mathematical notation
4. Verify your answer
5. If it's a proof, be rigorous

Format complex equations clearly using LaTeX-style notation where appropriate."""

    user_memory = user.get("memory", "")
    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Math Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user["id"], conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_research_request(prompt: str, user: Dict[str, Any], conv_id: str, stream: bool):
    system_prompt = get_system_prompt(prompt) + """

You are also a research assistant. When helping with research:
1. Provide well-structured, factual information
2. Cite sources when possible (even if general)
3. Present multiple perspectives on controversial topics
4. Identify gaps in current knowledge
5. Suggest further areas of investigation
Be thorough but concise."""

    user_memory = user.get("memory", "")
    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages, max_tokens=2048):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Research Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user["id"], conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_creative_request(prompt: str, user: Dict[str, Any], conv_id: str, stream: bool):
    system_prompt = get_system_prompt(prompt) + """

You are also a creative writing expert. When writing creative content:
1. Use vivid, engaging language
2. Create compelling characters and narratives
3. Pay attention to rhythm and flow
4. Match the requested style or genre
5. Be original and imaginative
Adapt your style to the specific creative request."""

    user_memory = user.get("memory", "")
    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages, max_tokens=2048):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Creative Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user["id"], conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_translation_request(prompt: str, user: Dict[str, Any], conv_id: str, stream: bool):
    system_prompt = get_system_prompt(prompt) + """

You are also a professional translator. When translating:
1. Preserve the meaning and tone of the original
2. Use natural, idiomatic language in the target
3. Handle cultural nuances appropriately
4. Maintain formatting where possible
5. If unsure about context, provide alternatives
Always indicate the source and target languages."""

    user_memory = user.get("memory", "")
    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Translation Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user["id"], conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_summary_request(prompt: str, user: Dict[str, Any], conv_id: str, stream: bool):
    system_prompt = get_system_prompt(prompt) + """

You are also a summarization expert. When summarizing:
1. Extract the most important points
2. Maintain accuracy - don't add information
3. Be concise but complete
4. Use bullet points for clarity when appropriate
5. Start with a brief overview, then key points
Tailor the summary length to the complexity of the content."""

    user_memory = user.get("memory", "")
    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Summary Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user["id"], conv_id, "assistant", reply)
    return {"reply": reply}


# =========================
# CHAT MANAGEMENT ENDPOINTS
# =========================
@app.post("/newchat")
async def new_chat(req: Request, res: Response):
    user = await get_user(req, res)
    cid = str(uuid.uuid4())
    await _execute_supabase_with_retry(
        supabase.table("conversations").insert({
            "id": cid, "user_id": user["id"],
            "title": "New Chat", "created_at": datetime.now(timezone.utc).isoformat()
        }),
        description="New Chat"
    )
    return {"conversation_id": cid}


@app.post("/stop")
async def stop_generation(req: Request, res: Response):
    user = await get_user(req, res)
    user_id = user["id"]

    task = active_streams.get(user_id)
    if task and not task.done():
        task.cancel()
        active_streams.pop(user_id, None)
        return {"status": "stopped"}
    return {"status": "no_active_stream"}


@app.post("/regenerate")
async def regenerate(req: Request, res: Response):
    body = await req.json()
    conv_id = body.get("conversation_id")

    user = await get_user(req, res)
    user_id = user["id"]

    if not conv_id:
        raise HTTPException(400, "conversation_id required")

    msgs = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("*")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=True)
        .limit(10),
        description="Regenerate History Lookup"
    )

    if not msgs.data:
        raise HTTPException(404, "No messages found")

    last_user_msg = None
    for m in msgs.data:
        if m["role"] == "user":
            last_user_msg = m
            break

    if not last_user_msg:
        raise HTTPException(400, "No user message to regenerate from")

    await _execute_supabase_with_retry(
        supabase.table("messages")
        .delete()
        .gt("created_at", last_user_msg["created_at"])
        .eq("role", "assistant")
        .eq("conversation_id", conv_id),
        description="Delete Old Assistant Message"
    )

    async def event_gen():
        task = asyncio.current_task()
        active_streams[user_id] = task
        try:
            history = await get_history(conv_id)
            
            # Use the last user prompt to check for creator questions
            last_prompt = last_user_msg.get("content", "")
            base_system = get_system_prompt(last_prompt)
            user_memory = user.get("memory", "")
            if user_memory:
                base_system += f"\n\nUser Context: {user_memory}"
            full_history = [{"role": "system", "content": base_system}] + history
            
            full_text = ""
            async for token in stream_groq_chat(full_history):
                if task and task.cancelled():
                    break
                full_text += token
                yield sse({"type": "token", "text": token})

            try:
                await save_message(user_id, conv_id, "assistant", full_text)
            except Exception as e:
                logger.error(f"Failed to save assistant message: {e}")

            yield sse({"type": "done"})
        
        except Exception as e:
            logger.error(f"Regenerate Stream Error: {e}")
            yield sse({"type": "error", "message": "An error occurred."})
        
        finally:
            active_streams.pop(user_id, None)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/chats")
async def list_chats(req: Request, res: Response):
    user = await get_user(req, res)
    
    result = await _execute_supabase_with_retry(
        supabase.table("conversations")
        .select("*")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True),
        description="List Chats"
    )
    return {"chats": result.data or []}


# =========================
# USER IDENTITY ENDPOINT
# =========================
@app.get("/user/info")
async def get_user_info(req: Request, res: Response):
    user = await get_user(req, res)
    return {
        "user_id": user["id"],
        "fingerprint": user.get("fingerprint", "")[:8] + "...",
        "is_identified": True
    }


@app.post("/user/merge")
async def merge_user(req: Request, res: Response):
    body = await req.json()
    target_id = body.get("target_user_id")
    
    user = await get_user(req, res)
    
    if not target_id or target_id == user["id"]:
        return {"status": "no_merge_needed"}
    
    try:
        await _execute_supabase_with_retry(
            supabase.table("conversations")
            .update({"user_id": target_id})
            .eq("user_id", user["id"]),
            description="Merge Conversations"
        )
        
        await _execute_supabase_with_retry(
            supabase.table("messages")
            .update({"user_id": target_id})
            .eq("user_id", user["id"]),
            description="Merge Messages"
        )
        
        fingerprint = user.get("fingerprint", "")
        set_all_cookies(res, target_id, fingerprint)
        
        return {"status": "merged", "new_user_id": target_id}
    
    except Exception as e:
        logger.error(f"User merge failed: {e}")
        raise HTTPException(500, "Failed to merge user data")


# =========================
# INTENT ANALYSIS ENDPOINT
# =========================
@app.post("/analyze-intent")
async def analyze_intent_endpoint(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")

    if not prompt:
        raise HTTPException(400, "Prompt required")

    intent_result = detect_intent(prompt)
    action_type = get_action_type(prompt)
    required_tools = get_required_tools(prompt)

    return {
        "intent": intent_result.to_dict() if intent_result else None,
        "action_type": action_type,
        "required_tools": required_tools,
        "confidence": intent_result.confidence if intent_result else 0.0
    }


# =========================
# MEDIA ENDPOINTS
# =========================

@app.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text")
    voice = data.get("voice", "alloy")

    if not text:
        raise HTTPException(400, "text required")
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI Key missing")

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=get_openai_headers(),
                json={"model": "tts-1", "voice": voice, "input": text}
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            return JSONResponse(
                status_code=e.response.status_code,
                content={"error": e.response.text}
            )

        return Response(content=r.content, media_type="audio/mpeg")

@app.post("/analysis")
async def analyze_file(
        req: Request,
        file: UploadFile = File(...),
        stream: bool = True
):
    user = await get_user(req, Response())

    content = await file.read()
    filename = file.filename.lower()
    content_type = file.content_type or ""

    if not content:
        raise HTTPException(400, "Empty file")

    if len(content) > 15_000_000:
        raise HTTPException(400, "File too large (15MB max)")

    if content_type.startswith("image/"):
        return await handle_image_analysis(content, stream)

    text = await universal_text_extractor(content, filename)
    return await handle_text_analysis(text, stream)


@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "alloy", "name": "Alloy"},
            {"id": "echo", "name": "Echo"},
            {"id": "fable", "name": "Fable"},
            {"id": "onyx", "name": "Onyx"},
            {"id": "nova", "name": "Nova"},
            {"id": "shimmer", "name": "Shimmer"}
        ]
    }


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI Key missing")

    content = await file.read()

    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": (file.filename, content, file.content_type)}
        data = {"model": "whisper-1"}
        r = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files, data=data
        )
        r.raise_for_status()
        return r.json()


# =========================
# MEDIA GENERATION HANDLERS
# =========================
async def handle_image_generation(
        prompt: str,
        user: Dict[str, Any],
        conv_id: str,
        stream: bool,
        style: str = None,
        size: str = "1024x1024",
        num_images: int = 1
):
    MAX_PROMPT_LEN = 3000
    MAX_IMAGES = 4

    if not OPENAI_API_KEY:
        msg = "No API Key"
        if stream:
            async def gen():
                yield sse({"type": "error", "message": msg})

            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"error": msg}

    if not prompt or not prompt.strip():
        raise HTTPException(400, "Prompt is required")

    original_len = len(prompt)
    if original_len > MAX_PROMPT_LEN:
        logger.warning(f"[IMG] Prompt trimmed from {original_len} → {MAX_PROMPT_LEN}")
        prompt = prompt[:MAX_PROMPT_LEN]

    num_images = max(1, min(num_images, MAX_IMAGES))

    STYLES = {
        "realistic": "ultra realistic, 4k, highly detailed",
        "cartoon": "cartoon style, vibrant colors",
        "anime": "anime style, studio ghibli inspired",
        "cinematic": "cinematic lighting, dramatic shadows",
        "cyberpunk": "cyberpunk, neon futuristic, high contrast",
    }

    if style in STYLES:
        prompt = f"{prompt}, {STYLES[style]}"

    start_time = time.time()
    logger.info(f"[IMG] Generating image | user={user['id']} | len={len(prompt)}")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers=get_openai_headers(),
                json={
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "size": size,
                    "n": num_images
                }
            )

            if r.status_code != 200:
                logger.error(f"[IMG ERROR] {r.status_code}: {r.text}")

            r.raise_for_status()
            data = r.json()

        latency = int((time.time() - start_time) * 1000)
        logger.info(f"[IMG] Image latency: {latency}ms")

    except httpx.HTTPStatusError as e:
        logger.error(f"[IMG FAIL] HTTP error: {str(e)}")

        if stream:
            async def gen():
                yield sse({
                    "type": "error",
                    "message": "Image generation failed",
                    "details": str(e)
                })

            return StreamingResponse(gen(), media_type="text/event-stream")

        return {"error": "Image generation failed", "details": str(e)}

    except Exception as e:
        logger.error(f"[IMG FAIL] Unexpected error: {str(e)}")

        if stream:
            async def gen():
                yield sse({
                    "type": "error",
                    "message": "Unexpected error",
                    "details": str(e)
                })

            return StreamingResponse(gen(), media_type="text/event-stream")

        return {"error": "Unexpected error", "details": str(e)}

    images = []

    for item in data.get("data", []):
        try:
            b64 = item.get("b64_json")
            url = item.get("url")

            if url:
                images.append({"url": url})
            elif b64:
                img_bytes = base64.b64decode(b64)
                fname = f"{uuid.uuid4().hex}.png"
                path = f"public/{fname}"

                try:
                    await asyncio.to_thread(
                        lambda: supabase.storage.from_("ai-images").upload(
                            path, img_bytes, {"content-type": "image/png"}
                        )
                    )
                    url = f"{SUPABASE_URL}/storage/v1/object/public/ai-images/{path}"
                except Exception as storage_err:
                    logger.warning(f"[IMG STORAGE FAIL] {storage_err}")
                    url = "data:image/png;base64," + b64
                
                images.append({"url": url})

        except Exception as decode_err:
            logger.error(f"[IMG DECODE FAIL] {decode_err}")

    if stream:
        async def gen():
            yield sse({"type": "images", "images": images})
            yield sse({"type": "done"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {"images": images}


async def handle_video_generation(prompt, user: Dict[str, Any], conv_id: str, stream):
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    if stream:
        async def gen():
            try:
                yield sse({"type": "status", "message": "Starting video generation..."})
                
                async with httpx.AsyncClient(timeout=120) as client:
                    create_res = await client.post(
                        "https://api.replicate.com/v1/predictions",
                        headers=headers,
                        json={
                            "version": "your-model-version-id",
                            "input": {
                                "prompt": prompt
                            }
                        }
                    )
                    create_res.raise_for_status()
                    prediction = create_res.json()
                    prediction_id = prediction["id"]
                    
                    yield sse({"type": "status", "message": "Processing video..."})

                    max_polls = 120
                    poll_count = 0
                    
                    while poll_count < max_polls:
                        poll_res = await client.get(
                            f"https://api.replicate.com/v1/predictions/{prediction_id}",
                            headers=headers
                        )
                        poll_res.raise_for_status()
                        data = poll_res.json()

                        if data["status"] == "succeeded":
                            raw_video_url = data["output"]
                            
                            yield sse({"type": "status", "message": "Adding watermark..."})
                            watermarked_url = await add_watermark_to_video(raw_video_url)
                            
                            yield sse({"type": "video", "url": watermarked_url})
                            yield sse({"type": "done"})
                            return
                            
                        elif data["status"] == "failed":
                            error_msg = data.get("error", "Unknown error")
                            yield sse({"type": "error", "message": f"Video generation failed: {error_msg}"})
                            return
                        
                        else:
                            poll_count += 1
                            await asyncio.sleep(2)

                    yield sse({"type": "error", "message": "Video generation timed out"})
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"Video gen HTTP error: {e}")
                yield sse({"type": "error", "message": f"API error: {e.response.status_code}"})
            except Exception as e:
                logger.error(f"Video gen error: {e}")
                yield sse({"type": "error", "message": str(e)})

        return StreamingResponse(gen(), media_type="text/event-stream")
    
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            create_res = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json={
                    "version": "your-model-version-id",
                    "input": {"prompt": prompt}
                }
            )
            create_res.raise_for_status()
            prediction = create_res.json()
            prediction_id = prediction["id"]

            while True:
                poll_res = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=headers
                )
                poll_res.raise_for_status()
                data = poll_res.json()

                if data["status"] == "succeeded":
                    raw_video_url = data["output"]
                    watermarked_url = await add_watermark_to_video(raw_video_url)
                    return {"video_url": watermarked_url}
                elif data["status"] == "failed":
                    raise HTTPException(500, f"Video generation failed: {data.get('error')}")

                await asyncio.sleep(2)


# =========================
# ROOT
# =========================
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "HeloxAi Backend",
        "intent_detection": "advanced",
        "user_recognition": "multi-cookie-fingerprint"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
