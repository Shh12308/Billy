import os
import re
import json
import base64
import uuid
import asyncio
import logging
from io import BytesIO
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heloxai.xyz", "https://www.heloxai.xyz", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Client
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Global State for Stream Cancellation (In-memory for single-instance/Railway sticky sessions)
active_streams: Dict[str, asyncio.Task] = {}


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
        """Pre-compile regex patterns for performance"""
        self.patterns = {
            IntentCategory.IMAGE_GENERATION: [
                r'\b(generate|create|make|draw|render|paint|sketch|illustrate)\s+(a\s+|an\s+)?(image|picture|photo|drawing|illustration|artwork|painting|sketch|graphic|visual)',
                r'\b(image|picture|photo|drawing|illustration)\s+(of|showing|depicting|with|for|about)',
                r'\b(text\s+to\s+image|txt2img|img2img)',
                r'\b(visualize|visualise)\s+(this|that|the|it)',
                r'\b(dall[eé]|midjourney|stable\s+diffusion|sd\s*xl|flux)',
                r'\b(generate|create)\s+(some\s+)?art',
                r'\b(make\s+(me\s+)?(a\s+)?(visual|graphic|thumbnail|logo|icon|banner|poster)',
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
                r'\b(implement\s+(the|a|this)\s+(\w+\s+)?(pattern|algorithm|logic|feature)',
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
                r'\b(api\s*(endpoint|route|handler|controller|gateway)',
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
                r'\b(definition|meaning)\s*(of|for)\s+',
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

        # Compile all patterns
        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.patterns.items()
        }

    def _init_synonyms(self):
        """Initialize synonym mappings for fuzzy keyword matching"""
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
        """Check if there's a negation word before the keyword (within 6 words)"""
        words_before = text[:keyword_pos].lower().split()[-6:]
        preceding_text = " ".join(words_before)
        return any(neg in preceding_text for neg in self.negation_words)

    def _calculate_confidence(
            self,
            matched_keywords: List[str],
            matched_patterns: List[str],
            text_length: int
    ) -> float:
        """Calculate confidence score based on matches"""
        if not matched_keywords and not matched_patterns:
            return 0.0

        pattern_confidence = min(len(matched_patterns) * 0.35, 0.65)
        keyword_confidence = min(len(matched_keywords) * 0.12, 0.25)
        multi_signal_bonus = 0.1 if (matched_keywords and matched_patterns) else 0.0
        length_factor = max(0.5, 1.0 - (text_length / 1500) * 0.4)

        confidence = (pattern_confidence + keyword_confidence + multi_signal_bonus) * length_factor
        return min(confidence, 1.0)

    def _are_related_intents(self, intent1: IntentCategory, intent2: IntentCategory) -> bool:
        """Check if two intents are related (can be sub-intents)"""
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
        """Detect all intents with confidence scores"""
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
        """Get the highest confidence intent"""
        results = self.detect_intents(text)
        return results[0] if results else None

    def get_action_type(self, text: str) -> str:
        """Get high-level action type for routing"""
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
        """Determine which tools/APIs are needed"""
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
        """Get specialized system prompt based on code sub-intent"""
        intent = self.get_primary_intent(text)
        if not intent:
            return "You are a helpful coding assistant."

        sub_prompts = {
            IntentCategory.CODE_DEBUG: """You are an expert debugger. When analyzing code issues:
1. Identify the root cause of the bug/error
2. Explain WHY it's happening (not just what)
3. Provide the exact fix with clear code blocks
4. Suggest how to prevent similar issues
Be precise and practical.""",

            IntentCategory.CODE_REVIEW: """You are a senior code reviewer. Provide constructive feedback on:
1. Code quality and readability
2. Potential bugs or edge cases
3. Performance considerations
4. Best practices and design patterns
5. Security concerns
Be specific and actionable in your suggestions.""",

            IntentCategory.CODE_GENERATION: """You are an expert software engineer. When writing code:
1. Write clean, well-structured, production-ready code
2. Include appropriate error handling
3. Add helpful comments for complex logic
4. Consider edge cases
5. Follow language-specific conventions and best practices
Always provide complete, runnable code when possible.""",

            IntentCategory.WEB_DEVELOPMENT: """You are a full-stack web developer expert. When building web components:
1. Use modern best practices and frameworks
2. Ensure responsive design
3. Consider accessibility (a11y)
4. Include proper styling
5. Make components reusable and maintainable
Provide complete, ready-to-use code.""",

            IntentCategory.API_DEVELOPMENT: """You are an API development expert. When creating APIs:
1. Follow RESTful principles (or GraphQL best practices)
2. Include proper error handling and status codes
3. Add input validation
4. Consider security (auth, rate limiting)
5. Document endpoints clearly
Provide complete, production-ready code.""",

            IntentCategory.DATABASE: """You are a database expert. When working with databases:
1. Design efficient, normalized schemas
2. Write optimized queries
3. Include proper indexes
4. Consider data integrity with constraints
5. Follow SQL best practices
Provide complete, ready-to-execute SQL/ORM code.""",
        }

        return sub_prompts.get(intent.intent, "You are a helpful coding assistant.")


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
    """Get detailed intent with confidence and metadata"""
    return get_detector().get_primary_intent(prompt)


def get_action_type(prompt: str) -> str:
    """Get high-level action type for routing"""
    return get_detector().get_action_type(prompt)


def get_required_tools(prompt: str) -> List[str]:
    """Get list of tools needed for the request"""
    return get_detector().get_required_tools(prompt)


def is_debug_request(prompt: str) -> bool:
    """Check if this is a debugging request"""
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_DEBUG


def is_review_request(prompt: str) -> bool:
    """Check if this is a code review request"""
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_REVIEW


def get_intent_confidence(prompt: str) -> float:
    """Get confidence score for detected intent"""
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
COOKIE_NAME = "HeloxAi4life"


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _execute_supabase_with_retry(query_builder, description="Supabase Operation"):
    """
    Executes a Supabase query builder with retries for transient errors (502, Cloudflare).
    """
    max_retries = 3
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(query_builder.execute)
        except Exception as e:
            last_exception = e
            error_str = str(e)
            # Check for 502 Bad Gateway or JSON decoding errors (which happen on 502 HTML responses)
            if "502" in error_str or "Bad Gateway" in error_str or "Expecting value" in error_str:
                logger.warning(f"{description} encountered transient error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1)) # Exponential backoff
                    continue
            else:
                # For non-transient errors, break immediately
                logger.error(f"{description} failed: {e}")
                break
    
    if last_exception:
        raise last_exception


async def get_user(request: Request, response: Response) -> Dict[str, Any]:
    """
    Robust user fetching logic.
    1. Checks Cookie 'HeloxAi4life' (used for Anonymous/Guest persistence).
    2. Syncs with 'public.users' table based on the provided SQL Schema.
    3. Returns user dict including is_premium and is_lifetime.
    """
    session_token = request.cookies.get(COOKIE_NAME)
    
    # Default user object (Free/Guest) aligned with SQL Schema
    user_obj = {
        "id": None,
        "email": None
    }

    # 1. Try to find user via Cookie (Guest/Anonymous Persistence)
    # NOTE: We assume the cookie contains a UUID that maps directly to users.id
    if session_token:
        try:
            # Check if user exists in DB by ID
            user_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("*").eq("id", session_token).limit(1),
                description="User Lookup by ID"
            )
            if user_resp.data:
                u = user_resp.data[0]
                # Map DB columns to our internal user object
                user_obj = {
                    "id": u["id"],
                    "email": u.get("email")
                }
                return user_obj
        except Exception as e:
            logger.error(f"Cookie user lookup failed: {e}")

    # 2. Create Anonymous User if none found (and we have or can generate a token)
    # If no cookie, generate a new UUID
    if not session_token:
        session_token = str(uuid.uuid4())
        response.set_cookie(
            key=COOKIE_NAME, 
            value=session_token, 
            max_age=86400*30, 
            httponly=True, 
            secure=True, 
            samesite="none"
        )
    
    # Create row in public.users matching the schema
    try:
        new_user_data = {
            "id": session_token, # PK
            "email": f"anon+{session_token[:8]}@local"
        )
        
        await _execute_supabase_with_retry(
            supabase.table("users").insert(new_user_data),
            description="Create Anonymous User"
        )
        
        user_obj["id"] = session_token
        # user_obj already has defaults set above
        
    except Exception as e:
        logger.error(f"Failed to create anonymous user: {e}")
        # Fallback: return session-only user object (won't persist to DB but won't crash)
        user_obj["id"] = session_token

    return user_obj


def get_groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}


def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


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


async def handle_text_analysis(text: str, stream: bool):
    text = text[:15000]

    messages = [
        {
            "role": "system",
            "content": """You analyze files. Detect type automatically and respond accordingly:

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
            async for token in stream_groq_chat(messages):
                yield sse({"type": "token", "text": token})
            yield sse({"type": "done"})

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


async def handle_image_analysis(image_bytes: bytes, stream: bool):
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
            yield sse({"type": "text", "text": result})
            yield sse({"type": "done"})

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
    """Streams response from Groq"""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens}
        ) as resp:
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


async def handle_code_assistant(prompt: str, user_id: str, conv_id: str, stream: bool):
    """Advanced code assistant with intent-aware system prompts"""
    system_prompt = get_detector().get_code_system_prompt(prompt)

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    # Get intent info for logging
    intent_result = detect_intent(prompt)
    logger.info(
        f"[CODE] sub_intent={intent_result.intent.value if intent_result else 'none'} "
        f"confidence={(intent_result.confidence if intent_result else 0):.2%}"
    )

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user_id] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})

                if conv_id:
                    try:
                        await save_message(user_id, conv_id, "assistant", full_text)
                    except Exception as e:
                        logger.error(f"Failed to save assistant message: {e}")
                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

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
        await save_message(user_id, conv_id, "assistant", reply)
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


@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    content_type = req.headers.get("content-type", "")

    # ------------------------
    # SAFE BODY PARSING
    # ------------------------
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

    # ------------------------
    # CONTINUE NORMAL FLOW
    # ------------------------
    prompt = body.get("prompt", "")
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)

    if not prompt:
        raise HTTPException(400, "Prompt required")

    user = await get_user(req, res)
    user_id = user["id"]

    # Handle conversation creation
    if not conv_id:
        conv_id = str(uuid.uuid4())
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert({
                "id": conv_id, "user_id": user_id,
                "title": prompt[:30], "created_at": datetime.now(timezone.utc).isoformat()
            }),
            description="Create Conversation"
        )

    await save_message(user_id, conv_id, "user", prompt)

    # ------------------------
    # ADVANCED INTENT DETECTION
    # ------------------------
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

    # ------------------------
    # PERMISSION CHECKS (Premium/Lifetime)
    # ------------------------
    if action_type in ["image", "video"]:
        # Check if user is allowed based on DB flags
        is_allowed = user.get("is_premium") or user.get("is_lifetime")
        if not is_allowed:
            error_msg = f"{'Image' if action_type == 'image' else 'Video'} generation requires a Premium or Lifetime plan."
            if stream:
                async def gen(): yield sse({"type": "error", "message": error_msg})
                return StreamingResponse(gen(), media_type="text/event-stream")
            raise HTTPException(403, error_msg)

    # ------------------------
    # INTENT-BASED ROUTING
    # ------------------------
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
            return await handler(prompt, stream)
        else:
            return await handler(prompt, user_id, conv_id, stream)

    # ------------------------
    # SPECIALIZED HANDLERS
    # ------------------------
    if action_type == "math":
        return await handle_math_request(prompt, user_id, conv_id, stream)

    if action_type == "research":
        return await handle_research_request(prompt, user_id, conv_id, stream)

    if action_type == "creative":
        return await handle_creative_request(prompt, user_id, conv_id, stream)

    if action_type == "translation":
        return await handle_translation_request(prompt, user_id, conv_id, stream)

    if action_type == "summary":
        return await handle_summary_request(prompt, user_id, conv_id, stream)

    # ------------------------
    # DEFAULT CHAT
    # ------------------------
    if stream:
        async def event_gen():
            task = asyncio.current_task()
            active_streams[user_id] = task

            try:
                history = await get_history(conv_id)
                full_history = [{"role": "system", "content": "You are a helpful AI."}] + history
                full_text = ""
                async for token in stream_groq_chat(full_history):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})

                try:
                    await save_message(user_id, conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")
                
                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    else:
        history = await get_history(conv_id)
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": "llama-3.3-70b-versatile", "messages": history, "max_tokens": 1024}
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            await save_message(user_id, conv_id, "assistant", reply)
            return {"reply": reply}


# =========================
# SPECIALIZED HANDLERS
# =========================
async def handle_math_request(prompt: str, user_id: str, conv_id: str, stream: bool):
    """Handle mathematical queries with specialized prompting"""
    system_prompt = """You are a mathematical expert. When solving math problems:
1. Show your work step-by-step
2. Explain each step clearly
3. Use proper mathematical notation
4. Verify your answer
5. If it's a proof, be rigorous

Format complex equations clearly using LaTeX-style notation where appropriate."""

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user_id] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user_id, conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user_id, conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_research_request(prompt: str, user_id: str, conv_id: str, stream: bool):
    """Handle research-oriented queries"""
    system_prompt = """You are a research assistant. When helping with research:
1. Provide well-structured, factual information
2. Cite sources when possible (even if general)
3. Present multiple perspectives on controversial topics
4. Identify gaps in current knowledge
5. Suggest further areas of investigation
Be thorough but concise."""

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user_id] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages, max_tokens=2048):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user_id, conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user_id, conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_creative_request(prompt: str, user_id: str, conv_id: str, stream: bool):
    """Handle creative writing requests"""
    system_prompt = """You are a creative writing expert. When writing creative content:
1. Use vivid, engaging language
2. Create compelling characters and narratives
3. Pay attention to rhythm and flow
4. Match the requested style or genre
5. Be original and imaginative
Adapt your style to the specific creative request."""

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user_id] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages, max_tokens=2048):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user_id, conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user_id, conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_translation_request(prompt: str, user_id: str, conv_id: str, stream: bool):
    """Handle translation requests"""
    system_prompt = """You are a professional translator. When translating:
1. Preserve the meaning and tone of the original
2. Use natural, idiomatic language in the target
3. Handle cultural nuances appropriately
4. Maintain formatting where possible
5. If unsure about context, provide alternatives
Always indicate the source and target languages."""

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user_id] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user_id, conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user_id, conv_id, "assistant", reply)
    return {"reply": reply}


async def handle_summary_request(prompt: str, user_id: str, conv_id: str, stream: bool):
    """Handle summarization requests"""
    system_prompt = """You are a summarization expert. When summarizing:
1. Extract the most important points
2. Maintain accuracy - don't add information
3. Be concise but complete
4. Use bullet points for clarity when appropriate
5. Start with a brief overview, then key points
Tailor the summary length to the complexity of the content."""

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user_id] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                try:
                    await save_message(user_id, conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    await save_message(user_id, conv_id, "assistant", reply)
    return {"reply": reply}


# =========================
# CHAT MANAGEMENT ENDPOINTS
# =========================
@app.post("/newchat")
async def new_chat(req: Request, res: Response):
    """Creates a new conversation and returns the ID"""
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
    """Cancels the current stream for the user"""
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
    """Regenerates the last assistant response"""
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
            full_text = ""
            async for token in stream_groq_chat(history):
                if task and task.cancelled():
                    break
                full_text += token
                yield sse({"type": "token", "text": token})

            try:
                await save_message(user_id, conv_id, "assistant", full_text)
            except Exception as e:
                logger.error(f"Failed to save assistant message: {e}")

            yield sse({"type": "done"})
        except asyncio.CancelledError:
            yield sse({"type": "stopped"})
        finally:
            active_streams.pop(user_id, None)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/chats")
async def list_chats(req: Request, res: Response):
    """List all user chats"""
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
# INTENT ANALYSIS ENDPOINT
# =========================
@app.post("/analyze-intent")
async def analyze_intent_endpoint(req: Request):
    """Analyze the intent of a prompt without executing it"""
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
    """Text to Speech"""
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
    """Speech to Text (Whisper)"""
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
        user_id: str,
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
    logger.info(f"[IMG] Generating image | user={user_id} | len={len(prompt)}")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers=get_openai_headers(),
                json={
                    "model": "dall-e-3", # Corrected model name
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
            # DALL-E 3 returns url, DALL-E 2 returns b64_json. Handle both.
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


async def handle_video_generation(prompt, user_id, conv_id, stream):
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        # 1. Create prediction
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

        # 2. Poll
        while True:
            poll_res = await client.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers
            )
            poll_res.raise_for_status()
            data = poll_res.json()

            if data["status"] == "succeeded":
                video_url = data["output"]
                break
            elif data["status"] == "failed":
                raise Exception("Video generation failed")

            await asyncio.sleep(2)


# =========================
# ROOT
# =========================
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "HeloxAi Backend",
        "intent_detection": "advanced"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
