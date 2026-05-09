"""Project-wide paths, constants, and shared regex/lexicon definitions."""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
BERTOPIC_DIR = DATA_DIR / "bertopic"
ENTITY_DIR = DATA_DIR / "entity"
SENTIMENT_DIR = DATA_DIR / "sentiment"

OUTPUTS_DIR = ROOT / "outputs"
BERTOPIC_MODEL_DIR = OUTPUTS_DIR / "bertopic_model"
SENTIMENT_MODEL_DIR = OUTPUTS_DIR / "sentiment"

for _p in (RAW_DIR, CLEANED_DIR, BERTOPIC_DIR, ENTITY_DIR, SENTIMENT_DIR,
           BERTOPIC_MODEL_DIR, SENTIMENT_MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── Data source ───────────────────────────────────────────────────────────────
DATA_URL = (
    "https://storage.googleapis.com/msca-bdp-data-open/"
    "news_final_project/news_final_project.parquet"
)

# ── AI relevance: triggers, high-signal terms, support terms ──────────────────
AI_TRIGGERS = [
    r"\bai\b", r"artificial intelligence", r"machine learning", r"deep learning",
    r"\bllm\b", r"large language model", r"generative ai", r"chatgpt",
    r"gpt-?\d+", r"openai", r"transformer", r"diffusion", r"stable diffusion",
    r"neural network", r"foundation model",
]

HIGH_SIGNAL_TERMS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "llm", "large language model", "generative ai", "nlp",
    "computer vision", "neural network", "foundation model",
    "autonomous systems", "robotics", "transformer", "diffusion model",
]

SUPPORT_TERMS = [
    "automation", "augmentation", "model training", "chatbot", "chatgpt",
    "copilot", "predictive analytics", "openai", "anthropic", "nvidia",
    "hugging face", "workflow", "personalization", "recommendation",
    "image generation",
]

BOILERPLATE_TERMS = [
    "cookie policy", "privacy policy", "terms of service", "subscribe",
    "newsletter", "advertisement", "sign up", "log in", "all rights reserved",
    "follow us", "share this", "comments are closed",
]

# ── Industry anchors (topic → industry mapping) ───────────────────────────────
# Each industry has 2 anchor descriptions (~50-70 words each); a topic's
# embedding is matched against the mean of these to assign an industry label.
INDUSTRY_ANCHORS = {
    "Office and Administrative Support": [
        "AI is automating routine administrative tasks such as scheduling, document processing, email triage, and data entry through workflow automation and intelligent assistants.",
        "Generative AI tools are reshaping back-office operations by reducing clerical workload, improving information retrieval, and enabling fewer staff to handle higher volumes of administrative work.",
    ],
    "Legal": [
        "AI is transforming legal services through document review automation, contract analysis, legal research assistants, and e-discovery powered by large language models.",
        "Law firms and corporate legal departments use AI to reduce research time, standardize compliance checks, and augment lawyers rather than fully replace legal judgment.",
    ],
    "Architecture and Engineering": [
        "AI supports architecture and engineering by enabling generative design, automated simulations, and optimization of structural, mechanical, and energy-efficient systems.",
        "Machine learning models are used to accelerate design iteration, detect errors earlier, and integrate digital twins into planning and construction workflows.",
    ],
    "Life, Physical, and Social Science": [
        "AI accelerates scientific discovery through data-driven modeling, pattern detection, and simulation across biology, chemistry, physics, and social sciences.",
        "Researchers apply machine learning to analyze large experimental datasets, automate hypothesis generation, and improve predictive accuracy in complex systems.",
    ],
    "Business and Financial Operations": [
        "AI is widely adopted in finance and business operations for forecasting, fraud detection, risk modeling, pricing, and decision support.",
        "Automation and predictive analytics improve operational efficiency while shifting human roles toward oversight, strategy, and exception handling.",
    ],
    "Community and Social Service": [
        "AI supports community and social services by improving case triage, resource allocation, and early risk detection for vulnerable populations.",
        "Decision-support tools help social workers prioritize interventions while raising ethical considerations around bias, transparency, and accountability.",
    ],
    "Management": [
        "AI augments management functions by providing data-driven insights for planning, performance monitoring, and organizational decision-making.",
        "Executives use AI-powered dashboards and forecasting tools to optimize strategy, workforce allocation, and operational efficiency.",
    ],
    "Sales and Related": [
        "AI is reshaping sales through customer segmentation, demand forecasting, recommendation systems, and automated lead scoring.",
        "Generative AI enhances sales productivity by assisting with outreach, proposal drafting, and customer interaction analysis.",
    ],
    "Computer and Mathematical": [
        "AI plays a central role in computer and mathematical occupations, including machine learning development, data science, algorithm design, and applied optimization.",
        "Professionals in this field both build AI systems and use them to automate coding, analysis, and model experimentation.",
    ],
    "Farming, Fishing, and Forestry": [
        "AI enables precision agriculture through crop monitoring, yield prediction, automated equipment, and environmental sensing.",
        "Machine learning helps optimize resource usage while reducing labor intensity and improving sustainability outcomes.",
    ],
    "Protective Service": [
        "AI supports protective services through predictive policing tools, surveillance analytics, and real-time risk assessment systems.",
        "Automation improves situational awareness and response speed, while human oversight remains critical for ethical and legal accountability.",
    ],
    "Healthcare Practitioners and Technical": [
        "AI assists healthcare practitioners with diagnostic imaging, clinical decision support, and patient risk stratification.",
        "These tools augment clinical expertise by improving accuracy and efficiency rather than fully replacing medical professionals.",
    ],
    "Educational Instruction and Library": [
        "AI impacts education by enabling personalized learning, automated grading, tutoring systems, and content recommendation.",
        "Educators increasingly use AI tools to support instruction, assessment, and administrative tasks while redefining teaching roles.",
    ],
    "Healthcare Support": [
        "AI automates healthcare support tasks such as scheduling, patient monitoring, documentation, and workflow coordination.",
        "Automation reduces administrative burden and allows support staff to focus on patient-facing and care coordination activities.",
    ],
    "Arts, Design, Entertainment, Sports, and Media": [
        "Generative AI is transforming creative industries through automated content creation, design assistance, and media production tools.",
        "AI augments creative workflows while raising questions around originality, intellectual property, and labor displacement.",
    ],
    "Personal Care and Service": [
        "AI has limited but growing impact on personal care services through scheduling tools, customer analytics, and service personalization.",
        "Most tasks remain human-centered, with AI primarily supporting logistics and operational efficiency.",
    ],
    "Food Preparation and Serving Related": [
        "AI is applied in food services for demand forecasting, inventory optimization, and automated ordering systems.",
        "While some back-of-house tasks are automated, customer interaction and service delivery remain largely human-driven.",
    ],
    "Transportation and Material Moving": [
        "AI affects transportation through route optimization, autonomous vehicles, predictive maintenance, and logistics planning.",
        "Automation improves efficiency and safety while gradually reshaping driver and operator roles.",
    ],
    "Production": [
        "AI enables smart manufacturing through predictive maintenance, quality inspection, robotics, and process optimization.",
        "Machine learning systems increase throughput and consistency while reducing manual intervention in production lines.",
    ],
    "Construction and Extraction": [
        "AI supports construction and extraction by improving site planning, equipment monitoring, safety analytics, and resource forecasting.",
        "Adoption is gradual due to physical constraints, regulatory requirements, and high reliance on skilled manual labor.",
    ],
    "Installation, Maintenance, and Repair": [
        "AI assists maintenance and repair work through predictive diagnostics, sensor-based monitoring, and automated troubleshooting.",
        "Human technicians remain essential, with AI primarily enhancing efficiency and preventive maintenance.",
    ],
    "Building and Grounds Cleaning and Maintenance": [
        "AI adoption in cleaning and grounds maintenance is limited but includes scheduling optimization, robotics, and facility monitoring.",
        "Most tasks remain manual, with AI improving planning and supervision rather than direct task execution.",
    ],
}

# ── Technology lexicon for spaCy PhraseMatcher (entity extraction) ────────────
# Phrasing focuses on AI *outcomes* and *adoption modes* (per assignment rubric:
# automation, augmentation, workflow redesign, cost reduction, etc.)
TECH_LEXICON = [
    # Automation
    "task automation", "process automation", "intelligent automation",
    "robotic process automation", "end-to-end automation",
    "autonomous systems", "human-in-the-loop automation", "labor substitution",
    # Augmentation
    "human-AI augmentation", "decision augmentation", "cognitive augmentation",
    "AI copilots", "assistive AI", "augmented decision-making",
    "productivity enhancement",
    # Workflow redesign
    "workflow redesign", "process re-engineering", "AI-native workflows",
    "pipeline optimization", "operational transformation",
    "intelligent orchestration", "end-to-end workflow optimization",
    # Cost reduction
    "cost reduction", "operational efficiency", "margin improvement",
    "scalability improvements", "resource optimization", "productivity gains",
    "throughput optimization",
    # Predictive / decision
    "predictive analytics", "decision intelligence", "real-time analytics",
    "optimization engines", "forecasting accuracy improvements",
    "risk scoring", "decision automation",
    # Personalization / experience
    "personalization at scale", "intelligent recommendation", "adaptive systems",
    "dynamic content generation", "conversational interfaces",
    "context-aware systems", "customer experience optimization",
    # Risk / governance
    "risk mitigation", "error reduction", "compliance monitoring",
    "model governance", "explainable AI", "bias detection", "auditability",
    # Workforce
    "workforce transformation", "job displacement", "skill shift",
    "reskilling and upskilling", "role redefinition", "organizational redesign",
    # Innovation
    "product innovation", "AI-enabled products", "new business models",
    "service innovation", "time-to-market reduction", "continuous improvement loops",
    # Adoption signal
    "uncertain impact", "mixed outcomes", "uneven adoption",
    "partial automation", "adoption barriers",
]

# ── Adoption-driver categories (regex patterns for inference-stage matching) ──
ADOPTION_DRIVER_PATTERNS = {
    "Efficiency / Productivity": [
        r"\befficiency\b", r"\bproductivity\b", r"\boperational efficiency\b",
        r"\bthroughput\b", r"\boutput\b", r"\bstreamline\b", r"\boptimiz",
    ],
    "Cost Reduction": [
        r"\bcost reduction\b", r"\bcost savings?\b", r"\blower costs?\b",
        r"\breduce expenses?\b", r"\bmargin improvement\b", r"\bsave money\b",
    ],
    "Automation / Task Replacement": [
        r"\bautomation\b", r"\bautomate\b", r"\bprocess automation\b",
        r"\brobotic process automation\b", r"\brpa\b", r"\btask automation\b",
        r"\bautonomous systems?\b",
    ],
    "Decision Support / Better Insights": [
        r"\bpredictive analytics\b", r"\bdecision intelligence\b", r"\bdecision support\b",
        r"\binsight(s)?\b", r"\bforecast(ing)?\b", r"\brecommendation(s)?\b",
        r"\bbetter decisions?\b", r"\bdata-driven\b",
    ],
    "Personalization / Customer Experience": [
        r"\bpersonalization\b", r"\bpersonalized\b", r"\bcustomer experience\b",
        r"\bcx\b", r"\bcustomer support\b", r"\bservice innovation\b",
        r"\bpersonalization at scale\b",
    ],
    "Innovation / New Business Models": [
        r"\binnovation\b", r"\bproduct innovation\b", r"\bnew business models?\b",
        r"\bnew revenue\b", r"\bmarket expansion\b", r"\bgrowth\b",
    ],
    "Workflow Redesign / Augmentation": [
        r"\bworkflow\b", r"\bworkflow redesign\b", r"\baugmented\b", r"\baugmentation\b",
        r"\bcopilot(s)?\b", r"\bassistant\b", r"\bhuman-in-the-loop\b",
        r"\bdecision augmentation\b", r"\bworkforce transformation\b",
    ],
    "Governance / Compliance / Auditability": [
        r"\bgovernance\b", r"\bmodel governance\b", r"\bcompliance\b",
        r"\bauditability\b", r"\baudit trail\b", r"\bregulatory\b",
        r"\bexplainable ai\b", r"\bexplainability\b", r"\bmonitoring\b",
    ],
    "Privacy / Security / Risk": [
        r"\bprivacy\b", r"\bsecurity\b", r"\bcybersecurity\b",
        r"\brisk mitigation\b", r"\brisk scoring\b", r"\bsurveillance\b",
        r"\bdata leakage\b", r"\bdata protection\b",
    ],
    "Bias / Fairness / Trust": [
        r"\bbias\b", r"\bbias detection\b", r"\bfairness\b",
        r"\btrust\b", r"\btrustworthy\b", r"\bhallucination(s)?\b",
        r"\breliability\b", r"\baccuracy\b",
    ],
    "Job Displacement / Labor Concerns": [
        r"\bjob displacement\b", r"\bjob loss(es)?\b", r"\blayoffs?\b",
        r"\breplace workers?\b", r"\bworker displacement\b", r"\blabor concerns?\b",
        r"\bdeskilling\b",
    ],
    "Adoption Barriers / Integration Challenges": [
        r"\badoption barriers?\b", r"\bintegration\b", r"\bimplementation\b",
        r"\bdeployment challenges?\b", r"\blegacy systems?\b", r"\bdata quality\b",
        r"\bchange management\b", r"\borganizational resistance\b",
    ],
    "Training / Reskilling": [
        r"\breskilling\b", r"\bupskilling\b", r"\btraining\b",
        r"\bworkforce training\b", r"\bemployee education\b",
    ],
}

# ── Sentiment label mapping ───────────────────────────────────────────────────
SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive", 3: "unclear"}
SENTIMENT_LABEL_TO_ID = {v: k for k, v in SENTIMENT_LABELS.items()}

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
