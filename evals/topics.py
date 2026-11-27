"""The fixed evaluation set: 12 topics.

For every topic we wrote down, BEFORE running the system:
  aspects  — five things a good report on the topic should cover (used for the Coverage metric);
  off      — a question the report cannot possibly answer (used for "I don't know" handling).
"""

from __future__ import annotations

TOPICS: list[dict[str, object]] = [
    {
        "topic": "State of electric two-wheelers in India",
        "aspects": [
            "market size or sales growth",
            "leading manufacturers",
            "government subsidies or policy",
            "charging or battery-swapping infrastructure",
            "barriers such as cost, range or safety",
        ],
        "off": "What is the boiling point of tungsten?",
    },
    {
        "topic": "Remote work and its effect on employee productivity",
        "aspects": [
            "evidence on productivity changes",
            "hybrid work models",
            "employee well-being or burnout",
            "collaboration and communication challenges",
            "employer policies such as return-to-office",
        ],
        "off": "Who painted the ceiling of the Sistine Chapel?",
    },
    {
        "topic": "Adoption of UPI digital payments in India",
        "aspects": [
            "transaction volume or growth",
            "role of NPCI or the regulator",
            "major payment apps",
            "merchant or rural adoption",
            "fraud, security or international expansion",
        ],
        "off": "How many moons does Jupiter have?",
    },
    {
        "topic": "Green hydrogen as a clean energy source",
        "aspects": [
            "how it is produced (electrolysis)",
            "production cost compared with grey hydrogen",
            "industrial or transport uses",
            "government strategies or subsidies",
            "storage, transport or infrastructure challenges",
        ],
        "off": "Who won the first season of the Indian Premier League?",
    },
    {
        "topic": "Use of generative AI in software development",
        "aspects": [
            "AI coding assistants and tools",
            "measured productivity impact",
            "code quality or security risks",
            "effect on developer jobs or skills",
            "adoption by companies",
        ],
        "off": "What is the capital city of Mongolia?",
    },
    {
        "topic": "Microplastics and human health",
        "aspects": [
            "sources of microplastics",
            "how humans are exposed",
            "evidence of health effects",
            "where they have been found in the human body",
            "regulation or ways to reduce exposure",
        ],
        "off": "Which company makes the PlayStation console?",
    },
    {
        "topic": "Global semiconductor supply chain after the 2020-2022 chip shortage",
        "aspects": [
            "causes of the shortage",
            "geographic concentration such as Taiwan or TSMC",
            "government subsidy programmes such as the CHIPS Act",
            "new fabrication plants or investment",
            "geopolitical or export-control tensions",
        ],
        "off": "How long is an Olympic swimming pool?",
    },
    {
        "topic": "Telemedicine adoption after the COVID-19 pandemic",
        "aspects": [
            "usage trends after the pandemic",
            "benefits for access to care",
            "regulation or reimbursement",
            "technology or digital-divide barriers",
            "quality of care or patient satisfaction",
        ],
        "off": "What year did the Berlin Wall fall?",
    },
    {
        "topic": "Lab-grown (cultivated) meat industry",
        "aspects": [
            "how cultivated meat is made",
            "regulatory approvals",
            "production cost or scaling",
            "consumer acceptance",
            "environmental impact",
        ],
        "off": "Who wrote the novel Pride and Prejudice?",
    },
    {
        "topic": "Central bank digital currencies (CBDCs)",
        "aspects": [
            "what a CBDC is and why central banks want one",
            "pilots or launches such as e-CNY or eNaira",
            "privacy concerns",
            "effect on commercial banks or financial stability",
            "cross-border payments",
        ],
        "off": "What is the chemical symbol for gold?",
    },
    {
        "topic": "Solid-state batteries for electric vehicles",
        "aspects": [
            "how they differ from lithium-ion batteries",
            "advantages such as energy density or safety",
            "companies developing them",
            "manufacturing or technical challenges",
            "commercialisation timeline",
        ],
        "off": "Which planet is known as the Red Planet?",
    },
    {
        "topic": "Impact of social media on teenage mental health",
        "aspects": [
            "research evidence on anxiety or depression",
            "sleep or screen-time effects",
            "cyberbullying or body image",
            "positive effects such as social connection",
            "regulation, age limits or parental controls",
        ],
        "off": "What is the tallest mountain in Africa?",
    },
]


def slug(topic: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in topic.lower()).strip("-")[:50]
