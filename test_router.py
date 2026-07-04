def _rule_based_route(question):
    q = question.lower().strip()
    rag_triggers = [
        "my ", " my ", "i am", "i'm", "my name", "my email", "my phone",
        "my skills", "my experience", "my education", "my certifications",
        "my projects", "my resume", "my work", "my job", "my company",
        "my address", "my profile", "about me", "who am i", "tell me about myself",
        "according to", "based on the document", "from the document",
        "in the document", "in the resume", "from the resume",
        "what does the document", "what is in the doc", "summarize the",
        "tell me about my", "list my", "what are my", "do i have",
        "show my", "give me my",
    ]
    if any(t in q for t in rag_triggers): return "rag"
    math_triggers = ["calculate", "compute", "how much is", "sum of", "product of",
                     "square root", "percentage of", "convert ", "solve for", "evaluate "]
    if any(t in q for t in math_triggers) and any(c.isdigit() for c in q): return "math"
    python_triggers = ["write a python", "python code", "run code", "execute code",
                       "write code", "write a script", "program that", "write a function"]
    if any(t in q for t in python_triggers): return "python"
    web_triggers = ["latest news", "news about", "current events", "what happened today",
                    "recent news", "who won the", "trending today", "live score"]
    if any(t in q for t in web_triggers): return "web"
    return None

tests = [
    ("what are my certifications?",       "rag"),
    ("what is my name?",                  "rag"),
    ("tell me about my skills",           "rag"),
    ("list my projects",                  "rag"),
    ("what is my work experience?",       "rag"),
    ("do i have any python skills?",      "rag"),
    ("what are my educational details?",  "rag"),
    ("give me my contact info",           "rag"),
    ("calculate 25 * 4 + 100",            "math"),
    ("write a python fibonacci function", "python"),
    ("latest news about AI today",        "web"),
    ("what is the capital of france?",    None),
    ("explain what RAG means in AI",      None),
]

passed = 0
for q, expected in tests:
    got = _rule_based_route(q)
    ok = "OK" if got == expected else "FAIL"
    print(f"[{ok}] [{str(got or 'LLM-fb'):>8}]  {q}")
    if got == expected: passed += 1

print(f"\nAccuracy: {passed}/{len(tests)} = {passed/len(tests)*100:.0f}%")
