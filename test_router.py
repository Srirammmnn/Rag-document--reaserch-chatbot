import os
from dotenv import load_dotenv
load_dotenv()

from agent import llm_based_route

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
    ("what is the capital of france?",    "llm"),
    ("explain what RAG means in AI",      "llm"),
]

passed = 0
for q, expected in tests:
    got = llm_based_route(q)
    ok = "OK" if got == expected else "FAIL"
    print(f"[{ok}] [{str(got):>6}]  {q} (Expected: {expected})")
    if got == expected: passed += 1

print(f"\nAccuracy: {passed}/{len(tests)} = {passed/len(tests)*100:.0f}%")
