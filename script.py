import os, sys, re, heapq, openai, json, base64
import numpy as np
from sklearn import svm
from pathlib import Path
from langchain_core.output_parsers.json import JsonOutputParser
from mimetypes import guess_type

openai.api_key = "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"
model_name = "gpt-4o"

class Node:
    def __init__(self, state, year, g):
        self.state = state
        self.year = year
        self.g = g
    def __lt__(self, other):
        return self.g < other.g

def heuristic(state, goal):
    return len(goal.get("skills", set()) - state.get("skills", set()))

def expand(state):
    remain = state["goal"].get("skills", set()) - state.get("skills", set())
    return [{
        "skills": state.get("skills", set()) | {s},
        "goal": state["goal"]
    } for s in list(remain)][:3]

def astar(start_state, years):
    q = [(0, Node(start_state, 0, 0))]
    seen = set()
    while q:
        f, node = heapq.heappop(q)
        key = (frozenset(node.state.get("skills", set())), node.year)
        if key in seen:
            continue
        seen.add(key)
        if node.year >= years:
            return node.state
        for ns in expand(node.state):
            g = node.g + 1
            h = heuristic(ns, ns["goal"])
            heapq.heappush(q, (g + h, Node(ns, node.year + 1, g)))
    return start_state

class ArchetypeClassifier:
    def __init__(self):
        self.clf = svm.SVC(probability=True)
        self.ready = False
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.ready = True
    def predict(self, x):
        return self.clf.predict([x])[0] if self.ready else "generalist"

classifier = ArchetypeClassifier()

def build_messages(resume, goal, years, detail, image_path=None):
    num_weeks = int(years * 12 * 4)

    sys_msg = (
        "You are CareerCompass, an AI career-planning agent. "
        "You may also receive a transcript image—extract GPA, key coursework, and academic signals. "
        "Produce pure JSON with keys: 'summary','roadmap','leetcode_schedule','recommended_jobs'. "
        f"Make this a {detail} plan. "
        "1) 'roadmap': break into monthly segments (if timeframe >2 months) or weekly segments (if <=2 months). "
        "   Each segment has 'period','focus','actions','projects','networking','prompt_engineering_tips','resources'. "
        "   Do NOT include LeetCode problems here. "
        f"2) 'leetcode_schedule': break the entire timeframe into {num_weeks} weekly segments. "
        "   Each weekly segment has 'period' (e.g. 'Week 1'), 'topics', and 'questions': list of at least 3 {title,url}. "
        "   Cover core algorithm topics (arrays, strings, DP, trees, graphs) in progressive order. "
        "3) 'recommended_jobs': list at least 5 job titles aligned with the goal."
    )

    user_content = [
        {"type": "text", "text": f"Resume:\n{resume}\nGoal role: {goal}\nTimeframe years: {years}\nTimeframe weeks: {num_weeks}"}
    ]

    if image_path:
        mime, _ = guess_type(image_path)
        if mime is None:
            mime = "image/png"
        image_data = Path(image_path).read_bytes()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{base64_image}",
                "detail": "auto"
            }
        })

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_content}
    ]

def generate_plan(resume, goal, years, detail, image_path=None):
    msgs = build_messages(resume, goal, years, detail, image_path)
    response = openai.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=0.7,
        max_tokens=2048
    )
    return response.choices[0].message.content.strip()

def flatten(obj, indent=0):
    spacer = "  " * indent
    out = ""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out += f"{spacer}{k}:\n"
                out += flatten(v, indent + 1)
            else:
                out += f"{spacer}{k}: {v}\n"
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                out += flatten(item, indent)
            else:
                out += f"{spacer}- {item}\n"
    else:
        out += f"{spacer}{obj}\n"
    return out

def main():
    raw = ' '.join(sys.argv[1:])
    m = re.search(
        r'plan\s+for\s+(?P<goal>.+?)\s+in\s+(?P<num>\d+)\s*(?P<unit>years?|months?)(?:\s+(?P<detail>basic|moderate|detailed|comprehensive))?',
        raw, re.IGNORECASE
    )
    if not m:
        print("Usage: python script.py plan for <goal> in <N> [years|months] [basic|moderate|detailed|comprehensive] [--image path]")
        return

    goal = m.group('goal')
    num = int(m.group('num'))
    unit = m.group('unit').lower()
    detail = m.group('detail') or 'moderate'
    years = num / 12 if unit.startswith('month') else num
    resume = sys.stdin.read()

    image_path = None
    if '--image' in sys.argv:
        idx = sys.argv.index('--image')
        if idx + 1 < len(sys.argv):
            image_path = sys.argv[idx + 1]

    plan_text = generate_plan(resume, goal, years, detail, image_path)
    parser = JsonOutputParser()

    try:
        plan_dict = parser.parse(plan_text)
    except Exception:
        print("Could not parse JSON. Writing raw cleaned text instead.")
        cleaned = re.sub(r"```json|```", "", plan_text).strip()
        with open("output.txt", "w") as f:
            f.write(cleaned + "\n")
        sys.exit(0)

    plain = flatten(plan_dict)
    with open("output.txt", "w") as f:
        f.write(plain)

    print("Wrote plain-text plan to output.txt.")

if __name__ == "__main__":
    main()