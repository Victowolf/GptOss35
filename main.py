# main.py
import re
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vllm import LLM, SamplingParams

# --------------------------------------------------------------------
# Load GPT-OSS-20B via vLLM
# --------------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-20b"

llm = LLM(
    model=MODEL_NAME,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,   # important for 35GB MIG
    max_model_len=2048,
    enforce_eager=True            # controls KV cache
)

# --------------------------------------------------------------------
# FastAPI Application
# --------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Harmony Prompt Template (UNCHANGED)
# --------------------------------------------------------------------
HARMONY_TEMPLATE = """<|start|>system<|message|>
{system_msg}
<|end|>

<|start|>developer<|message|>
{developer_msg}
<|end|>

<|start|>user<|message|>
{user_msg}
<|end|>

<|start|>assistant
<|channel|>final<|message|>
"""

def build_prompt(system, developer, user):
    return HARMONY_TEMPLATE.format(
        system_msg=system.strip(),
        developer_msg=developer.strip(),
        user_msg=user.strip(),
    )

# --------------------------------------------------------------------
# Extract ONLY final channel
# --------------------------------------------------------------------
FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*)",
    re.S
)

@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):

    system_msg = """
        You are a browser-native AI Concierge for ET.
        Understand user intent from context and memory.
        Map needs to relevant ET products/services.
        Be concise, non-intrusive, and action-focused.
    """
    developer_msg = """
        Return only the final answer as plain text.

        Rules:
        - No JSON
        - No markdown formatting
        - No extra tags or system messages
        - No reasoning or analysis

        Be clear, concise, and complete.
    """

    harmony_prompt = build_prompt(system_msg, developer_msg, prompt)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024   # safe for 35GB
    )

    outputs = llm.generate([harmony_prompt], sampling_params)
    raw = outputs[0].outputs[0].text

    # Extract final channel
    m = FINAL_RE.search(raw)
    if not m:
        cleaned = re.sub(r"<\|.*?\|>", "", raw).strip()
        return JSONResponse({"response": cleaned})

    final_text = m.group(1).strip()

    # Remove any rogue analysis
    final_text = re.sub(r"<\|channel\|>analysis.*", "", final_text, flags=re.S).strip()

    return JSONResponse({"response": final_text})

# --------------------------------------------------------------------
# Health Endpoint
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS vLLM Server Running"}