import os
import json
import urllib.parse
from typing import List, Literal, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# ---------- Models ----------

class HistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None  # only used for tool messages


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryMessage] = []


class ChatResponse(BaseModel):
    reply: str


# ---------- Setup ----------

app = FastAPI(
    title="Utah Tourism Chat",
    description="LLM-based Utah tourism assistant using Crawl4AI as a web-scraping tool.",
    version="0.1.0",
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CRAWL4AI_BASE = os.environ.get("CRAWL4AI_BASE", "http://crawl4ai:11235")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ---------- Tool implementation ----------

async def crawl4ai_scrape(url: str) -> str:
    """
    Call Crawl4AI's /md/{url} endpoint to get Markdown for a page.
    Docs: /md returns LLM-ready markdown. :contentReference[oaicite:1]{index=1}
    """
    encoded = urllib.parse.quote(url, safe="")
    full_url = f"{CRAWL4AI_BASE}/md/{encoded}"

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.get(full_url)
        resp.raise_for_status()

        # Some versions return plain text, some JSON. Be tolerant.
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
            # Try a few likely fields; fall back to raw text
            return (
                data.get("markdown")
                or data.get("result", {}).get("markdown")
                or json.dumps(data)
            )
        else:
            return resp.text


# ---------- FastAPI route ----------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint:
    - Sends user + history to OpenAI with a `scrape_url` tool.
    - If model calls the tool, we call Crawl4AI and then do a second LLM call.
    """
    system_prompt = (
        "You are a friendly, precise Utah tourism assistant. "
        "You help users plan trips to Utah: national parks (Zion, Bryce, Arches, Canyonlands), "
        "state parks, hiking, camping, scenic drives, winter sports, hotels, and logistics.\n\n"
        "You have access to a `scrape_url` tool that fetches fresh web content via Crawl4AI. "
        "Use it when you need up-to-date details like fees, opening hours, "
        "road closures, or current conditions from official or reputable tourism sites.\n\n"
        "Always summarize and explain results in your own words rather than pasting raw markdown."
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Include prior conversation if provided
    for m in req.history:
        # Chat Completion API does not expect 'name' unless role='tool'
        msg = {"role": m.role, "content": m.content}
        if m.role == "tool" and m.name:
            msg["name"] = m.name
        messages.append(msg)

    # Current user turn
    messages.append({"role": "user", "content": req.message})

    # Define the tool that the model can choose to call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scrape_url",
                "description": (
                    "Fetch markdown content of a webpage using Crawl4AI. "
                    "Use this for up-to-date Utah tourism info such as park pages, "
                    "official visitor sites, trail conditions, etc."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to crawl. Prefer official or reputable sites."
                        }
                    },
                    "required": ["url"]
                },
            },
        }
    ]

    # ---- First call: let the model decide whether to use the tool ----
    first = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    choice = first.choices[0]
    message = choice.message

    # If the model answered directly, just return that
    if not message.tool_calls:
        return ChatResponse(reply=message.content or "")

    # Otherwise, execute the tool(s), append results, and ask the model again
    for tool_call in message.tool_calls:
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")

        if fn_name == "scrape_url":
            url = args.get("url")
            if not url:
                tool_result = "Tool error: no URL provided."
            else:
                try:
                    tool_result = await crawl4ai_scrape(url)
                except Exception as e:
                    tool_result = f"Tool error while scraping {url}: {e}"

            # Add the tool call + tool result into the conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": fn_name,
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    # ---- Second call: model sees tool output and produces final answer ----
    second = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )

    final_msg = second.choices[0].message
    return ChatResponse(reply=final_msg.content or "")
