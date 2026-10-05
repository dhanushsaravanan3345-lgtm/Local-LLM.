import asyncio
import json
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..errors import AppError
from ..services import llm_service as llm
from ..services import memory_service as mem
from ..services import rag_service as rag

router = APIRouter(prefix="/api")
log = logging.getLogger("chat")


def get_time_or_date_answer(text: str) -> str | None:
    """Return real India time/date for time/date questions."""
    q = text.lower().strip()

    time_words = [
        "current time",
        "what time is it",
        "what is the time",
        "time right now",
        "time now",
        "right now",
    ]

    date_words = [
        "today's date",
        "what is today's date",
        "what is the date",
        "current date",
        "today date",
        "date today",
    ]

    asks_time = any(word in q for word in time_words)
    asks_date = any(word in q for word in date_words)

    if not asks_time and not asks_date:
        return None

    # India Standard Time
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    if asks_time:
        return (
            f"The current time in India (IST) is "
            f"{now.strftime('%I:%M:%S %p')}."
        )

    if asks_date:
        return (
            f"Today's date in India is "
            f"{now.strftime('%A, %d %B %Y')}."
        )

    return None


class ChatRequest(BaseModel):
    message: str = Field("", max_length=200_000)
    conversation_id: int | None = None
    use_documents: bool = True
    regenerate: bool = False


async def _prepare(req: ChatRequest):
    """Validate, persist the user turn, build the message list."""
    text = (req.message or "").strip()

    if req.regenerate:
        if (
            not req.conversation_id
            or not mem.conversation_exists(req.conversation_id)
        ):
            raise AppError(
                "not_found",
                "Conversation not found.",
                404,
            )

        cid = req.conversation_id
        text = mem.pop_last_for_regen(cid)

    else:
        if not text:
            raise AppError(
                "empty_prompt",
                "Please type a message first.",
                400,
            )

        if len(text) > settings.max_prompt_chars:
            raise AppError(
                "prompt_too_large",
                f"Your message is too long ({len(text):,} characters). "
                f"The limit is {settings.max_prompt_chars:,}.",
                413,
            )

        cid = req.conversation_id

        if cid is not None and not mem.conversation_exists(cid):
            raise AppError(
                "not_found",
                "Conversation not found.",
                404,
            )

        cid = cid or mem.create_conversation()
        mem.add_message(cid, "user", text)

    system = mem.system_prompt()
    sources: list[str] = []

    if req.use_documents and rag.has_documents():
        try:
            hits = await rag.retrieve(text)

        except AppError as e:
            log.warning("RAG skipped: %s", e.code)
            hits = []

        if hits:
            sources = list(
                dict.fromkeys(h["name"] for h in hits)
            )

            ctx = "\n\n".join(
                f"[Source: {h['name']}]\n{h['text']}"
                for h in hits
            )

            system += (
                "\n\nUse the following excerpts from the user's "
                "local documents when relevant, and mention the "
                "source name. If they do not contain the answer, "
                "say so.\n\n"
                + ctx
            )

    messages = [
        {
            "role": "system",
            "content": system,
        }
    ] + mem.history_for_model(cid)

    return cid, messages, sources


def _sse(obj: dict) -> str:
    return (
        f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
    )


# ---------------------------------------------------------
# NORMAL CHAT ENDPOINT
# ---------------------------------------------------------

@router.post("/chat")
async def chat(req: ChatRequest):
    cid, messages, sources = await _prepare(req)

    t0 = time.perf_counter()

    # Use real computer time for time/date questions.
    realtime_answer = get_time_or_date_answer(req.message)

    if realtime_answer:
        text = realtime_answer

        stats = {
            "source": "system_clock",
            "tokens": 0,
        }

    else:
        text, stats = await llm.chat_once(
            messages,
            settings.model,
        )

    mem.add_message(
        cid,
        "assistant",
        text,
        sources,
    )

    return {
        "conversation_id": cid,
        "answer": text,
        "sources": sources,
        "stats": {
            **stats,
            "total_s": round(
                time.perf_counter() - t0,
                2,
            ),
        },
    }


# ---------------------------------------------------------
# STREAMING CHAT ENDPOINT
# ---------------------------------------------------------

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):

    cid, messages, sources = await _prepare(req)

    realtime_answer = get_time_or_date_answer(
        req.message
    )

    async def gen():

        parts: list[str] = []
        msg_id = None

        t0 = time.perf_counter()
        first = None
        stats: dict = {}

        yield _sse({
            "type": "meta",
            "conversation_id": cid,
            "sources": sources,
        })

        try:

            # -------------------------------------------------
            # REAL TIME / DATE ANSWER
            # -------------------------------------------------

            if realtime_answer:

                mem.add_message(
                    cid,
                    "assistant",
                    realtime_answer,
                    sources,
                )

                yield _sse({
                    "type": "token",
                    "text": realtime_answer,
                })

                yield _sse({
                    "type": "done",
                    "stats": {
                        "source": "system_clock",
                        "tokens": 0,
                        "first_token_s": 0,
                        "total_s": round(
                            time.perf_counter() - t0,
                            2,
                        ),
                    },
                })

                return

            # -------------------------------------------------
            # NORMAL AI RESPONSE
            # -------------------------------------------------

            async for ev in llm.stream_chat(
                messages,
                settings.model,
            ):

                if "token" in ev:

                    parts.append(ev["token"])

                    if first is None:

                        first = round(
                            time.perf_counter() - t0,
                            2,
                        )

                        msg_id = mem.add_message(
                            cid,
                            "assistant",
                            "".join(parts),
                            sources,
                        )

                    elif len(parts) % 20 == 0:

                        mem.update_message(
                            msg_id,
                            "".join(parts),
                        )

                    yield _sse({
                        "type": "token",
                        "text": ev["token"],
                    })

                elif ev.get("done"):

                    stats = ev["stats"]

            # Save final answer
            if msg_id:

                mem.update_message(
                    msg_id,
                    "".join(parts),
                )

            yield _sse({
                "type": "done",
                "stats": {
                    **stats,
                    "first_token_s": first,
                    "total_s": round(
                        time.perf_counter() - t0,
                        2,
                    ),
                },
            })

        except AppError as e:

            yield _sse({
                "type": "error",
                "code": e.code,
                "message": e.message,
            })

        except asyncio.CancelledError:

            raise

        except Exception:

            log.exception(
                "stream failed"
            )

            yield _sse({
                "type": "error",
                "code": "internal",
                "message": (
                    "Something went wrong while "
                    "generating."
                ),
            })

        finally:

            if msg_id:

                try:

                    mem.update_message(
                        msg_id,
                        "".join(parts),
                    )

                except Exception:

                    log.exception(
                        "could not save answer"
                    )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------
# CONVERSATIONS
# ---------------------------------------------------------

@router.get("/conversations")
async def conversations():

    return {
        "conversations": mem.list_conversations()
    }


@router.get("/conversations/{cid}")
async def conversation(cid: int):

    if not mem.conversation_exists(cid):

        raise AppError(
            "not_found",
            "Conversation not found.",
            404,
        )

    return {
        "id": cid,
        "messages": mem.get_messages(cid),
    }


@router.delete("/conversations/{cid}")
async def delete_conversation(cid: int):

    mem.delete_conversation(cid)

    return {
        "deleted": True
    }


@router.delete("/chat/history")
async def clear_history(
    conversation_id: int | None = None
):

    if conversation_id is None:

        mem.clear_all()

    else:

        mem.delete_conversation(
            conversation_id
        )

    return {
        "cleared": True
    }