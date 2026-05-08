"""
VidGist 🎬 — paste any YouTube URL, get an AI summary in seconds.

Uses Gemini's native YouTube understanding (no transcript scraping required).
For long videos (over ~50 minutes), automatically chunks into smaller segments
and merges the summaries.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import streamlit as st
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "gemini-flash-lite-latest"
CHUNK_SIZE_SECONDS = 30 * 60           # 30-minute chunks
MAX_VIDEO_SECONDS = 2 * 60 * 60        # hard cap at 2 hours
RATE_LIMIT_PAUSE = 4                   # seconds between chunk calls

SAMPLE_VIDEOS = [
    {
        "title": "Steve Jobs Stanford Speech",
        "duration": "15 min",
        "emoji": "🎓",
        "url": "https://www.youtube.com/watch?v=UF8uR6Z6KLc",
    },
    {
        "title": "Master Procrastinator (TED)",
        "duration": "14 min",
        "emoji": "💡",
        "url": "https://www.youtube.com/watch?v=arj7oStGLkU",
    },
    {
        "title": "Start Studying For Real",
        "duration": "5 min",
        "emoji": "📚",
        "url": "https://www.youtube.com/watch?v=IlU-zDU6aQ0",
    },
]

YOUTUBE_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.|m\.)?"
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)

PROMPT = """You are a world-class video summarizer. Watch the provided video segment carefully and produce a thorough, information-rich summary.

Return your response in exactly this Markdown structure:

## TL;DR
A 4-5 sentence summary capturing the essence of this segment, including the speaker (if identifiable), the topic, the main argument, and the conclusion.

## 🎯 Key Takeaways
- 8 to 12 bullet points covering the most important ideas, insights, arguments, examples, and context.
- Be specific and concrete. Use actual numbers, names, examples, and details from the video — never generic platitudes.
- Each bullet should be substantive (1-2 sentences) and stand on its own.
- Cover both *what* was said and *why it matters*.

## 💬 Notable Quotes
- 2 to 4 direct, memorable quotes from the speaker(s).
- Format each as: > "Quote text here." — Speaker (if identifiable)
- Pick quotes that are quotable on their own, not generic statements.

## ⏱️ Worth Watching
- 3 to 5 timestamps with what's noteworthy at each (use mm:ss or hh:mm:ss format).
- Format: **0:42** — what happens at this point.
- Pick the most insightful, surprising, or quotable moments — not the boring intro.

## ✅ Action Items
- 3 to 5 practical things the viewer can apply, try, or remember.
- Be specific. "Read more books" is bad. "Read 'Atomic Habits' chapter 3 on habit stacking" is good.
- If the video is purely informational with no actionable advice, write "(none — this is an informational video)".

Do not include any preamble. Start directly with the `## TL;DR` heading."""

MERGE_PROMPT = """Below are summaries of consecutive segments of a single long video. Combine them into ONE cohesive summary that reads as if it covered the whole video, using the same Markdown structure (## TL;DR / ## 🎯 Key Takeaways / ## 💬 Notable Quotes / ## ⏱️ Worth Watching / ## ✅ Action Items). Deduplicate overlapping points, preserve the most important 8-12 takeaways overall, and keep all the most memorable quotes. For the timestamps, adjust them to reflect their position in the FULL video (e.g., a 5:00 timestamp in segment 2 of a video chunked at 30-min boundaries should become 35:00). Do not include any preamble — start directly with `## TL;DR`."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    video_id: str
    canonical_url: str


def parse_youtube_url(url: str) -> VideoInfo | None:
    """Return a VideoInfo if the URL parses as a YouTube link, else None."""
    if not url:
        return None
    m = YOUTUBE_URL_RE.match(url.strip())
    if not m:
        return None
    vid = m.group(1)
    return VideoInfo(video_id=vid, canonical_url=f"https://www.youtube.com/watch?v={vid}")


def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key.strip())


# Deterministic config: same input → same (or near-identical) output
GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.0,        # most deterministic
    top_p=0.95,
    top_k=1,                # always pick the most likely next token
    max_output_tokens=4096,  # roomy enough for our richer summaries
)


def summarize_segment(
    client: genai.Client,
    url: str,
    start_s: int | None = None,
    end_s: int | None = None,
) -> str:
    """Call Gemini for a single video (or video segment if offsets provided)."""
    video_metadata = None
    if start_s is not None and end_s is not None:
        video_metadata = types.VideoMetadata(
            start_offset=f"{start_s}s",
            end_offset=f"{end_s}s",
        )

    file_part = types.Part(
        file_data=types.FileData(file_uri=url, mime_type="video/*"),
        video_metadata=video_metadata,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=types.Content(parts=[file_part, types.Part(text=PROMPT)]),
        config=GENERATION_CONFIG,
    )
    return (response.text or "").strip()


def merge_summaries(client: genai.Client, summaries: list[str]) -> str:
    joined = "\n\n---\n\n".join(
        f"### Segment {i + 1}\n\n{s}" for i, s in enumerate(summaries)
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=f"{MERGE_PROMPT}\n\n{joined}",
        config=GENERATION_CONFIG,
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def setup_page() -> None:
    st.set_page_config(
        page_title="VidGist — YouTube video summaries in seconds",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(
                    ellipse 60% 30% at 50% 0%,
                    rgba(20, 184, 166, 0.10),
                    transparent 70%
                ), #0a0e0d;
            }
            .block-container {
                padding-top: 2rem;
                max-width: 760px;
            }
            h1, h2, h3 { letter-spacing: -0.02em; }
            .vg-hero { text-align: center; margin-bottom: 8px; }
            .vg-hero-title {
                font-size: clamp(2rem, 6vw, 3rem);
                font-weight: 800;
                background: linear-gradient(135deg, #ffffff 25%, #5eead4 60%, #14b8a6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 6px;
            }
            .vg-hero-sub {
                color: #94a3b8;
                font-size: 1rem;
                max-width: 520px;
                margin: 0 auto 20px;
                line-height: 1.5;
            }
            .vg-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(20, 184, 166, 0.10);
                border: 1px solid rgba(20, 184, 166, 0.25);
                color: #5eead4;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.04em;
                padding: 4px 12px;
                border-radius: 999px;
                margin-bottom: 16px;
                text-transform: uppercase;
            }
            .stButton > button {
                background: linear-gradient(135deg, #14b8a6, #0ea5e9);
                color: white;
                border: none;
                font-weight: 600;
                border-radius: 10px;
                padding: 0.55rem 1.2rem;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 24px rgba(20, 184, 166, 0.35);
                color: white;
            }
            .vg-footer {
                text-align: center;
                margin-top: 48px;
                padding: 20px;
                color: #64748b;
                font-size: 13px;
                border-top: 1px solid rgba(255,255,255,0.06);
            }
            .vg-footer a {
                color: #14b8a6;
                font-weight: 500;
                text-decoration: none;
            }
            .vg-footer a:hover {
                color: #5eead4;
                text-decoration: underline;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="vg-hero">
            <div class="vg-badge">🔒 Free · BYOK · No transcript scraping</div>
            <div class="vg-hero-title">VidGist 🎬</div>
            <div class="vg-hero-sub">
                Paste any YouTube URL. Get a TL;DR, key takeaways, and action items
                in seconds — powered by Gemini's native video understanding.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="vg-footer">
            Built by Rahul ·
            <a href="https://www.linkedin.com/in/rahul-reddy-avula-37572b328/" target="_blank">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_key_input() -> str:
    expanded = not bool(st.session_state.get("api_key"))
    with st.expander("🔑 Gemini API key (required, free, 60-second setup)", expanded=expanded):
        st.markdown(
            "VidGist runs on **your own** free Gemini API key — your videos and key never touch any server I control.\n\n"
            "1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)\n"
            "2. Click **Create API key** (free, no credit card)\n"
            "3. Paste it below"
        )
        key = st.text_input(
            "Paste your Gemini API key",
            type="password",
            value=st.session_state.get("api_key", ""),
            placeholder="AIzaSy...",
            label_visibility="collapsed",
        )
        if key and key != st.session_state.get("api_key", ""):
            st.session_state["api_key"] = key.strip()
            st.success("✅ Key saved for this session.")
    return st.session_state.get("api_key", "")


def render_samples() -> str | None:
    st.markdown("**Or try a sample:**")
    cols = st.columns(len(SAMPLE_VIDEOS))
    chosen: str | None = None
    for col, sample in zip(cols, SAMPLE_VIDEOS):
        with col:
            label = f"{sample['emoji']} {sample['title']}\n_{sample['duration']}_"
            if st.button(label, key=f"sample-{sample['url']}", use_container_width=True):
                chosen = sample["url"]
    return chosen


def render_summary(summary: str, info: VideoInfo, was_chunked: bool) -> None:
    st.video(info.canonical_url)
    if was_chunked:
        st.info("🧩 This was a long video — VidGist split it into 30-minute chunks and merged the summaries.")
    st.markdown(summary)
    st.download_button(
        "💾 Download summary as Markdown",
        data=summary,
        file_name=f"vidgist-{info.video_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def run_summary(client: genai.Client, info: VideoInfo, long_video_mode: bool) -> None:
    """Drive the full summarisation flow + render results."""
    if long_video_mode:
        # Chunked path: 0–30, 30–60, 60–90, 90–120 min (capped at 2h)
        chunks: list[tuple[int, int]] = []
        t = 0
        while t < MAX_VIDEO_SECONDS:
            chunks.append((t, t + CHUNK_SIZE_SECONDS))
            t += CHUNK_SIZE_SECONDS

        progress = st.progress(0.0, text="Starting chunked summarisation…")
        segments: list[str] = []
        successful_chunks = 0
        # Rough per-chunk estimate (Gemini call + rate-limit pause + buffer)
        SECONDS_PER_CHUNK = 45

        for i, (start_s, end_s) in enumerate(chunks):
            remaining_chunks = len(chunks) - i
            eta_sec = remaining_chunks * SECONDS_PER_CHUNK + 15  # +15 for merge step
            eta_min = eta_sec // 60
            eta_text = f"~{eta_min} min remaining" if eta_min >= 1 else f"~{eta_sec}s remaining"
            progress.progress(
                i / (len(chunks) + 1),
                text=f"Summarising minute {start_s // 60}–{end_s // 60}… ({eta_text})",
            )
            try:
                seg = summarize_segment(client, info.canonical_url, start_s, end_s)
                if seg:
                    segments.append(seg)
                    successful_chunks += 1
            except Exception as exc:
                msg = str(exc)
                # Past the end of the video, Gemini errors out — that's our signal to stop.
                if "INVALID_ARGUMENT" in msg or "out of range" in msg.lower() or "empty video" in msg.lower():
                    break
                st.warning(f"Chunk {i + 1} ({start_s // 60}–{end_s // 60} min) failed: {msg[:200]}")
            time.sleep(RATE_LIMIT_PAUSE)

        if not segments:
            st.error("Couldn't summarise any part of this video. Please double-check the URL.")
            return

        progress.progress(0.95, text="Merging chunk summaries…")
        final = merge_summaries(client, segments) if len(segments) > 1 else segments[0]
        progress.progress(1.0, text="Done!")
        progress.empty()

        st.success(f"Combined {successful_chunks} segment summaries into one.")
        render_summary(final, info, was_chunked=True)
    else:
        with st.spinner("Watching the video and writing your summary…"):
            summary = summarize_segment(client, info.canonical_url)
        if not summary:
            st.error("Gemini returned an empty summary. Please try again or enable chunked mode.")
            return
        render_summary(summary, info, was_chunked=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_page()
    render_hero()

    api_key = render_api_key_input()
    sample_picked = render_samples()

    # If a sample was clicked, treat it as the user's URL for this run.
    typed_url = st.text_input(
        "Paste a YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_input",
    )
    url_value = sample_picked or typed_url

    long_video_mode = st.checkbox(
        "🧩 Chunked mode (for videos longer than ~50 minutes)",
        value=False,
        help="Splits the video into 30-minute chunks and merges the summaries. "
             "Use this for podcasts or lectures over ~50 minutes. Takes longer.",
    )

    go = st.button(
        "✨ Summarize video",
        type="primary",
        use_container_width=True,
        disabled=not (url_value and api_key),
    )

    # Estimated processing time hint
    if long_video_mode:
        st.caption(
            "⏱️ **Expected time:** 2 min (1-hour video) · 3 min (90-min video) · 4 min (2-hour video). "
            "Each 30-min chunk takes ~30–45s plus a 4s rate-limit pause + a final merge step."
        )
    else:
        st.caption(
            "⏱️ **Expected time:** ~20s (under 5 min video) · ~30–60s (5–20 min) · ~60–90s (20–50 min). "
            "Tick chunked mode above for videos longer than 50 minutes."
        )

    if not api_key and url_value:
        st.warning("Please paste your Gemini API key above to continue.")
    if not url_value and api_key:
        st.info("Paste a YouTube URL or click a sample video to try it out.")

    # Auto-run when a sample button is pressed (only if API key is set)
    should_run = go or (sample_picked and api_key)

    if should_run and url_value and api_key:
        info = parse_youtube_url(url_value)
        if not info:
            st.error("That doesn't look like a YouTube URL. Please paste a link from youtube.com or youtu.be.")
        else:
            try:
                client = make_client(api_key)
            except Exception as exc:
                st.error(f"Couldn't initialise Gemini client: {exc}")
                render_footer()
                return

            try:
                run_summary(client, info, long_video_mode)
            except Exception as exc:
                msg = str(exc)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    st.error(
                        "🚦 **Your Gemini API key has hit its quota limit.**\n\n"
                        "**What to do:**\n"
                        "- Wait ~1 minute and retry (per-minute limit reset), or\n"
                        "- Wait until tomorrow if it's a daily limit (resets midnight Pacific), or\n"
                        "- **Generate a new free key** at "
                        "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and paste it above.\n\n"
                        "Free tier limits: 15 requests/min, 1,500 requests/day per project. "
                        "Each chunk uses 1 request — so a 2-hour video = 5 requests."
                    )
                elif "API_KEY" in msg.upper() or "401" in msg or "403" in msg:
                    st.error(
                        "🔑 **Your API key was rejected.** It may be wrong, expired, or revoked.\n\n"
                        "Generate a new free key at "
                        "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and paste it above."
                    )
                elif "video" in msg.lower() and ("long" in msg.lower() or "duration" in msg.lower()):
                    st.error(
                        "📏 **This video is too long for a single Gemini call.** "
                        "Tick **🧩 Chunked mode** above and try again."
                    )
                else:
                    st.error(f"Something went wrong: {msg[:300]}")

    render_footer()


if __name__ == "__main__":
    main()
