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
MAX_VIDEO_SECONDS = 4 * 60 * 60        # hard cap at 4 hours (covers most podcasts)

# Gemini free tier: 1M tokens/min. A 30-min video chunk uses ~250-500K tokens.
# Pause 18s between chunks so we stay safely below the per-minute token budget.
RATE_LIMIT_PAUSE = 18

# When a chunk 429's, retry with these waits (auto-recover from transient TPM hits).
RETRY_BACKOFF_SECONDS = (30, 60)

SECONDS_PER_CHUNK_ESTIMATE = 65        # avg time per chunk (Gemini call + 18s pause)

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

PROMPT = """You are a world-class video summarizer. Watch the provided video segment carefully and produce a thorough, information-rich summary that someone could read instead of watching the video.

CRITICAL ACCURACY RULES (read these first):
- ONLY summarise what is actually in the video. Do NOT invent facts, names, numbers, quotes, or timestamps.
- If you cannot identify the speaker, write "Unknown" — do not guess based on the topic.
- If something isn't said or shown in the video, do NOT include it. Fewer accurate bullets > more fabricated ones.
- Quotes must be word-for-word from the audio, not paraphrased into pseudo-quotes.
- Timestamps must reflect actual moments in the video segment you watched.
- Do not state the exact total duration of the video in minutes — focus on the content.

Return your response in exactly this Markdown structure:

## 📌 At a Glance
- **Type:** [tutorial / lecture / interview / podcast / vlog / talk / news / review / explainer / documentary]
- **Topic:** [one-line topic in 8 words or less]
- **Speaker / Channel:** [name if identifiable, else "Unknown"]
- **Best for:** [one line on who would benefit most from watching]

## TL;DR
A detailed 8-10 sentence summary that someone can read instead of watching. Cover:
- Who is speaking and what's their credibility / context (if identifiable)
- The exact topic and the main question or problem the video answers
- The central argument or thesis being made
- The 3-5 most important supporting points, examples, or stories used
- Any concrete numbers, frameworks, or specific examples mentioned
- The conclusion, recommendation, or takeaway the speaker leaves you with
- Why this video is or isn't worth watching in full

Write it as flowing prose, not bullets. Be substantive — this should give the reader 80% of the value of watching.

## 🎯 Key Takeaways
- 10 to 14 bullet points covering the most important ideas, insights, arguments, examples, frameworks, and context.
- Be specific and concrete. Use actual numbers, names, products, books, studies, dates, and examples mentioned in the video — never generic platitudes.
- Each bullet should be substantive (1-2 sentences) and stand on its own.
- Cover both *what* was said and *why it matters* or *how it works*.

## 💬 Notable Quotes
- 3 to 5 direct, memorable quotes from the speaker(s).
- Format each as: > "Quote text here." — Speaker (if identifiable)
- Pick quotes that are quotable on their own, not generic statements.
- Preserve the exact wording.

## ⏱️ Worth Watching
- 4 to 6 timestamps with a one-line description of what's noteworthy at each (mm:ss or hh:mm:ss).
- Format: **0:42** — what happens at this point.
- Pick the most insightful, surprising, funny, or quotable moments — not the intro/outro.

## 🔑 Concepts & Terms
- 3 to 6 important concepts, frameworks, jargon, or proper nouns the speaker uses.
- Format: **Concept name** — one-line definition or context.
- Skip if the video is purely conversational with no specialised concepts.

## ✅ Action Items
- 4 to 6 practical things the viewer can apply, try, or remember.
- Be specific. "Read more books" is bad. "Read 'Atomic Habits' chapter 3 on habit stacking" is good.
- If the video is purely informational with no actionable advice, write "(none — this is an informational/entertainment video)".

Do not include any preamble. Start directly with the `## 📌 At a Glance` heading."""

def build_merge_prompt(num_chunks: int) -> str:
    # Scale section sizes with chunk count so a 4-hour video gets a proportionally bigger summary.
    min_takeaways = max(12, num_chunks * 3)              # ≥3 per chunk, min 12
    min_quotes = max(4, num_chunks)                       # ≥1 per chunk, min 4
    min_timestamps = max(6, num_chunks * 2)               # 2 per chunk, min 6
    tldr_sentences = "10-12" if num_chunks >= 4 else "8-10"

    return f"""You are merging {num_chunks} consecutive segment summaries of a single video into ONE comprehensive summary.

CRITICAL ACCURACY RULES:
- ONLY use information that is in the segment summaries below. Do NOT invent, guess, or extrapolate facts.
- Do NOT state the video's exact duration in minutes — you don't know it precisely. If you must reference length, write "the video" or "this multi-segment video", never a specific minute count.
- If a segment summary says "no speaker identified", do NOT make up a name.
- If something isn't covered in the source summaries, do NOT include it.
- Preserve detail from EVERY segment — readers want to know what happened in every part of the video, not just the highlights.

Use exactly this Markdown structure:

## 📌 At a Glance
- **Type:** [tutorial/lecture/interview/podcast/vlog/talk/news/review/explainer/documentary — pick the best fit based on the segment summaries]
- **Topic:** [one-line topic in 8 words or less, drawn from the segments]
- **Speaker / Channel:** [name if identifiable from the segments, else "Unknown"]
- **Best for:** [one-line audience description]

## TL;DR
A flowing {tldr_sentences} sentence summary covering the speaker, central thesis, the major ideas across the whole video (not just the first 30 minutes), key examples, and the conclusion. For long videos, mention how the content develops or shifts over time. Do NOT include the exact duration in minutes.

## 🗂️ Section by Section
For EACH segment, write 3-5 bullet points covering what happens in that 30-minute window. Format:

### Minutes 0–30
- Bullet 1
- Bullet 2
- Bullet 3

### Minutes 30–60
- Bullet 1
- ...

(Continue for ALL {num_chunks} segments. Do not skip any. Do not invent content for segments that aren't in the source.)

## 🎯 Key Takeaways
- AT LEAST {min_takeaways} bullet points, drawn from across the whole video.
- At least 3 takeaways per segment — don't load up on the first segments and skip the later ones.
- Be specific: numbers, names, frameworks, concrete examples — but ONLY ones actually mentioned in the source summaries.

## 💬 Notable Quotes
- AT LEAST {min_quotes} direct quotes spread across the video.
- Format: > "Quote." — Speaker (timestamp if identifiable)
- Pick at least one quote from each part of the video — early, middle, late.
- ONLY use quotes that appear in the source segment summaries. Do not paraphrase into pseudo-quotes.

## ⏱️ Worth Watching
- AT LEAST {min_timestamps} timestamps spread across the whole video.
- AT LEAST 1-2 timestamps from each segment — do not cluster them all in the first hour.
- Adjust timestamps to reflect position in the FULL video. A "5:00" inside segment 2 of a 30-min-chunked video means 35:00 in the full video.
- Format: **35:00** — what's notable.

## 🔑 Concepts & Terms
- 5-10 important concepts, frameworks, jargon, or proper nouns drawn from the segments.
- Format: **Term** — one-line definition or context.

## ✅ Action Items
- 5-8 specific, practical things the viewer can apply, try, or remember, drawn from the source segments.
- Cover advice from across the video, not just one section.

Do not include any preamble. Start directly with `## 📌 At a Glance`."""


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
    max_output_tokens=8192,  # large enough for full 4-hour merged output
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


def summarize_segment_with_retry(
    client: genai.Client,
    url: str,
    start_s: int,
    end_s: int,
    on_wait: callable | None = None,
) -> str:
    """Call summarize_segment with auto-backoff on transient TPM 429s.

    Long videos chunked into 30-min slices burn through Gemini's
    tokens-per-minute budget fast. Instead of giving up the moment a
    429 happens, wait and retry — usually the TPM window resets within
    30-60s and the next chunk goes through.
    """
    last_exc: Exception | None = None
    for attempt, delay in enumerate((0,) + RETRY_BACKOFF_SECONDS):
        if delay:
            if on_wait:
                on_wait(delay, attempt)
            time.sleep(delay)
        try:
            return summarize_segment(client, url, start_s, end_s)
        except Exception as exc:
            msg = str(exc)
            last_exc = exc
            # Past-end-of-video / fatal errors → propagate immediately, don't retry.
            if any(needle in msg for needle in ("INVALID_ARGUMENT", "API_KEY", "PERMISSION_DENIED", "401", "403")):
                raise
            # Transient quota hit → retry with backoff.
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                continue
            # Unknown error → propagate.
            raise
    if last_exc:
        raise last_exc
    return ""


def merge_summaries(client: genai.Client, summaries: list[str]) -> str:
    num_chunks = len(summaries)
    merge_prompt = build_merge_prompt(num_chunks)

    # Label each segment with its actual time range so the LLM can adjust timestamps correctly.
    joined_parts = []
    for i, s in enumerate(summaries):
        start_min = i * (CHUNK_SIZE_SECONDS // 60)
        end_min = start_min + (CHUNK_SIZE_SECONDS // 60)
        joined_parts.append(
            f"### Segment {i + 1} (full-video minutes {start_min}–{end_min})\n\n{s}"
        )
    joined = "\n\n---\n\n".join(joined_parts)

    response = client.models.generate_content(
        model=MODEL,
        contents=f"{merge_prompt}\n\nHere are the {num_chunks} segment summaries to merge:\n\n{joined}",
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
            /* Hide Streamlit's default chrome so visitors only see the VidGist UI:
               - the hamburger / Deploy menu (top-right, includes 'View source' / 'Fork' on community cloud)
               - the toolbar (the bar that shows the running status + GitHub icon)
               - the 'Made with Streamlit' footer */
            #MainMenu {visibility: hidden !important;}
            header[data-testid="stHeader"] {visibility: hidden !important; height: 0 !important;}
            [data-testid="stToolbar"] {display: none !important;}
            footer {visibility: hidden !important;}
            .stDeployButton {display: none !important;}
            /* Streamlit Community Cloud's "Hosted with Streamlit" badge */
            div[class^="viewerBadge_"], a[class^="viewerBadge_"] {display: none !important;}

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
            "3. Paste it below\n\n"
            "**Free tier limits:** 15 requests/min, 1,500 requests/day. "
            "One short video = 1 request. A 2-hour video uses 5 requests (4 chunks + merge)."
        )
        key = st.text_input(
            "Paste your Gemini API key",
            type="password",
            value=st.session_state.get("api_key", ""),
            placeholder="AIzaSy...",
            label_visibility="collapsed",
        )
        if key and key != st.session_state.get("api_key", ""):
            cleaned = key.strip()
            st.session_state["api_key"] = cleaned
            # Wipe any stale results / errors from a previous (bad-key) run
            for stale_key in ("last_summary", "last_chunks", "last_error", "last_video_id"):
                st.session_state.pop(stale_key, None)
            # Soft format hint — Gemini keys start with AIzaSy, are 39 chars
            if not cleaned.startswith("AIzaSy") or len(cleaned) < 35:
                st.warning(
                    "⚠️ This doesn't look like a typical Gemini key (should start with `AIzaSy` and be ~39 characters). "
                    "Saving anyway — if it doesn't work, double-check at "
                    "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
                )
            else:
                st.success("✅ Key saved. Old errors cleared — try again.")
            # Force a fresh render so any error messages from the previous run disappear
            st.rerun()
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


def render_summary(
    summary: str,
    info: VideoInfo,
    was_chunked: bool,
    chunk_summaries: list[str] | None = None,
) -> None:
    st.video(info.canonical_url)
    if was_chunked:
        st.info(
            "🧩 This was a long video — VidGist split it into 30-minute chunks. "
            "Below is the merged summary; expand the per-segment details further down "
            "for the full chunk-by-chunk breakdown."
        )
    st.markdown(summary)

    # Compose the downloadable file: merged summary + all chunk details
    download_content = summary
    if chunk_summaries and len(chunk_summaries) > 1:
        st.divider()
        st.subheader("🔍 Per-segment details")
        st.caption(
            "Each 30-minute segment summarised in full — expand any segment to read "
            "the complete summary for that part of the video."
        )
        chunk_block_lines = ["", "---", "", "# Per-Segment Summaries", ""]
        for i, seg in enumerate(chunk_summaries):
            start_min = i * (CHUNK_SIZE_SECONDS // 60)
            end_min = start_min + (CHUNK_SIZE_SECONDS // 60)
            label = f"Segment {i + 1} · minutes {start_min}–{end_min}"
            with st.expander(label):
                st.markdown(seg)
            chunk_block_lines.append(f"## {label}")
            chunk_block_lines.append("")
            chunk_block_lines.append(seg)
            chunk_block_lines.append("")
        download_content = summary + "\n".join(chunk_block_lines)

    st.download_button(
        "💾 Download summary as Markdown",
        data=download_content,
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

        quota_hit = False
        first_failure_msg: str | None = None
        for i, (start_s, end_s) in enumerate(chunks):
            remaining_chunks = len(chunks) - i
            eta_sec = remaining_chunks * SECONDS_PER_CHUNK_ESTIMATE + 15  # +15 for merge
            eta_min = eta_sec // 60
            eta_rem = eta_sec % 60
            if eta_min >= 1:
                eta_text = f"~{eta_min} min {eta_rem}s remaining" if eta_rem else f"~{eta_min} min remaining"
            else:
                eta_text = f"~{eta_sec}s remaining"
            progress.progress(
                i / (len(chunks) + 1),
                text=f"Summarising minute {start_s // 60}–{end_s // 60}… ({eta_text})",
            )

            # Per-chunk waiting indicator for retries
            chunk_label = f"Minute {start_s // 60}–{end_s // 60}"
            def on_retry_wait(delay: int, attempt: int, _label=chunk_label, _i=i, _total=len(chunks)):
                progress.progress(
                    _i / (_total + 1),
                    text=f"{_label}: hit per-minute limit, waiting {delay}s before retry…",
                )

            try:
                seg = summarize_segment_with_retry(
                    client, info.canonical_url, start_s, end_s, on_wait=on_retry_wait
                )
                if seg:
                    segments.append(seg)
                    successful_chunks += 1
            except Exception as exc:
                msg = str(exc)
                lower = msg.lower()

                # Quota still exhausted after all retries → stop and use what we have.
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    quota_hit = True
                    break

                # "Past end of video" — only quietly break if we ALREADY have at least
                # one chunk. If the very first chunk fails with INVALID_ARGUMENT,
                # that's a real problem (e.g. Gemini ignored our slicing and rejected
                # the whole 4-hour video), so we should propagate the real error.
                is_past_end = (
                    "INVALID_ARGUMENT" in msg
                    and ("offset" in lower or "duration" in lower or "out of range" in lower or "empty video" in lower)
                )
                if is_past_end and segments:
                    break

                # 10,800-image / frame-count error → real issue, surface it cleanly.
                if "10800" in msg or ("images" in lower and "fewer" in lower):
                    first_failure_msg = (
                        "Gemini hit its frame-count limit even on a single chunk "
                        "— this video has unusually dense visuals (e.g. lots of cuts or text)."
                    )
                    break

                # Bad/expired key, permission denied → real issue, propagate.
                if any(s in msg.upper() for s in ("API_KEY", "PERMISSION_DENIED", "UNAUTHENTICATED")) or "401" in msg or "403" in msg:
                    first_failure_msg = "API key rejected. Generate a fresh key at aistudio.google.com/apikey."
                    break

                # Safety / blocked / not found → real issue, surface it.
                if any(s in msg.upper() for s in ("SAFETY", "BLOCKED", "RECITATION")):
                    first_failure_msg = "Gemini blocked this video for policy reasons (often happens with copyrighted music videos, age-restricted content, or live streams)."
                    break
                if "NOT_FOUND" in msg.upper() or "404" in msg:
                    first_failure_msg = "Gemini couldn't access this video — it may be private, deleted, region-blocked, age-gated, or a live stream."
                    break

                # Unknown error: capture the first one for diagnostic and warn.
                if first_failure_msg is None:
                    first_failure_msg = msg[:400]
                st.warning(f"Chunk {i + 1} ({start_s // 60}–{end_s // 60} min) failed: {msg[:200]}")
            time.sleep(RATE_LIMIT_PAUSE)

        progress.empty()

        if quota_hit and not segments:
            st.error(
                "🚦 **Your API key hit its quota before any chunk completed.**\n\n"
                "**Fix:**\n"
                "- Wait a minute and retry (per-minute limit), or\n"
                "- Wait until tomorrow midnight Pacific (daily limit reset), or\n"
                "- **Generate a new free key** at "
                "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and paste it above."
            )
            return

        if not segments:
            # Show the actual reason chunk 1 failed instead of a generic message.
            if first_failure_msg:
                st.error(
                    f"❌ **Couldn't summarise any chunk of this video.**\n\n"
                    f"**Reason:** {first_failure_msg}\n\n"
                    f"Try a different video, or untick chunked mode if it's under 50 minutes."
                )
            else:
                st.error(
                    "❌ **Couldn't summarise any chunk of this video.** "
                    "Gemini returned empty responses for every segment. "
                    "This sometimes happens with audio-less videos, live streams, or content Gemini can't access. "
                    "Try a different video URL."
                )
            return

        # Build the final summary — merged if we have all chunks, otherwise raw.
        merge_skipped_for_quota = False
        if len(segments) > 1:
            try:
                final = merge_summaries(client, segments)
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    merge_skipped_for_quota = True
                    # Concatenate all segments instead of dropping data — user gets
                    # every chunk's content even if the merge call couldn't run.
                    final = "\n\n---\n\n".join(
                        f"## Segment {i + 1} (minutes {i * 30}–{(i + 1) * 30})\n\n{s}"
                        for i, s in enumerate(segments)
                    )
                else:
                    raise
        else:
            final = segments[0]

        # ONE consolidated status message — no contradictions.
        if quota_hit and merge_skipped_for_quota:
            st.warning(
                f"🚦 **Your key hit its per-minute or daily token limit** "
                f"(only {successful_chunks} of {len(chunks)} chunks ran, and the merge step couldn't run either). "
                f"Showing the raw per-segment summaries below. "
                f"To get a clean merged summary, wait a minute or generate a fresh free key at "
                f"[aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
            )
        elif quota_hit:
            st.warning(
                f"🚦 **Your key hit its per-minute or daily token limit partway through** "
                f"(only {successful_chunks} of {len(chunks)} chunks completed). "
                f"The summary below covers minutes 0–{successful_chunks * 30} only. "
                f"To get the full video summary, wait a minute or generate a fresh free key at "
                f"[aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
            )
        elif merge_skipped_for_quota:
            st.warning(
                "🚦 **Hit the token limit on the final merge step.** "
                "All chunks were summarised — showing them as raw per-segment summaries below "
                "rather than a single merged view. Wait a minute and click Summarize again "
                "to get the merged version."
            )
        else:
            st.success(f"✅ Combined {successful_chunks} segment summaries into one.")

        render_summary(final, info, was_chunked=True, chunk_summaries=segments)
    else:
        with st.spinner("Watching the video and writing your summary…"):
            summary = summarize_segment(client, info.canonical_url)
        if not summary or len(summary.strip()) < 30:
            st.error(
                "🤐 **Gemini returned an empty or very short summary.** "
                "This usually means the video has no audio, is too long for a single call, "
                "or was blocked by safety filters. Try ticking **🧩 Chunked mode**, "
                "or pick a different video."
            )
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
    # Strip whitespace, newlines, and trailing slashes that get pasted accidentally
    url_value = (sample_picked or typed_url or "").strip().rstrip("/")

    long_video_mode = st.checkbox(
        "🧩 Chunked mode (for videos longer than ~50 minutes)",
        value=False,
        help="Splits the video into 30-minute chunks and merges the summaries. "
             "Use this for podcasts or lectures over ~50 minutes. Takes longer.",
    )

    # Optional: ask user for video length so we can compute an accurate ETA
    video_length_min = None
    if long_video_mode:
        video_length_hours = st.slider(
            "Roughly how long is this video? (hours)",
            min_value=0.5,
            max_value=4.0,
            value=1.5,
            step=0.5,
            format="%.1f h",
            help="Just used to compute an accurate progress estimate. "
                 "The summary itself works regardless. Each 0.5h = one chunk.",
        )
        video_length_min = int(video_length_hours * 60)

    # Always-clickable button — validation happens on submit so users don't
    # get blocked by Streamlit's text-input timing (text_input only pushes
    # to session_state on Enter / blur, which made the disabled flag stale).
    go = st.button(
        "✨ Summarize video",
        type="primary",
        use_container_width=True,
    )

    # Tiny hint right under the button so first-time users know what to do
    # without seeing it as a scary warning when they haven't filled things in yet.
    if not api_key:
        st.caption("☝️ Paste your free Gemini API key above first.")
    elif not url_value:
        st.caption("☝️ Paste a YouTube URL above (or click a sample), then press the button.")

    # Estimated processing time hint — dynamic
    if long_video_mode and video_length_min:
        chunks_needed = (video_length_min * 60 + CHUNK_SIZE_SECONDS - 1) // CHUNK_SIZE_SECONDS
        eta_sec = chunks_needed * SECONDS_PER_CHUNK_ESTIMATE + 15  # +15 for merge
        eta_min = eta_sec // 60
        eta_rem_sec = eta_sec % 60
        eta_text = f"~{eta_min} min {eta_rem_sec}s" if eta_rem_sec else f"~{eta_min} min"
        hours_label = f"{video_length_hours:.1f} h"
        st.caption(
            f"⏱️ **Expected time for a {hours_label} video: {eta_text}** "
            f"({chunks_needed} chunks of 0.5 h + final merge step). "
            f"Hard cap: 4 h videos."
        )
    elif long_video_mode:
        st.caption(
            "⏱️ **Expected time:** ~2 min (1 h) · ~3 min (1.5 h) · ~5 min (2.5 h) · "
            "~7 min (4 h). Hard cap: 4-hour videos. Use the slider above for an accurate estimate."
        )
    else:
        st.caption(
            "⏱️ **Expected time:** ~20 s (under 5 min) · ~30–60 s (5–20 min) · ~60–90 s (20–50 min). "
            "**Tick chunked mode above for videos longer than 50 minutes** "
            "(otherwise Gemini will reject videos over ~3 hours with a frame-count error)."
        )

    # On click, validate first so users get a clear message instead of silently nothing.
    if go and not api_key:
        st.warning("⚠️ Please paste your Gemini API key in the section above first.")
        render_footer()
        return
    if go and not url_value:
        st.warning("⚠️ Please paste a YouTube URL above (or click a sample) before summarising.")
        render_footer()
        return

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
                msg_upper = msg.upper()
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
                elif "API_KEY" in msg_upper or "401" in msg or "403" in msg or "PERMISSION_DENIED" in msg_upper or "UNAUTHENTICATED" in msg_upper:
                    st.error(
                        "🔑 **Your API key was rejected.** It may be wrong, expired, or revoked.\n\n"
                        "Generate a new free key at "
                        "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and paste it above."
                    )
                elif "10800" in msg or ("images" in msg.lower() and "fewer" in msg.lower()):
                    st.error(
                        "📏 **This video is too long for a single Gemini call** "
                        "(over ~3 hours of frames at default sampling).\n\n"
                        "**Fix:** Tick **🧩 Chunked mode** above and try again — "
                        "VidGist will split it into 30-minute chunks and merge the summaries. "
                        "Hard cap is 4-hour videos."
                    )
                elif "video" in msg.lower() and ("long" in msg.lower() or "duration" in msg.lower() or "size" in msg.lower()):
                    st.error(
                        "📏 **This video is too long for a single Gemini call.** "
                        "Tick **🧩 Chunked mode** above and try again."
                    )
                elif "SAFETY" in msg_upper or "BLOCKED" in msg_upper or "RECITATION" in msg_upper:
                    st.error(
                        "🛑 **Gemini blocked this video for safety/policy reasons.** "
                        "This sometimes happens with copyrighted music videos, age-restricted content, or live streams. "
                        "Try a different video."
                    )
                elif "NOT_FOUND" in msg_upper or "404" in msg or "could not" in msg.lower():
                    st.error(
                        "🚫 **Gemini couldn't access this video.** It may be private, deleted, region-blocked, "
                        "age-gated, or a live stream. Try a different public video URL."
                    )
                elif "DEADLINE_EXCEEDED" in msg_upper or "timeout" in msg.lower():
                    st.error(
                        "⏱️ **Gemini timed out.** Long or complex videos sometimes do this. "
                        "Try ticking **🧩 Chunked mode** above, or pick a shorter section of the video."
                    )
                elif "UNAVAILABLE" in msg_upper or "503" in msg or "500" in msg:
                    st.error(
                        "🌐 **Gemini's servers are temporarily unavailable.** Wait a moment and try again. "
                        "If it keeps happening, check [status.cloud.google.com](https://status.cloud.google.com)."
                    )
                else:
                    st.error(
                        f"❌ Something went wrong:\n\n```\n{msg[:400]}\n```\n\n"
                        f"Please try again, or open an issue on the GitHub repo with this error."
                    )

    render_footer()


if __name__ == "__main__":
    main()
