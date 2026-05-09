# VidGist 🎬

> Paste any YouTube URL. Get a TL;DR, key takeaways, and action items in seconds.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

VidGist uses native AI video understanding — no transcript scraping, no
YouTube IP-blocking issues, no broken demos. Long videos (up to 4 hours) are
handled by automatic chunking and merge.

## Features

- **Paste a YouTube URL → instant summary** (TL;DR + key takeaways + action items)
- **Long-video chunking** — split videos up to 4 hours into 30-minute chunks, summarise each, merge
- **Bring your own free API key** — get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Try-it-now sample buttons** so visitors can test in one click
- **Markdown export** — download the summary
- **Helpful errors** for missing keys, bad URLs, rate limits, and over-long videos

## Tech Stack

| Layer | Tool |
|---|---|
| App | Streamlit |
| Language | Python 3.11+ |
| Hosting | Streamlit Community Cloud |

## Run Locally

```bash
git clone <repo-url>
cd vidgist
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>, paste your free API key in the expander, and
try a sample video.

## How It Works

1. You paste a YouTube URL.
2. VidGist sends the URL straight to the AI model — the model fetches and watches the video itself, so the app never has to scrape transcripts.
3. For chunked mode, VidGist passes `start_offset` and `end_offset` to scope each call to a 30-minute window, then asks the model one more time to merge the per-chunk summaries.
4. The final summary is rendered as Markdown and offered as a download.

## Author

Built by **Rahul** · [LinkedIn](https://www.linkedin.com/in/rahul-reddy-avula-37572b328/)
