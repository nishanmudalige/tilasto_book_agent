# Tilasto-opas: Finnish–English textbook assistant

An adaptation of `stat1301_agent.zip`: same Flask + Claude API + Markdown/MathJax/Plotly interface, but a **page-aware search index built from a Finnish PDF**.

**Source:** Juhani Tähtinen, Eero Laakkonen and Mari Broberg (2020), *Tilastollisen aineiston käsittelyn ja tulkinnan perusteita*, University of Turku, second revised edition (254 PDF pages).

## Features

- Ask questions in **Finnish or English**. English questions are converted into Finnish search terms using a small Claude Haiku call before local BM25 retrieval.
- Translate excerpts or named book pages from Finnish into **English** (use the checkbox).
- Cite the document as clickable `book p. NN` links; these open the exact PDF viewer page. For the main text, **PDF page = printed page + 2**.
- Retrieve relevant *passages*, not the entire PDF, before each Claude Sonnet call.
- Simple PDF page image is sent to Claude when the question explicitly mentions a diagram, table, image, plot or figure. This costs more per request.
- Support Markdown, MathJax mathematics, and the original plot renderer.
- Optional shared access code for a small private demo.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Extract PDF text, build the page-aware BM25 index, translate queries, query Claude, serve the web app |
| `data/book.pdf` | The provided Finnish textbook (included for local use; confirm redistribution rights before public publication) |
| `templates/index.html` | Bilingual chat interface and clickable PDF citations |
| `static/figures.js` | Statistical plot renderer inherited from STAT1301 |
| `.env.example` | Copy this to `.env` and enter your own API key |
| `requirements.txt` | Python packages |
| `render.yaml` | Render Blueprint configuration |

## Run on Mac (Terminal)

```bash
cd ~/Desktop/tilasto_book_agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
open -e .env     # put your actual ANTHROPIC_API_KEY in the file
python app.py
```

Open `http://127.0.0.1:5000/`. `app.py` loads your `.env` automatically using `python-dotenv`; you do not need to export your API key manually. On Render, add keys in Environment, not `.env`.

## Try these

- `Mitä kirjassa sanotaan luottamusväleistä?`
- `Explain effect size and tell me which pages I should read.`
- `Translate printed page 49 into English.` (or enable **Translate source into English**)
- `Where is logistic regression explained?`
- `Selitä kaksisuuntainen varianssianalyysi suomeksi.`
- `Explain the table on page 49.` (sends a page image so Claude can inspect its layout)

## Deploy on Render

1. Create a **private GitHub repository** and add all the project files. **Do not commit `.env`**. Confirm permissions before publishing the full textbook; if uncertain, keep the repository and deployed service private/restricted.
2. In Render, click **New → Blueprint**, connect the repo, and approve `render.yaml`.
3. Provide `ANTHROPIC_API_KEY` in Render when prompted, and set `APP_PASSWORD` to a strong access code to limit unauthorised API use. `FLASK_SECRET_KEY` is generated automatically.
4. Visit your Render URL and test `/health`, the chat and a source link.
5. Later updates: commit and push changes; Render will redeploy.

**Do not** put your Claude API key in HTML, JavaScript, or the GitHub repository. A Claude web subscription is not the same as API billing; activate API access in the Anthropic Console.

## Troubleshooting

- `ModuleNotFoundError: flask` — activate `.venv` and run `python -m pip install -r requirements.txt`.
- `zsh: command not found: flask` — run `python app.py` rather than `flask run`.
- `Claude API key invalid` — check `.env` or Render → Environment → `ANTHROPIC_API_KEY`; redeploy if changed.
- `NotFoundError` / model error — verify your account has access to `ANTHROPIC_MODEL` and `ANTHROPIC_SEARCH_MODEL`; consult model IDs at https://platform.claude.com/docs/en/models/overview.
- No matching source — specify a page number or use a Finnish statistical keyword.
- Translation only partially complete — request one page at a time; source excerpts are size-bounded.
- Bad rendering of a table or diagram — ask specifically for the `table on page NN` or `figure on page NN`, prompting a visual read of that page.
- PDF link returns 401 — enter the access code and send a first question, then open the link.
- PDF citations are **PDF viewer** pages internally, not physical book pages. For book p. 49, link opens PDF page 51.

**Limitations:** BM25 is not semantic vector retrieval; rare paraphrases may be missed. English query expansion improves this but does not guarantee full recall. Model answers/translations should be checked against the original pages, especially mathematical notation and complicated tables. The agent has no independent permission to redistribute the textbook or a translated edition.
