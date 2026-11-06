# Arabic RAG Assistant

[![tests](https://github.com/eslammohibrahim/arabic-rag-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/eslammohibrahim/arabic-rag-assistant/actions/workflows/tests.yml)

You ask a question in Arabic, it searches a set of Arabic Wikipedia paragraphs, gives the best three to a small language model running locally, and answers with the sources it used. No paid API, everything runs on my laptop's CPU.

In numbers: for 89% of test questions the right paragraph shows up in the top 3 search results. Retrieval is what makes the answers work. Without it the model gets an F1 of 0.11, with it 0.50 (0.61 is the ceiling, when I hand it the correct paragraph myself). I also read 50 random answers by hand: 27 were correct. When search found the right paragraph that went up to 27 out of 40, and when it didn't, none of the 10 were right. One answer takes about 13 seconds on CPU.

```text
Question: ما الجائزة التي حصل عليها نجيب محفوظ؟
Answer:   جائزة نوبل في الأدب.
Sources:  [1] نجيب محفوظ  [2] نجيب محفوظ  [3] نجيب محفوظ
```

## How it works

```
question ──► retrieval (BM25 + embeddings, fused) ──► top 3 paragraphs ──► qwen2.5 1.5B (local, Ollama) ──► answer + sources
```

Search combines two methods. BM25 does keyword matching, after Arabic normalization and a light stemmer that strips "ال" and attached prepositions. multilingual-e5-small does semantic search with embeddings. The two rankings are merged with reciprocal rank fusion.

The question and the top 3 paragraphs then go to `qwen2.5:1.5b` through [Ollama](https://ollama.com). The prompt asks it to answer only from the paragraphs and to say "لا أعرف" if the answer isn't there. The API returns the paragraphs along with the answer so you can check where it came from.

## Data

[ARCD](https://huggingface.co/datasets/hsseinmz/arcd) (Arabic Reading Comprehension Dataset, MIT license) has crowd-written questions about Arabic Wikipedia articles, each with an answer span. That gives 460 unique paragraphs from 146 articles to search over, and 1,388 questions after removing 7 duplicates. I used ARCD's train split (691 questions) as a dev set to pick the retrieval method, and kept the validation split (697 questions) as the test set.

Two things about this data matter for the results. A lot of questions only make sense next to their paragraph, like "بماذا اشتهر؟" (what was he known for?), which doesn't say who. And some reference answers are just wrong: the answer to "متى ولد توماس ألفا أديسون؟" is "ألفا", which is part of his name, not a date. Others have stray brackets. That pulls the automatic scores down even when the model is right, so I also checked answers by hand.

## Retrieval

Scored by whether the paragraph the question was written about comes back in the top results.

| Method | Dev: top 3 | Test: top 1 | Test: top 3 | Test: top 5 | Test MRR@10 |
|---|---|---|---|---|---|
| BM25, no stemming | 83.2% | 62.7% | 82.9% | 88.0% | 0.735 |
| BM25 + Arabic light stemming | 87.0% | 65.3% | 85.9% | 89.7% | 0.761 |
| Dense: multilingual-e5-small | 88.9% | 61.1% | 85.8% | 90.2% | 0.739 |
| Dense: multilingual-e5-base | 88.1% | 62.0% | 86.5% | 91.5% | 0.748 |
| Dense: paraphrase-multilingual-MiniLM | 55.4% | 33.9% | 54.5% | 60.4% | 0.455 |
| Hybrid: BM25 + stemming + e5-small | 91.5% | 68.0% | 89.0% | 92.3% | 0.784 |

![Retrieval quality](reports/figures/retrieval.png)

I picked the hybrid based on the dev questions, and it held up on test. Keyword search and embeddings fail on different questions, so combining them helps. The light stemmer was worth adding to BM25 (top-1 went from 62.7% to 65.3%). I expected e5-base to beat e5-small since it's more than twice the size (278M vs 118M parameters), but it didn't. The paraphrase model did badly because it was trained to match similar sentences, not a question to the paragraph that answers it.

To see why search misses, I split the test questions ([`src/analyze_misses.py`](src/analyze_misses.py)):

| Test questions | Count | Correct paragraph in top 3 |
|---|---|---|
| Name their subject (e.g. "ما الجائزة التي حصل عليها نجيب محفوظ؟") | 444 | 95.7% |
| Don't name it (e.g. "بماذا اشتهر؟") | 253 | 77.1% |
| Two content words or fewer | 85 | 67.1% |

So most misses are questions that don't carry enough information on their own. In a chat app you'd fix this by rewriting the question using the previous messages before searching.

One more thing I found in the data: 76 paragraphs (16.5%) are fully contained inside a longer paragraph from the same article. If search returns the longer one, the table above counts it as a miss even though the answer is there. Counting a hit whenever the answer text appears in the retrieved paragraphs, the hybrid gets 74.3% at top 1 and 89.5% at top 3. The order of the methods stays the same.

## Answers

I ran the same 300 random test questions through the model three ways:

| Setting | Exact match | Word-overlap F1 | Answer contained | Says "لا أعرف" |
|---|---|---|---|---|
| No retrieval (closed book) | 0.3% | 0.11 | 3.3% | 0% |
| RAG: top 3 retrieved paragraphs | 23.0% | 0.50 | 36.3% | 0.3% |
| Correct paragraph given (upper bound) | 29.3% | 0.61 | 43.3% | 0.7% |

![Answer quality](reports/figures/answers.png)

| RAG answers, split by retrieval | Questions | F1 | Answer contained |
|---|---|---|---|
| Retrieval found the right paragraph | 255 | 0.58 | 41.2% |
| Retrieval missed it | 45 | 0.08 | 8.9% |

Without retrieval the model basically doesn't know the answers, but it never says so. It answered all 300 questions anyway. With retrieval, the answers get close to the upper bound whenever the right paragraph is found, which makes retrieval the part worth improving.

The automatic scores undercount, because the model rewords correct answers and some references are wrong. So I read 50 random RAG answers against their paragraphs ([`reports/manual_review.json`](reports/manual_review.json)):

| | Correct | Partly correct | Wrong |
|---|---|---|---|
| All 50 answers | 27 (54%) | 2 | 21 |
| Retrieval found the right paragraph (40) | 27 (67.5%) | 2 | 11 |
| Retrieval missed (10) | 0 | 0 | 10 |

When the paragraph was there and the answer was still wrong, it was usually a misreading (the wrong neighboring countries, for example), and once it just repeated the question. The model also drifts out of Arabic sometimes: one reviewed answer came out in Chinese characters, and when I tried "من رسم الموناليزا؟" myself it answered "Leonardo da Vinci" in English.

It also guesses on questions the collection doesn't cover. I asked "ما هي عاصمة البرازيل؟", search returned unrelated paragraphs (Ronaldo, Morocco), and instead of "لا أعرف" the model answered from memory, wrongly ("رويو دي جانيرو"; the capital is Brasília). A prompt isn't enough to stop a 1.5B model from guessing.

Speed: about 13 seconds per question (median of 20 calls to `/ask` on my laptop CPU). Most of that is the model reading the three paragraphs. Without paragraphs it answers in about 1.5 seconds.

## API

```bash
uvicorn app.main:app
# open http://127.0.0.1:8000 for the web page, http://127.0.0.1:8000/docs for the API docs
```

| Endpoint | Description |
|---|---|
| `GET /` | Web page in Arabic (right-to-left) with example questions |
| `POST /ask` | `{"question": "..."}` → answer plus the 3 source paragraphs |
| `GET /search?q=...&k=3` | Retrieval only, no language model needed |
| `GET /health` | Whether the retriever is loaded and the language model is reachable |

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "ما الجائزة التي حصل عليها نجيب محفوظ؟"}'
```

```json
{
  "answer": "جائزة نوبل في الأدب.",
  "sources": [
    {"passage_id": 66, "title": "نجيب محفوظ", "text": "نجيب محفوظ (11 ديسمبر 1911 - 30 أغسطس 2006) روائي مصري، هو أ...", "score": 0.0325},
    {"passage_id": 67, "title": "نجيب محفوظ", "text": "...", "score": 0.0325},
    {"passage_id": 443, "title": "نجيب محفوظ", "text": "...", "score": 0.0317}
  ]
}
```

If Ollama isn't running, `/ask` returns 503 instead of crashing, and a question under 3 characters returns 422.

## Run it

```bash
# 1. Local language model
#    Install Ollama from https://ollama.com, then:
ollama pull qwen2.5:1.5b

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate          # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 3. Evaluate and serve
python src/eval_retrieval.py    # compares retrieval methods (downloads data and embedding models)
python src/analyze_misses.py    # why retrieval misses, overlapping paragraphs
python src/eval_generation.py   # answer quality on 300 test questions (about 75 minutes on CPU)
uvicorn app.main:app

pytest                          # 11 tests; the language model is stubbed, so Ollama isn't needed
```

## Project structure

```
src/
  data.py              ARCD loading: paragraphs, questions, dev/test split
  text.py              Arabic normalization, stopwords, light stemming
  retrieval.py         BM25, dense, and hybrid retrievers; recall and MRR
  llm.py               client for the local Ollama server
  rag.py               prompts and the retrieve-then-answer pipeline
  metrics.py           exact match, F1, contains-answer, "I don't know" detection
  eval_retrieval.py    retrieval comparison (dev for choosing, test for confirming)
  analyze_misses.py    retrieval failure analysis
  eval_generation.py   closed book vs RAG vs correct-paragraph answers
app/
  main.py              FastAPI service
  static/index.html    Arabic web page
tests/                 text, metrics, and API tests
reports/               results (JSON), every generated answer (JSONL), figures
```

## What I'd do next

Rewrite follow-up questions using the conversation before searching, since that's where most retrieval misses come from. Add a real answerability check that refuses when the retrieved paragraphs don't match the question, instead of trusting the prompt. Try qwen2.5 3B and 7B to see if they stay in Arabic. And speed it up, either with fewer and shorter paragraphs or a GPU.
