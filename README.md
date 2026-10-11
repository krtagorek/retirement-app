# Retirement Readiness Check

A week-one Streamlit app that helps a user upload financial history, enter retirement goals, and receive a clear retirement projection with inflation context and AI-generated guidance.

## What it does

- Upload a spreadsheet with monthly income, expenses, profits, retirement balances, and total assets.
- Enter current age, target retirement age, 401(k) annual contribution, and desired annual income.
- Generate a deterministic retirement projection using compound-growth math.
- Convert the projected balance into an estimated annual retirement income using a 4% withdrawal-rate assumption.
- Pull CPI inflation context from FRED when an API key is available.
- Generate concise, plain-English guidance with OpenAI.
- Ask follow-up questions in the Planning Assistant to explore what-if scenarios.

## How this works

- **Projection math**: The app uses a simple compound-growth model to estimate retirement balance from the user’s current age, target retirement age, starting balance, and annual contribution.
- **Income estimate**: The app converts the projected retirement balance into an estimated annual retirement income using a 4% withdrawal-rate assumption.
- **FRED inflation context**: The app pulls CPI data from FRED so the user can see the current inflation backdrop alongside the projection.
- **LLM guidance**: The app sends the projection and financial snapshot to OpenAI to generate friendly, plain-English guidance and next-step suggestions.

## Why it matters

This app is built for a fast, approachable retirement readiness check. It is designed to show:

- where a user stands today,
- whether they appear on track,
- how much of an income gap may remain,
- and what economic context may affect future purchasing power.

## Demo flow

1. Open the app.
2. Download the sample spreadsheet if needed.
3. Upload your own file or the sample file.
4. Enter your criteria.
5. Click **See My Summary**.
6. Review the status card, projected retirement income, income gap, charts, inflation snapshot, and guidance.
7. Use the Planning Assistant to ask what-if questions like increasing your contribution.

## Summary page experience

The Summary page is organized to reduce scrolling and keep the main takeaway obvious:

- **Overview**: top-level status, projected income, income gap, chart, and financial snapshot.
- **Recommendations**: full guidance text in a clean readable format.
- **Planning Assistant**: chat-style what-if exploration with persistent conversation history and a clear button.
- **Technical Details**: assumptions and other supporting information.
- The upload area validates required columns, accepts extra columns, and shows clear feedback for unreadable files or missing fields.

## Sample data

The repository includes `sample_financial_data.csv` with realistic values for a 45-year-old profile. It includes:

- monthly income
- monthly expenses
- monthly profit
- 401(k) balance
- Roth IRA balance
- taxable investments
- cash savings
- home equity
- other assets
- total assets

## Tech stack

- Python
- Streamlit
- Pandas
- `fredapi`
- OpenAI API
- LangGraph

## Local setup

### 1) Create and activate the virtual environment

If your project already has `.venv`, activate it.

```bash
source .venv/bin/activate
```

### 2) Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Set:

- `FRED_API_KEY`
- `OPENAI_API_KEY`

Optional:

- `OPENAI_MODEL`

### 3) Install dependencies

If you are using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

## Tests

Run the unit tests with the project venv:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Assumptions

- The math engine uses a fixed long-term annual return assumption unless changed in code.
- The projected retirement income uses a 4% withdrawal-rate assumption.
- CPI inflation context is fetched from FRED when the API key is available.
- Guidance is educational and not personalized financial advice.
- Extra spreadsheet columns are allowed, but required columns must be present.
- Unreadable or incompatible files are rejected with a friendly error message.

## Transparency

The app shows a short assumptions box on the Summary page so users can quickly see:

- the return rate used in the projection,
- the 4% withdrawal assumption,
- the starting retirement balance,
- the inflation source,
- and the educational-use-only disclaimer.

## Screenshots

Add screenshots here once you capture them from the running app:

- `screenshots/details-page.png` — the input page with upload and criteria fields
- `screenshots/end-user-flow.png` — the end-user workflow diagram
- `screenshots/technical-architecture.png` — the technical architecture diagram
- `screenshots/summary-page.png` — the summary page with the status card and charts
- `screenshots/summary-page1.png` — guidance or assistant view
- `screenshots/summary-page2.png` — another summary view
- `screenshots/summary-page3.png` — another summary view

If you want to include a short demo video or GIF, add it here as well:

- `screenshots/demo.gif`

## Diagrams

- `screenshots/end-user-flow.png` — end-user workflow diagram
- `screenshots/technical-architecture.png` — technical architecture diagram with Planning Assistant flow

## Project structure

- `app.py` — Streamlit interface and app flow
- `utils.py` — projection math, CSV handling, FRED integration, and OpenAI synthesis
- `sample_financial_data.csv` — demo spreadsheet
- `tests/` — unit tests for core utilities

## Notes

If you want to present this project, the strongest story is:

- a user uploads real financial history,
- the app produces an understandable retirement snapshot,
- the app explains the result in plain English,
- and the Planning Assistant lets them explore what-if scenarios.
