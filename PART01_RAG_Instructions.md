# Deep Learning RAG Workshop — Student Instructions

**Repository:** `https://github.com/gitmystuff/deep-learning-rag-agent.git`

Use this guide together with **RAG_Workshop_Code_Reference.docx**.

- This Markdown guide tells you **what to do** and gives the Windows/Mac terminal commands.
- Do **not** put your Groq API key in GitHub, screenshots, Canvas, Teams, or shared documents.
- This workshop uses a **Markdown-only RAG demo** for document ingestion. PDF support is not required for the workshop submission.

---

## Step 1 — Make sure Git is installed

### Windows

Open PowerShell and run:

```powershell
git --version
```

If Git is not installed and `winget` is available:

```powershell
winget install --id Git.Git -e --source winget
```

Close and reopen PowerShell, then run:

```powershell
git --version
```

If `winget` is unavailable, install Git for Windows from:

`https://git-scm.com/download/win`

### Mac

Open Terminal and run:

```bash
git --version
```

If Git is not installed:

```bash
xcode-select --install
```

If you already use Homebrew, you may instead run:

```bash
brew install git
```

Verify:

```bash
git --version
```

---

## Step 2 — Clone the workshop repository

### Windows PowerShell

```powershell
git clone https://github.com/gitmystuff/deep-learning-rag-agent.git
cd deep-learning-rag-agent
dir
```

### Mac Terminal

```bash
git clone https://github.com/gitmystuff/deep-learning-rag-agent.git
cd deep-learning-rag-agent
ls
```

You should see files/folders such as `src`, `tests`, `docs`, `examples`, `pyproject.toml`, and `README.md`.

---

## Step 3 — Install `uv`

First check:

```text
uv --version
```

### Windows PowerShell

If `uv` is missing:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then run:

```powershell
uv --version
```

If Windows still cannot find `uv`, temporarily add the common install folder to PATH:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv --version
```

### Mac Terminal

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal, then run:

```bash
uv --version
```

If needed for zsh:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
uv --version
```

---

## Step 4 — Install the project dependencies

Run from inside the `deep-learning-rag-agent` folder.

### Windows and Mac

```text
uv sync
```

For workshop compatibility, also install/record `torchvision`:

```text
uv add torchvision
```

A deprecation warning is okay as long as the command completes successfully.

---

## Step 5 — Create the local `.env` file

### Windows PowerShell

```powershell
Copy-Item .env.example .env
dir .env*
```

Open it:

```powershell
notepad .env
```

### Mac Terminal

```bash
cp .env.example .env
ls -la .env*
```

Open it with one of the following:

```bash
open -e .env
```

or

```bash
nano .env
```

### Required workshop values

Make sure the `.env` file uses:

```text
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your_own_groq_api_key
```

Keep the existing local embedding setting for `all-MiniLM-L6-v2`.

**Important:** Never share the real API key. The `.env` file must stay local and must not be committed to GitHub.

---

## Step 6 — Open the project in VS Code

### Windows and Mac

```text
code .
```

If the `code` command is unavailable, open VS Code normally and choose **File → Open Folder → deep-learning-rag-agent**.

---

## Step 7 — Complete the Python TODO sections

Open the pyhton files and complete the required edits in this order:

1. `src/rag_agent/config.py`
   - local embedding factory
   - Groq LLM factory
2. `src/rag_agent/vectorstore/store.py`
   - ChromaDB initialization
   - duplicate detection
   - ingestion
   - retrieval/query
3. `src/rag_agent/corpus/chunker.py`
   - metadata inference
   - Markdown chunking
   - single-file chunking
4. `src/rag_agent/ui/app.py`
   - Streamlit document ingestion
   - cached LLM
   - RAG question-and-answer chat
5. `src/rag_agent/agent/graph.py`
   - optional compatibility section if you want the starter LangGraph graph to compile

For this workshop, you do **not** need to implement PDF ingestion or the unused multi-file chunk helper. The Streamlit demo will ingest one or more Markdown files through the UI.

---

## Step 8 — Start the Streamlit application

### Windows and Mac

```text
uv run streamlit run src/rag_agent/ui/app.py
```

Streamlit should open in your browser. If it does not, open the local URL shown in the terminal, usually:

`http://localhost:8501`

You should see a document-ingestion sidebar and a RAG question-and-answer chat area.

---

## Step 9 — Prepare at least one Markdown note

Create a Markdown note related to a technical topic you want the RAG system to answer questions about or you can use the file provided.

Recommended filename format:

`topic_difficulty.md`

Examples:

- `lstm_intermediate.md`
- `cnn_intermediate.md`
- `ann_beginner.md`

Use clear headings and enough text for the system to retrieve meaningful information. You may create more than one note.

---

## Step 10 — Ingest the note(s)

In the Streamlit sidebar:

1. Choose one or more `.md` files.
2. Click **Ingest Documents**.
3. Confirm that Streamlit reports chunks added or duplicates skipped.
4. Confirm that the displayed ChromaDB chunk count is greater than zero.

If you upload the same note again, seeing a duplicate skipped is expected.

---

## Step 11 — Ask the RAG system a question

Use the chat input to ask a question that can be answered from your uploaded note.

Examples:

- “What does the forget gate do?”
- “What is a CNN commonly used for?”
- Ask your own question based on your own note.

The final answer should be generated from retrieved context, and the Streamlit interface should show the source information.

---

## Step 12 — Required submission evidence: screenshot of the LLM answer

Your GitHub repository **must contain a screenshot showing the working RAG demo**.

The screenshot must visibly show:

- your question,
- the LLM/RAG answer,
- and, if possible, the source/citation area.

Do **not** show your `.env` file or API key in the screenshot.

Create a screenshots folder.

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force screenshots
```

Take a screenshot with **Windows + Shift + S** and save/copy it into the project as:

`screenshots/rag_demo.png`

### Mac

```bash
mkdir -p screenshots
```

Take a screenshot with **Shift + Command + 4** and save/move it into the project as:

`screenshots/rag_demo.png`

Before submission, confirm the file exists inside the project.

---

## Step 13 — Final local checks

### Windows and Mac

Check your Git status:

```text
git status
```

Check that `.env` is ignored:

```text
git check-ignore .env
```

If the command prints `.env`, that is good — Git is ignoring it.

Before pushing, make sure:

- the application runs,
- at least one note can be ingested,
- at least one question receives a grounded answer,
- `screenshots/rag_demo.png` exists,
- `.env` is **not** being committed,
- your real `GROQ_API_KEY` does not appear in any tracked file.

You do not need to upload local ChromaDB database files if they are large; the project should be reproducible from the code and sample Markdown note(s).

---

## Step 14 — Publish the completed project to your own GitHub repository

Because you cloned the instructor repository, first create a **new empty repository** under your own GitHub account.

On GitHub:

1. Click **New repository**.
2. Give it a name such as `deep-learning-rag-agent-yourname`.
3. Choose **Public** unless your instructor says otherwise.
4. Do **not** initialize it with a README, `.gitignore`, or license.
5. Create the repository.

Then return to your project terminal.

### Windows and Mac — check the current remote

```text
git remote -v
```

If `origin` still points to the instructor repository, preserve it as `upstream`:

```text
git remote rename origin upstream
```

Add your own GitHub repository as the new `origin`:

```text
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Stage your completed work:

```text
git add .
git status
```

**Stop and inspect `git status`.** Confirm that `.env` is not staged.

Commit:

```text
git commit -m "Complete RAG workshop project"
```

Use `main` as the branch and push:

```text
git branch -M main
git push -u origin main
```

If your local `origin` already points to your own repository, skip the remote rename/add commands and just commit and push.

---

## Step 15 — Final GitHub submission checklist

Open your personal GitHub repository in the browser and verify all of the following:

- [ ] Source code is present.
- [ ] Your Markdown test/study note(s) are present.
- [ ] `screenshots/rag_demo.png` is present and displays a question and the LLM answer.
- [ ] `.env` is **not** present.
- [ ] Your API key is **not** visible anywhere.
- [ ] The repository opens successfully and shows your final commit.

### Submit

Submit the URL of **your personal completed GitHub repository** as instructed.

Your required evidence is the repository itself, including the screenshot of the working LLM/RAG question-and-answer result.
