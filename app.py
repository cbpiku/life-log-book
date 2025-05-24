import streamlit as st
import sqlite3, json, os, random
from datetime import datetime
import pandas as pd
import re
import json


# LLM & CrewAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from crewai import Agent, Task, Crew



# --- Data Loading & Mapping ---
def load_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)

def build_verse_map(verses, translations, commentaries):
    """Map (chapter_number, verse_number) to all data."""
    verse_map = {}
    for v in verses:
        key = (int(v['chapter_number']), int(v['verse_number']))
        translation = [t for t in translations if t['verseNumber'] == key[1]]
        commentary = [c for c in commentaries if c['verseNumber'] == key[1]]
        verse_map[key] = {
            'verse': v,
            'translation': translation,
            'commentary': commentary
        }
    return verse_map

# --- Load Data ---
verses = load_json('verse.json')
chapters = load_json('chapters.json')
commentaries = load_json('commentary.json')
translations = load_json('translation.json')
verse_map = build_verse_map(verses, translations, commentaries)

def get_random_verse_key():
    v = random.choice(verses)
    return int(v['chapter_number']), int(v['verse_number'])

def get_verse_context(chapter, verse):
    return verse_map.get((chapter, verse), {})

def get_random_verse(return_full=False):
    if not verses:
        raise ValueError("No verses loaded!")
    v = random.choice(verses)
    if return_full:
        return v
    return int(v['chapter_number']), int(v['verse_number']), v['text']

def get_mp3_path(chapter, verse, base_dir='verse_recitation'):
    return os.path.join(base_dir, str(chapter), f"{verse}.mp3")

def show_geeta_advice(chapter=None, verse=None, text=None):
    if chapter is None or verse is None or text is None:
        chapter, verse, text = get_random_verse()
    st.markdown(f"**Chapter {chapter}, Verse {verse}:**\n\n{text}")
    mp3_path = get_mp3_path(chapter, verse)
    if os.path.exists(mp3_path):
        with open(mp3_path, 'rb') as audio_file:
            st.audio(audio_file.read(), format='audio/mp3')
    else:
        st.warning("Recitation audio not found for this verse.")

def get_display_log(refined_log, raw_log):
    # Add more checks as needed for other generic LLM refusals
    if not refined_log or "I cannot provide a response that promotes or glorifies violence, harm, or illegal activities, including sexual violence and exploitation of a minor. Can I help with something else?" in refined_log or "as an AI" in refined_log:
        return raw_log
    return refined_log

import re
import json

def parse_advice_string(advice_string):
    """Extract emotional_state, relevant_verse, and advice from a JSON or JSON-like string."""
    try:
        data = json.loads(advice_string)
        return (
            data.get("emotional_state", ""),
            data.get("relevant_verse", ""),
            data.get("advice", "")
        )
    except Exception:
        # Fallback: Use regex to extract fields if not valid JSON
        emo = re.search(r'"?emotional_state"?\s*:\s*"([^"]+)"', advice_string)
        verse = re.search(r'"?relevant_verse"?\s*:\s*"([^"]+)"', advice_string)
        adv = re.search(r'"?advice"?\s*:\s*"([^"]+)"', advice_string)
        return (
            emo.group(1) if emo else "",
            verse.group(1) if verse else "",
            adv.group(1) if adv else advice_string
        )


# --- Database Setup ---
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect('data/logbook.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            location TEXT,
            mood TEXT,
            raw_log TEXT,
            refined_log TEXT,
            advice TEXT,
            verse TEXT,
            steps INTEGER,
            distance REAL
        )
    ''')
    conn.commit()
    return conn, c

def save_log(timestamp, location, mood, raw_log, refined_log, advice, verse, steps, distance):
    conn, c = init_db()
    c.execute('''
        INSERT INTO logs 
        (timestamp, location, mood, raw_log, refined_log, advice, verse, steps, distance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, location, mood, raw_log, refined_log, advice, verse, steps, distance))
    conn.commit()
    conn.close()

def get_all_logs():
    conn, c = init_db()
    c.execute('SELECT * FROM logs ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows

# --- LLM & CrewAI Setup ---

llm = ChatOllama(model="ollama/gemma3:1b", base_url="http://localhost:11434")

# Update research_template to enforce English output
research_template = PromptTemplate.from_template("""
You are a Bhagavad Gita research expert.
Your job is to analyze the user's log, detect the emotional state, and offer relevant advice based ONLY on the provided context (verse, translations, and commentaries).

**Instructions:**
1. If the User Log is in Hindi or Marathi, translate it into English before proceeding.
2. Use ONLY the provided verse text, translations, and commentaries for your advice.
3. Reflect the emotional tone honestly.
4. Advice should align with Bhagavad Gita wisdom.

**Input:**
                                                 

User Log:
{user_log}

Verse:
{verse_text}

Translations:
{translations}

Commentaries:
{commentaries}

**Output:** Provide a JSON in pure English, formatted like:

{{
  "emotional_state": "<A brief emotional summary — one line>",
  "relevant_verse": "<Verse reference (e.g., Chapter 2, Verse 47) or short quote>.",
  "advice": "<A simple, micro-advice sentence of 1-2 lines — clear, positive, actionable>"
}}
Example advice: "Focus on your actions, not the praise. Peace comes from within."

Keep it practical, clear, and aligned with the Gita’s teachings — no long discourses.
You are also a helpful text editor.
Refine the following user log by fixing grammar, spelling, and sentence structure, while preserving the original meaning.
Do not generate advice, commentary, or content not present in the log.
""")


writer_template = PromptTemplate.from_template("""
You are a motivational diary writer. 
Using the user's log, research output, and verse context — craft a beautiful, encouraging diary entry in English.

**Instructions:**
1. If any part of the User Log, Translations, or Commentaries is in Hindi or Marathi, translate it into English before use.
2. Use the research agent's analysis and advice exactly.
3. Maintain a positive, reflective, and warm tone.

**Input:**
                                               

User Log:
{user_log}

Research Agent Output (in JSON English format):
{research_output}

Verse:
{verse_text}

Translations:
{translations}

Commentaries:
{commentaries}

**Your diary entry must include:**
- A refined summary of the user's day.
- The micro Gita advice (short, 1-2 sentences)
- One motivational tip for the user.
- One good habit suggestion they can start tomorrow.

**Final Output:**  
A complete, thoughtful, English-only diary entry with no additional comments or headings.
Make it feel personal, human, and comforting — not formal or preachy.
You are also a helpful text editor.
Refine the following user log by fixing grammar, spelling, and sentence structure, while preserving the original meaning.
Do not generate advice, commentary, or content not present in the log.
""")


def research_agent_fn(inputs):
    """
    Analyze the user's log and provide Bhagavad Gita advice using only the provided context.
    Args:
        inputs (dict): {
            'user_log': str,
            'verse_text': str,
            'translations': list of dicts,
            'commentaries': list of dicts
        }
    Returns:
        str: LLM output as JSON string with keys: emotional_state, relevant_verse, advice
    """
    try:
        translations_text = "\n".join(
            [f"{t.get('authorName', 'Unknown')}: {t.get('description', '')}" for t in inputs.get('translations', [])]
        )
        commentaries_text = "\n".join(
            [f"{c.get('authorName', 'Unknown')}: {c.get('description', '')}" for c in inputs.get('commentaries', [])]
        )
        prompt = research_template.format(
            user_log=inputs.get('user_log', ''),
            verse_text=inputs.get('verse_text', ''),
            translations=translations_text,
            commentaries=commentaries_text
        )
        return llm(prompt)
    except Exception as e:
        return json.dumps({
            "emotional_state": "Could not analyze.",
            "relevant_verse": "",
            "advice": f"Error in research_agent_fn: {str(e)}"
        })

def writer_agent_fn(inputs):
    """
    Beautify the diary entry, add Gita advice, a motivational tip, and a good habit suggestion.
    Args:
        inputs (dict): {
            'user_log': str,
            'research_output': str (JSON from research_agent_fn),
            'verse_text': str,
            'translations': list of dicts,
            'commentaries': list of dicts
        }
    Returns:
        str: Motivational diary entry in English
    """
    try:
        translations_text = "\n".join(
            [f"{t.get('authorName', 'Unknown')}: {t.get('description', '')}" for t in inputs.get('translations', [])]
        )
        commentaries_text = "\n".join(
            [f"{c.get('authorName', 'Unknown')}: {c.get('description', '')}" for c in inputs.get('commentaries', [])]
        )
        prompt = writer_template.format(
            user_log=inputs.get('user_log', ''),
            research_output=inputs.get('research_output', ''),
            verse_text=inputs.get('verse_text', ''),
            translations=translations_text,
            commentaries=commentaries_text
        )
        return llm(prompt)
    except Exception as e:
        return f"Error in writer_agent_fn: {str(e)}"



# --- CrewAI Pipeline ---


def run_crew(user_log, context):
    # Define agents
    research_agent = Agent(
        role="Gita Research Analyst",
        goal="Analyze the user's log and provide Gita advice.",
        backstory="Expert in analyzing emotions and mapping them to Bhagavad Gita teachings.",
        llm=llm,
        verbose=True
    )

    writer_agent = Agent(
        role="Motivational Logbook Writer",
        goal="Create a motivational diary entry with positive psychology and Gita wisdom.",
        backstory="Specialist in habit-building and motivational writing.",
        llm=llm,
        verbose=True
    )
    

    # Define tasks
    research_task = Task(
        description=research_template.format(
            user_log=user_log,
            verse_text=context['verse'].get('text', ''),
            translations="\n".join([f"{t['authorName']}: {t['description']}" for t in context.get('translation', [])]),
            commentaries="\n".join([f"{c['authorName']}: {c['description']}" for c in context.get('commentary', [])])
        ),
        agent=research_agent,
        name="research_task",
        expected_output="JSON with emotional_state, relevant_verse, advice"
    )

    writer_task = Task(
        description=writer_template.format(
            user_log=user_log,
            research_output="{research_task}",
            verse_text=context['verse'].get('text', ''),
            translations="\n".join([f"{t['authorName']}: {t['description']}" for t in context.get('translation', [])]),
            commentaries="\n".join([f"{c['authorName']}: {c['description']}" for c in context.get('commentary', [])])
        ),
        agent=writer_agent,
        name="writer_task",
        expected_output="A beautiful diary entry with a good habit and motivational tip"
    )

    # Define Crew
    crew = Crew(
        agents=[research_agent, writer_agent],
        tasks=[research_task, writer_task],
        verbose=True
    )

    results = crew.kickoff()

    # Find the output for the research task
    research_output = next(task.raw for task in results.tasks_output if task.name == "research_task")

    # Find the output for the writer task
    diary_entry = next(task.raw for task in results.tasks_output if task.name == "writer_task")

    print("Research output:", research_output)
    print("Diary entry:", diary_entry)

    try:    
        research_json = json.loads(research_output)
        advice = research_json.get("advice", "")
        relevant_verse = research_json.get("relevant_verse", "")
    except Exception:
        advice, relevant_verse = research_output, ""

    return diary_entry, advice, relevant_verse, context

# --- Streamlit App ---
st.set_page_config(page_title="📓 Life Log Book", layout="wide")

tab1, tab2 = st.tabs(["📝 Daily Log Entry", "📖 Diary & Insights"])

with tab1:
    st.title("📝 Daily Log Entry")

    now = datetime.now()
    date_str = now.strftime("%A, %d %B %Y")
    time_str = now.strftime("%I:%M %p")
    location = "Leamington Spa, UK"

    st.markdown(f"**Date:** {date_str}")
    st.markdown(f"**Time:** {time_str}")
    st.markdown(f"**Location:** {location}")

    mood_options = ["😀 Happy", "😐 Neutral", "😔 Sad", "😡 Angry", "😰 Stressed", "🤩 Excited"]
    mood = st.radio("Mood:", mood_options, horizontal=True)

    user_log = st.text_area("Your thoughts:", placeholder="What's on your mind today? (You can write in Hindi or English)")

    steps = st.number_input("Steps taken today:", min_value=0, value=0)
    distance = st.number_input("Walk distance (km):", min_value=0.0, value=0.0, format="%.2f")

    if st.button("Save & Analyze"):
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        chapter, verse = get_random_verse_key()
        context = get_verse_context(chapter, verse)
        diary_entry, advice, relevant_verse, ctx = run_crew(user_log, context)
        save_log(timestamp, location, mood, user_log, diary_entry, advice, relevant_verse, steps, distance)
        st.success("Log saved successfully!")
        st.markdown("### Your Motivational Diary Entry")
        st.write(diary_entry)
        if advice or relevant_verse:
            st.markdown("### Gita Advice")
            st.write(advice)
            st.markdown("### Gita Verse")
            st.write(relevant_verse)
            if ctx.get('translation'):
                st.markdown("**Translation(s):**")
                for t in ctx['translation']:
                    st.write(f"- {t['authorName']}: {t['description']}")
            if ctx.get('commentary'):
                st.markdown("**Commentary(s):**")
                for c in ctx['commentary']:
                    st.write(f"- {c['authorName']}: {c['description']}")
        # Play audio for the verse used in the advice:
        if ctx and ctx.get('verse'):
            v = ctx['verse']
            chapter = int(v['chapter_number'])
            verse = int(v['verse_number'])
            text = v['text']
            show_geeta_advice(chapter, verse, text)
        else:
            show_geeta_advice()  # fallback to random
with tab2:
    st.title("📖 Diary & Insights")
    logs = get_all_logs()

    if logs:
        for log in logs:
            (_id, timestamp, location, mood, raw_log, refined_log, advice, verse, steps, distance) = log

            display_log = get_display_log(refined_log, raw_log)
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            date_str = dt.strftime("%A, %d %B %Y")
            time_str = dt.strftime("%I:%M %p")

            st.markdown(f"""
                <div style="border-radius: 10px; border: 1px solid #e0e0e0; background: #f8fafd; margin-bottom: 2em; padding: 1.1em 1.3em;">
                  <div style="color: #666; font-size: 0.95em; margin-bottom: 0.7em;">
                    <b>Date:</b> {date_str} &nbsp; <b>Time:</b> {time_str} &nbsp; <b>Location:</b> {location}
                  </div>
                  <div style="margin-bottom: 0.7em;">
                    <b>Diary Entry:</b>
                    <div style="margin-top:0.3em; color:#222;">{display_log}</div>
                  </div>
            """, unsafe_allow_html=True)

            # --- Parse and display advice fields in separate boxes ---
            emotional_state, relevant_verse, advice_text = parse_advice_string(advice)

            st.markdown("**Gita Advice**")
            st.text_area("Emotional State", emotional_state, height=70, key=f"emo_{_id}")
            st.text_area("Relevant Verse", relevant_verse, height=70, key=f"verse_{_id}")
            st.text_area("Advice", advice_text, height=100, key=f"advice_{_id}")

            st.markdown(f"""
                <div style="color: #888; font-size: 0.92em; margin-top: 1em;">
                    <b>Mood:</b> {mood} &nbsp; | &nbsp; <b>Steps:</b> {steps} &nbsp; | &nbsp; <b>Distance:</b> {distance} km
                </div>
                <hr style="margin-top:2em; margin-bottom:2em;">
            """, unsafe_allow_html=True)
    else:
        st.info("No logs found yet. Start adding your entries!")







