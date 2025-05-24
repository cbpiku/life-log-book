# 📓 Life Log Book

A motivational diary and daily logbook app powered by Bhagavad Gita wisdom and modern AI.  
Reflect on your day, track your mood and activities, and receive personalized advice and encouragement—rooted in ancient philosophy and delivered with the help of cutting-edge language models.

---

## ✨ Features

- **Daily Logging:** Record your thoughts, mood, steps, and distance walked each day.
- **AI-Powered Insights:** Receive practical, Gita-inspired advice based on your daily reflections.
- **Emotional Analysis:** The app detects your emotional state and matches it with relevant verses and commentaries.
- **Motivational Diary Entries:** Get beautifully crafted diary entries with motivational tips and habit suggestions.
- **Audio Recitations:** Listen to verse recitations for deeper engagement.
- **Multi-language Support:** Write your entries in English, Hindi, or Marathi—AI will translate and respond in English.
- **Data Privacy:** Your logs are stored locally for your privacy.

---

## 🚀 Getting Started

### Prerequisites

- [Python 3.12.6](https://www.python.org/downloads/)
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com/) running the `gemma3:1b` model (for LLM features)
- Project data files: `verse.json`, `chapters.json`, `commentary.json`, `translation.json`, and verse recitations in `verse_recitation/`

### Installation

1. **Clone the repository:**

git clone https://github.com/yourusername/life-log-book.git

cd life-log-book


2. **Install dependencies:**

Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt


3. **Run Ollama with the required model:**
- Make sure Ollama is running and has the `gemma3:1b` model available.
- Update the `base_url` in `app.py` if your Ollama instance is remote or uses a different port.

4. **Start the app:**
python run.py


---

## 🖥️ Usage

- **Daily Log Entry:**  
Go to the "Daily Log Entry" tab, select your mood, write your thoughts (in English, Hindi ), and log your steps/distance. Click "Save & Analyze" to get your motivational diary entry and Gita advice.
- **Diary & Insights:**  
Review all your previous entries, emotional states, advice, and motivational tips in the "Diary & Insights" tab.

---

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request to suggest improvements or new features.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Inspired by the timeless wisdom of the Bhagavad Gita.
- Built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), and [Ollama](https://ollama.com/).

---

Start your journey of self-reflection and growth—**one log at a time!**

---

**Note:**  
Replace `yourusername` in the clone URL with your actual GitHub username, and ensure you have all required data files in place.




