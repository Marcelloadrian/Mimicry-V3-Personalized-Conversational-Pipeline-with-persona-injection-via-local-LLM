# Mimicry V3 — Hybrid Intent Classification & Local LLM Pipeline

> *Paper: Personalized Mimic Chatbot Based on WhatsApp and IG Conversation History Using a Hybrid Intent Classification and Local LLM Pipeline*

Mimicry V3 is a full-scale hybrid pipeline implementation that combines intent classification efficiency with the generative power of local LLMs[cite: 3]. This project is designed to imitate a specific user's conversational style with higher contextual accuracy through intent-aware prompting[cite: 3].

### Core Architecture
1. **Hybrid Intent Classifier**: Utilizes TF-IDF with a comparative classifier suite (Naive Bayes, SVM, Random Forest) to identify conversation topics in real-time[cite: 3].
2. **JSON Knowledge Base**: A structured storage system mapping intent labels to persona context, tone, and specific behavioral traits[cite: 3].
3. **Persona Modeling**: Integrates hand-fed data (WhatsApp/IG) using few-shot prompting to maintain consistency for the "Marcel" persona[cite: 3].
4. **Local LLM Engine**: Employs **Gemma 2B** as the generative core, optimized for local deployment on hardware like Google Colab T4[cite: 3].

### Key Features
* **Reproducibility**: Locked seed (42) to ensure consistent experimental results[cite: 3].
* **Smart Fallback**: Automated fallback mechanisms for scenarios where intents are undetected or LLM responses are insufficient[cite: 3].
* **Comparative Evaluation**: Automated benchmarking of classifiers to automatically select the model with the highest cross-validation accuracy[cite: 3].
* **Interactive Pipeline**: Supports interactive chat mode with intent-debugging capabilities via the `!intent` command[cite: 3].

### Project Structure
* `MARCEL_HANDFEED_SESI[1-3].json`: Input conversation data serving as the persona backbone[cite: 3].
* `mimicry_v3_hybrid.ipynb`: Main execution notebook containing preprocessing, training, and inference logic[cite: 3].
* `Intent Taxonomy`: Defined topic categories including Identity, Mood, Philosophy, Values, etc[cite: 3].

### Installation & Usage
1. **Setup**: Ensure the environment has GPU access (Colab T4 recommended)[cite: 3].
2. **Dependencies**: Requires `transformers`, `accelerate`, `bitsandbytes`, and `scikit-learn`[cite: 3].
3. **Execution**:
   - Run **Cell 6** to train and evaluate the intent classifier[cite: 3].
   - Run **Cell 8** to load Gemma 2B into memory[cite: 3].
   - Use `generate_response()` for inference or run **Cell 12** to enter interactive chat mode[cite: 3].

### Performance Evaluation
V3 includes automated performance benchmarking comparing Naive Bayes, SVM, and Random Forest models based on cross-validation accuracy, ensuring the pipeline utilizes the most stable classification model for triggering context-aware LLM prompts[cite: 3].
