# GPT-2 Text Generation with Hugging Face 🤖📝

A simple script that loads a pre-trained GPT-2 model and generates text from a prompt using top-k sampling with different temperatures.

---

## 📌 Objective

Demonstrate:
- Loading a pre-trained transformer (`gpt2`)
- Text generation with Hugging Face's `transformers` library
- Sampling with `top_k=50` and varying `temperature` (0.7 and 1.0)

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install torch transformers

▶️ How to Run

python generate.py

📤 Output

A file named generated_outputs.txt will be created with two generated samples:

=== Output with temperature=0.7 ===
Once upon a time...

=== Output with temperature=1.0 ===
Once upon a time...

🧠 Sampling Settings
Parameter	Value
Model	gpt2
Max Tokens	50
Top-k	50
Temperatures	0.7, 1.0

📁 Project Files

gpt2-text-generation/
├── generate.py              # Main script
├── generated_outputs.txt    # Generated samples
└── requirements.txt         # Dependencies (optional)
