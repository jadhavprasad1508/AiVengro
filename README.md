🚀 Product Recommendation System Powered by LLM

A Hybrid AI Recommendation Engine combining Machine Learning and LLM-powered Explainability

📌 Project Overview

The Product Recommendation System Powered-by-LLM is an end-to-end, full-stack AI application designed to recommend highly relevant products based on a customer’s purchase history. The system uses a hybrid ML pipeline—Collaborative Filtering, Content-Based Filtering, Cosine Similarity, and Apriori Association rules—enhanced with LLaMA-3 (OpenRouter API) to generate unique, personalized natural-language explanations for each recommended product.

Built using Python, Flask, Pandas, Scikit-learn, HTML/CSS/JS, this project demonstrates a production-oriented architecture capable of real-world deployment for retail analytics, cross-sell/upsell automation, and customer personalization.

⚙️ Features

🔍 Personalized product recommendations using:
Collaborative Filtering (Item-Based)
Content-Based Filtering
Cosine Similarity
Apriori "Also Bought" rules
Hybrid scoring engine

🧠 LLM-Generated Explanations using LLaMA-3 via OpenRouter API
📊 Clean UI displaying purchase history, quantities, and customer country
⚡ Real-time API response with explanation toggle
🖥️ Modern, responsive frontend with a clear recommendation layout
🧩 Modular and scalable backend design
🔐 Env-based configuration for API keys

📂 Folder Structure
project-root/
│
├── app.py                   # Flask backend application
├── recommender.py           # ML logic & recommendation engine
├── templates/
│   └── index.html           # Frontend UI
├── static/
│   └── css/ (optional)      # Styling files
├── artifacts/
│   ├── product_map.pkl
│   ├── similarity_matrix.pkl
│   ├── apriori_rules.pkl
│   └── other ML assets...
├── requirements.txt
├── Notebook.ipynb           # Data prep + artifact generation notebook
├── .env.example             # Example environment variables
└── README.md

🛠️ Tech Stack

Languages: Python, JavaScript, HTML, CSS
Frameworks/Libraries: Flask, Pandas, NumPy, Scikit-learn, mlxtend
AI/ML Models: Collaborative Filtering, Apriori, Cosine Similarity
LLM: LLaMA-3 via OpenRouter API
Tools: VS Code, Jupyter Notebook, Conda, Git, GitHub

🧪 Installation Instructions
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Recommendation.git
cd Recommendation

2️⃣ Create Virtual Environment
conda create -n ai_recommender python=3.10
conda activate ai_recommender

or

python -m venv venv
venv\Scripts\activate    # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Create .env File
Create a file named .env in the project folder:

OPENROUTER_API_KEY=
OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct

5️⃣ Generate ML Artifacts (Optional)
If artifacts are not included, run the notebook:
jupyter notebook Notebook.ipynb

6️⃣ Run the Application
python app.py

Your app will start at:
http://127.0.0.1:5000

▶️ Usage Guidelines

Enter a Customer ID in the input box.
Select a Recommendation Type (Hybrid, Item-CF, Apriori, etc.).
Click “Generate Recommendations”.

View:

Purchase history (with quantities)
Customer's country
Ranked product suggestions
Toggle LLM Explanations to reveal unique, contextual insights per product.

📈 Results & Performance

Trained on 540K+ transactions and 4,372 customers
LLM explains why each product was recommended in 2–3 sentences

📧 Contact

For queries, collaborations, or opportunities:

Prasad Jadhav
📩 Email: prasadjadhav71017@gmail.com
🔗 LinkedIn:https://www.linkedin.com/in/prasadjadhavdatascience/

