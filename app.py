# app.py — Product Recommender (finalized)
import os
import json
import time
import socket
import hashlib
import logging
import pickle
import requests
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify
import numpy as np

# ====== CONFIG & LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("product_recommender")

APP_ROOT = Path(__file__).parent.resolve()
ARTIFACTS_DIR = APP_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-3.3b-instruct:free")

# LLM cache file
LLM_CACHE_FILE = ARTIFACTS_DIR / "llm_explanations_cache.pkl"

# Flask app
app = Flask(__name__, template_folder="templates")

# ====== LOAD ARTIFACTS (pickles produced by notebook) ======
def load_artifacts():
    required = [
        "product_df.pkl",
        "user_item.pkl",
        "item_ids.pkl",
        "item_item_sim_matrix.pkl",
        "apriori_rules_processed.pkl"
    ]
    missing = [f for f in required if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts in {ARTIFACTS_DIR}: {missing}")

    artifacts = {}
    artifacts["product_df"] = pickle.load(open(ARTIFACTS_DIR / "product_df.pkl", "rb"))
    artifacts["user_item"] = pickle.load(open(ARTIFACTS_DIR / "user_item.pkl", "rb"))
    artifacts["item_ids"] = pickle.load(open(ARTIFACTS_DIR / "item_ids.pkl", "rb"))
    artifacts["item_item_sim_matrix"] = pickle.load(open(ARTIFACTS_DIR / "item_item_sim_matrix.pkl", "rb"))
    artifacts["rules_processed"] = pickle.load(open(ARTIFACTS_DIR / "apriori_rules_processed.pkl", "rb"))

    # safe mapping description -> product id (pick most frequent if duplicates)
    df = artifacts["product_df"]
    try:
        artifacts["desc_to_pid"] = df.groupby("Description")["Product_id"].agg(lambda s: s.value_counts().index[0]).to_dict()
    except Exception:
        # fallback: try direct mapping if unique
        artifacts["desc_to_pid"] = dict(zip(df["Description"].astype(str), df["Product_id"].astype(str)))

    return artifacts

ARTS = load_artifacts()
product_df = ARTS["product_df"]
user_item = ARTS["user_item"]
item_ids = ARTS["item_ids"]
item_item_sim_matrix = ARTS["item_item_sim_matrix"]
rules_processed = ARTS["rules_processed"]
desc_to_pid = ARTS["desc_to_pid"]

# ====== LLM CACHE ======
try:
    llm_cache = pickle.load(open(LLM_CACHE_FILE, "rb"))
except Exception:
    llm_cache = {}

def _cache_key(customer_id, rec_descriptions, mode):
    key_src = json.dumps({"cid": str(customer_id), "recs": rec_descriptions, "mode": mode}, sort_keys=True)
    return hashlib.sha256(key_src.encode()).hexdigest()

def _save_llm_cache():
    try:
        pickle.dump(llm_cache, open(LLM_CACHE_FILE, "wb"))
    except Exception as e:
        logger.warning("Failed to persist LLM cache: %s", e)

# ====== PROMPT TEMPLATES & FEW-SHOT EXAMPLES ======
PROMPT_TEMPLATES = {
    "item_type": """
You are a helpful shopping assistant.
Customer recent purchases: {purchased}
We are recommending: {recommended}
Now focus on this product: "{product}"
Product features / short summary: {features}

Write a unique, customer-specific 2–3 sentence explanation:
1) Show how this product complements or is similar to the customer's past purchases.
2) Explain the concrete benefit the customer will get.
Avoid boilerplate phrases like "based on your past purchases". Output ONLY the explanation text.
""".strip(),

    "customer_type": """
You are a savvy shopping advisor.
Customer recent purchases: {purchased}
Recommended products: {recommended}
For the product: "{product}" (features: {features})

Write a unique 2-sentence explanation:
- Why this fits the customer's known preferences.
- What real value or use-case it provides.
Be specific, mention features or use-cases, and avoid generic templates.
""".strip(),

    "hybrid": """
You are an AI shopping concierge.
The customer has a purchase history: {purchased}
You have combined product affinity, recency, peer influence and price to recommend products.
Recommended product: "{product}"
Product features: {features}

In 2–3 short sentences:
- Explain which factors (affinity, recency, peer influence, price/utility) contributed and how.
- Explain why this product suits the customer's current needs.
Return only the explanation text.
""".strip(),

    "default": """
You are a helpful recommender.
Customer recent purchases: {purchased}
Product: "{product}"
Features: {features}

Provide a unique 2-sentence explanation connecting the product to the customer.
""".strip()
}

FEW_SHOT_EXAMPLES = [
    {
        "product": "Vintage Tea Tin",
        "example": "Because you’ve chosen classic kitchen pieces, this Vintage Tea Tin complements that aesthetic while preserving tea freshness. Its airtight seal and retro finish make it both practical and decorative."
    },
    {
        "product": "Childrens Apron - Red Spot",
        "example": "Ideal for family baking, this apron’s washable fabric and adjustable straps make it a safe, long-lasting choice for kids. It pairs well with the colorful kitchenware you often select."
    },
    {
        "product": "Ceramic Money Bank - Retro",
        "example": "A decorative tabletop accent that doubles as a practical savings tool, it suits your taste for functional vintage objects. Its sturdy ceramic build and charming pattern make it both collectible and useful."
    }
]

# ====== UTILITIES ======
def _host_resolves(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
        return True
    except Exception as e:
        logger.debug("DNS resolution failed for %s: %s", hostname, e)
        return False

def _get_product_text_features(description: str, max_len: int = 140) -> str:
    """
    Build a short features summary from product_df row(s). Truncate to keep prompts compact.
    If product_df contains additional fields like 'UnitPrice' or 'Category', use them.
    """
    desc = (description or "").strip()
    rows = product_df[product_df["Description"] == desc]
    # Try to fetch some extra fields if present
    extras = []
    if not rows.empty:
        row = rows.iloc[0]
        for col in ["UnitPrice", "Country", "Category", "Brand"]:
            if col in row and not pd_isna(row[col]):
                extras.append(f"{col}: {row[col]}")
    base = desc if desc else "useful features and durable build"
    feat = base
    if extras:
        feat = f"{base} ({'; '.join(extras)})"
    if len(feat) > max_len:
        feat = feat[:max_len].rsplit(" ", 1)[0] + "..."
    return feat

def pd_isna(x):
    return x is None or (hasattr(x, "isna") and x.isna())

# ====== RECOMMENDERS (unchanged core logic) ======
def recommend_item_based_cf(customer_id, top_n=10):
    if customer_id not in user_item.index:
        return []
    user_vector = user_item.loc[customer_id].values
    scores = user_vector.dot(item_item_sim_matrix)
    purchased_mask = user_vector > 0
    scores[purchased_mask] = -np.inf
    top_idxs = np.argsort(-scores)[:top_n]
    recs = []
    for idx in top_idxs:
        pid = item_ids[idx]
        desc_row = product_df.loc[product_df["Product_id"] == pid, "Description"]
        desc = desc_row.values[0] if not desc_row.empty else ""
        recs.append({"Product_id": str(pid), "Description": desc, "score": float(scores[idx])})
    return recs

def apriori_recommend_from_history(user_history_descriptions, top_k=10):
    user_set = set([d.strip() for d in user_history_descriptions if isinstance(d, str)])
    candidate_scores = defaultdict(float)
    supporting = defaultdict(list)
    for r in rules_processed:
        if set(r.get("antecedents", [])).issubset(user_set):
            boost = float(r.get("confidence", 0.0)) * float(r.get("lift", 0.0))
            for cons in r.get("consequents", []):
                candidate_scores[cons] += boost
                supporting[cons].append(r)
    results = sorted(
        [{"Description": cons, "score": score, "rule_count": len(supporting[cons]), "rules": supporting[cons]}
         for cons, score in candidate_scores.items()],
        key=lambda x: -x["score"]
    )
    mapped = []
    for r in results[:top_k]:
        pid = desc_to_pid.get(r["Description"], None)
        mapped.append({"Product_id": str(pid) if pid else None, "Description": r["Description"], "score": float(r["score"])})
    return mapped

def recommend_hybrid(customer_id, top_n=10):
    if customer_id not in user_item.index:
        top_pop = product_df.sort_values("total_revenue", ascending=False).head(top_n)
        return [{"Product_id": str(r["Product_id"]), "Description": r["Description"], "score": float(r["total_revenue"])} for _, r in top_pop.iterrows()]
    cf = recommend_item_based_cf(customer_id, top_n=500)
    apr = apriori_recommend_from_history(get_user_history_descriptions(customer_id), top_k=500)
    cf_map = {c["Product_id"]: c["score"] for c in cf}
    apr_map = {desc_to_pid.get(a["Description"], "X"): a["score"] for a in apr}
    candidates = set(list(cf_map.keys()) + list(apr_map.keys()))
    scored = []
    for pid in candidates:
        desc_vals = product_df.loc[product_df["Product_id"] == pid, "Description"].values
        desc = desc_vals[0] if len(desc_vals) else ""
        score = 0.7 * cf_map.get(pid, 0.0) + 0.3 * apr_map.get(pid, 0.0)
        scored.append({"Product_id": str(pid), "Description": desc, "score": float(score)})
    return sorted(scored, key=lambda x: -x["score"])[:top_n]

# ====== USER HISTORY HELPERS ======
def get_user_history(customer_id):
    if customer_id not in user_item.index:
        return [], None
    user_row = user_item.loc[customer_id]
    purchased = user_row[user_row > 0].sort_values(ascending=False)
    history = [{"Description": product_df.loc[product_df["Product_id"] == pid, "Description"].values[0],
                "Product_id": str(pid), "Quantity": int(q)} for pid, q in purchased.items()]
    return history, None

def get_user_history_descriptions(customer_id):
    history, _ = get_user_history(customer_id)
    return [h["Description"] for h in history]

# ====== LLM CALLS: per-product prompt + paraphrase retry + fallback ======
def render_prompt(mode, purchased_list, recommended_list, product, features):
    tpl = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["default"])
    purchased = ", ".join(purchased_list[:8]) if purchased_list else "No prior purchases"
    recommended = ", ".join([r for r in recommended_list[:8]]) if recommended_list else product
    return tpl.format(purchased=purchased, recommended=recommended, product=product, features=features)

def _call_openrouter_with_prompt(prompt_text, max_tokens=220):
    if not OPENROUTER_API_KEY:
        return None, "no_api_key"
    if not _host_resolves("openrouter.ai"):
        return None, "dns_fail"

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a creative retail assistant who writes personalized recommendations."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.85
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=25)
    except Exception as e:
        logger.warning("OpenRouter network error: %s", e)
        return None, str(e)

    if resp.status_code != 200:
        logger.warning("OpenRouter status %s: %s", resp.status_code, resp.text[:400])
        return None, f"status_{resp.status_code}"

    try:
        data = resp.json()
        # typical OpenRouter returns choices[0].message.content
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning("OpenRouter parse error: %s", e)
        content = ""

    # Paraphrase retry if too short
    if not content or len(content.split()) < 6:
        paraphrase_prompt = prompt_text + "\n\nIf your previous response was generic, paraphrase in a different style and be more specific (2–3 sentences)."
        payload["messages"][1]["content"] = paraphrase_prompt
        try:
            resp2 = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=25)
            if resp2.status_code == 200:
                data2 = resp2.json()
                content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content2:
                    content = content2
        except Exception as e:
            logger.warning("OpenRouter paraphrase error: %s", e)

    return content if content else None, None

def _fallback_explanation(product_name: str, product_features_text: str, user_history: list):
    # create a more specific fallback explanation using features & small user-history snippet
    snippet = ""
    if user_history:
        snippet = " Since you recently bought " + ", ".join(user_history[:3]) + ","
    return (f"{product_name} pairs well with items in your recent purchases.{snippet} "
            f"It offers {product_features_text}. This combination makes it practical and a great match for your needs.")

def generate_explanation_openrouter_per_product(customer_id, recommended_products, user_history, mode="hybrid"):
    """
    Generate unique explanations for each product. Returns list of {product, explanation}.
    Uses template rendering + per-product LLM calls, with caching and fallback.
    """
    rec_descs = [r.get("Description", "") for r in recommended_products]
    cache_key = _cache_key(customer_id, rec_descs, mode)
    if cache_key in llm_cache:
        logger.info("LLM cache hit: %s", cache_key[:8])
        return llm_cache[cache_key]

    history_list = user_history if isinstance(user_history, list) else list(user_history or [])
    explanations = []

    # Build the recommended list as string for prompt context
    recommended_list = rec_descs

    for r in recommended_products:
        desc = r.get("Description", "")
        features = _get_product_text_features(desc)
        prompt_text = render_prompt(mode, history_list, recommended_list, desc, features)
        # optionally prepend few-shot examples
        few_shot_block = "\n\n".join([f"Product: {ex['product']}\nExplanation: {ex['example']}" for ex in FEW_SHOT_EXAMPLES])
        full_prompt = few_shot_block + "\n\n" + prompt_text

        content, err = _call_openrouter_with_prompt(full_prompt)
        if content:
            # ensure 2-3 sentences
            sentences = [s.strip() for s in content.replace("\n", " ").split('.') if s.strip()]
            if len(sentences) > 3:
                content = '. '.join(sentences[:3]).strip() + '.'
            elif len(sentences) == 0:
                content = _fallback_explanation(desc, features, history_list)
            explanations.append({"product": desc, "explanation": content})
        else:
            # fallback
            fb = _fallback_explanation(desc, features, history_list)
            explanations.append({"product": desc, "explanation": fb})

        # polite pacing to avoid throttling / rate-limit issues
        time.sleep(1.1)

    # Post-process to ensure uniqueness (simple)
    seen = {}
    for i, ex in enumerate(explanations):
        txt = ex["explanation"].strip()
        if txt in seen:
            # tweak duplicate by appending short product-specific clause
            explanations[i]["explanation"] = txt + f" Notably, {ex['product']} stands out with its { _get_product_text_features(ex['product'], max_len=60) }."
        else:
            seen[txt] = True

    # cache & persist
    llm_cache[cache_key] = explanations
    _save_llm_cache()

    return explanations

# ====== HIGH-LEVEL WRAPPER (used by Flask) ======
def recommend_for_customer(customer_id, n=10, rec_type="hybrid", include_explanations=False):
    """
    Returns dict: {customer_id, recommendations: [...], purchase_history: [...], explanations: [...]}
    rec_type: "hybrid", "item_cf", "apriori", "popular"
    """
    # coerce id type where appropriate
    try:
        cid = int(customer_id)
    except Exception:
        try:
            cid = float(customer_id)
        except Exception:
            cid = customer_id

    if cid not in user_item.index:
        top_pop = product_df.sort_values("total_revenue", ascending=False).head(n)
        results = [{"Product_id": str(r["Product_id"]), "Description": r["Description"], "score": float(r.get("total_revenue", 0.0))} for _, r in top_pop.iterrows()]
    else:
        if rec_type == "item_cf":
            results = recommend_item_based_cf(cid, top_n=n)
        elif rec_type == "apriori":
            results = apriori_recommend_from_history(get_user_history_descriptions(cid), top_k=n)
        else:
            # default hybrid
            results = recommend_hybrid(cid, top_n=n)

    history, country = get_user_history(cid)
    history_descs = [h["Description"] for h in history][:8]

    explanations = None
    if include_explanations and results:
        explanations = generate_explanation_openrouter_per_product(cid, results, history_descs, mode=rec_type)

    return {
        "customer_id": str(customer_id),
        "found_in_data": cid in user_item.index,
        "country": country,
        "purchase_history": history,
        "recommendations": results,
        "explanations": explanations
    }

# ====== FLASK ENDPOINTS ======
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    payload = request.get_json(force=True)
    customer_id = str(payload.get("customer_id", "")).strip()
    if not customer_id:
        return jsonify({"error": "customer_id required"}), 400

    rec_type = payload.get("rec_type", "hybrid")
    n = int(payload.get("n", 10))
    include_explanations = bool(payload.get("explain", True))

    try:
        resp = recommend_for_customer(customer_id, n=n, rec_type=rec_type, include_explanations=include_explanations)
        return jsonify(resp)
    except Exception as e:
        logger.exception("Error in recommendation: %s", e)
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

# ====== RUN ======
if __name__ == "__main__":
    # In production, run via gunicorn; debug=False for safety
    app.run(host="0.0.0.0", port=5000, debug=False)
