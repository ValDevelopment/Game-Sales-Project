import ast
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack, vstack

st.set_page_config(page_title="Publisher Recommender", layout="wide")
st.title("Publisher Recommender")

@st.cache_data
def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    def parse_list(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        s = str(x).strip()
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []

    if "tags" not in df.columns:
        raise ValueError("CSV must include a 'tags' column (list stored as string).")

    df["tags"] = df["tags"].apply(parse_list)

    if "publisher" not in df.columns:
        raise ValueError("CSV must include a 'publisher' column.")

    df["publisher"] = df["publisher"].fillna("").astype(str).str.strip()
    df = df[df["publisher"] != ""]

    if "price" not in df.columns:
        raise ValueError("CSV must include a 'price' column.")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median())

    df["tag_text"] = df["tags"].apply(lambda xs: " ".join(xs) if isinstance(xs, list) else "")

    if "steamUrl" not in df.columns:
        df["steamUrl"] = ""

    return df

@st.cache_resource
def fit_model(pairs: pd.DataFrame, price_weight: float = 0.35, min_games: int = 3):
    tfidf = TfidfVectorizer(min_df=3, max_df=0.8)
    X_tags = tfidf.fit_transform(pairs["tag_text"])

    scaler_price = StandardScaler(with_mean=False)
    X_price = scaler_price.fit_transform(pairs[["price"]].to_numpy())
    X_price = csr_matrix(X_price)

    X_all = hstack([X_tags, price_weight * X_price]).tocsr()

    pubs = pairs["publisher"].astype(str).values
    unique_pubs, counts = np.unique(pubs, return_counts=True)
    pub_counts = dict(zip(unique_pubs, counts))

    kept_pubs = [p for p in unique_pubs if pub_counts[p] >= min_games]

    profiles = []
    for pub in kept_pubs:
        idx = np.where(pubs == pub)[0]
        vec = X_all[idx].sum(axis=0) / len(idx)
        profiles.append(csr_matrix(vec))

    P = vstack(profiles).tocsr()  
    return tfidf, scaler_price, X_all, P, kept_pubs, pub_counts

def recommend(pairs, tfidf, scaler_price, X_all, P, kept_pubs, pub_counts, tags, price, top_k=10, price_weight=0.35):
    tag_text = " ".join(tags)
    v_tags = tfidf.transform([tag_text])

    v_price = scaler_price.transform([[float(price)]])
    v_price = csr_matrix(v_price)

    v_all = hstack([v_tags, price_weight * v_price]).tocsr()

    sims = cosine_similarity(v_all, P).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]

    recs = []
    for i in top_idx:
        pub = kept_pubs[i]
        recs.append((pub, float(sims[i]), int(pub_counts[pub])))
    return recs, v_all

def get_examples(pairs, X_all, v_all, publisher, n=3):
    idx = pairs.index[pairs["publisher"] == publisher].to_numpy()
    if len(idx) == 0:
        return pd.DataFrame(columns=["name", "steamUrl", "price"])

    sims = cosine_similarity(v_all, X_all[idx]).ravel()
    top_local = idx[np.argsort(sims)[::-1]]

    ex = (
        pairs.loc[top_local, ["steamId", "name", "steamUrl", "price"]]
             .drop_duplicates(subset=["steamId"])
             .head(n)
    )
    return ex

# UI

from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "pixel_games_sales_pairs.csv"

try:
    pairs = load_pairs(DATA_PATH)
except Exception as e:
    st.error("Failed to load application data.")
    st.caption(f"Internal error: {e}")
    st.stop()


st.title("Indie Publisher Recommender (Tags + Price)")
st.write("Pick tags and a price point. The app returns publishers with similar catalogs and example games.")

try:
    pairs = load_pairs(DATA_PATH)
except Exception as e:
    st.error(f"Could not load {DATA_PATH}. Error: {e}")
    st.stop()

all_tags = sorted({t for tags in pairs["tags"] for t in tags})
default_tags = [t for t in ["Roguelike", "Deckbuilding", "Card Game", "Pixel Graphics", "Strategy"] if t in all_tags]

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    selected_tags = st.multiselect("Tags", all_tags, default=default_tags)
with col2:
    price = st.number_input("Price ($)", min_value=0.0, max_value=100.0, value=14.99, step=0.50)
with col3:
    top_k = st.slider("Top K publishers", 5, 25, 10)

price_weight = st.slider("Price influence", 0.0, 1.0, 0.35, 0.05)
min_games = st.slider("Min games per publisher", 1, 20, 3, 1)

tfidf, scaler_price, X_all, P, kept_pubs, pub_counts = fit_model(pairs, price_weight=price_weight, min_games=min_games)

if st.button("Recommend", type="primary"):
    if len(selected_tags) == 0:
        st.warning("Select at least one tag.")
        st.stop()

    recs, v_all = recommend(
        pairs, tfidf, scaler_price, X_all, P, kept_pubs, pub_counts,
        selected_tags, price, top_k=top_k, price_weight=price_weight
    )

    df_recs = pd.DataFrame(recs, columns=["Publisher", "Similarity Score", "Published Games"])

    st.subheader("Recommended Publishers:")
    for pub, score, n_games in recs:
        with st.expander(f"{pub} — Similarity Score = {score:.3f} — Games = {n_games}"):
            ex = get_examples(pairs, X_all, v_all, pub, n=3)
            if ex.empty:
                st.write("No examples found.")
            else:
                for _, r in ex.iterrows():
                    st.markdown(f"- **{r['name']}** (${r['price']:.2f}) — [Steam link]({r['steamUrl']})")


st.success("Loaded data successfully.")
c1, c2, c3 = st.columns(3)
c1.metric("Rows (pairs)", f"{pairs.shape[0]:,}")
c2.metric("Unique games", f"{pairs['steamId'].nunique():,}" if "steamId" in pairs.columns else "N/A")
c3.metric("Unique publishers", f"{pairs['publisher'].nunique():,}")

# data preview
st.subheader("Data preview")
preview_cols = [c for c in ["steamId", "name", "publisher", "price", "steamUrl", "tags"] if c in pairs.columns]
st.dataframe(pairs[preview_cols].head(20), use_container_width=True)