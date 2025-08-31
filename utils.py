
import re
from collections import Counter

import numpy as np
import pandas as pd

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import matplotlib

from sklearn.feature_extraction.text import CountVectorizer



SUSPECT_NAME_RE = r"(?:_after$|_sum$|^labs_sum$|delivery|birth|outcome|eclampsia|preeclampsia|gestational|hypertension|aspirin|match)"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (out.columns.str.strip().str.lower()
                   .str.replace(r"[^a-z0-9_]+","_", regex=True).str.strip("_"))
    return out


def simple_eda(df: pd.DataFrame, topn: int = 5):
    """
    Simple EDA – expects a DataFrame (already loaded).
    - Plots top-N missingness offenders (default 5) if any exist.
    - Returns the same DataFrame.
    """

    # --- Quick samples and info ---
    print('********** Simple EDA **********')
    print('\nData INFO')
    df.info()                        # dataframe info
    # --- Shape and dtypes ---
    print("\nShape\n:", df.shape)
    print("\nDtypes Counts:", df.dtypes.value_counts())

    for cand in ("y","target"):
        if cand in df.columns:
            TARGET = cand
            print(f"Label\target column name: {TARGET}")
            break
    if TARGET not in df.columns:
        raise SystemExit(f"Target column '{TARGET}' not found.")

    # --- Target prevalence ---
    prev = df[TARGET].mean()
    print(f"Target prevalence: {prev:.3%}")

    # --- Missingness ---
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss10 = miss[miss > 10].round(1)

    # --- Plot missingness (top N offenders) ---
    if not miss10.empty:
        print(f"\n=== PLOT: Columns with >10% missing (Top {topn}) ===\n")
        ax = miss10.head(topn).sort_values().plot(kind="barh", figsize=(8, 8))
        ax.set_xlabel("% missing")
        print(f"Number of columns with >10% missing values: {len(miss10)}")
        ax.set_title(f"Columns with >10% missing (Top {topn})")
        plt.tight_layout()
        plt.show()
    else:
        print("\n(no columns exceed 10% missing — skipping plot)")

    # --- Numeric and categorical columns ---
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(["Y"], errors="ignore")
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    print("Number of numeric columns:", len(num_cols), "Number of categorical columns:", len(cat_cols))

    # --- Quick describe of clinical_sheet ---
    print("\n(Name if categorical - Text column is: clinical_sheet)")
    print("\n(Describe text column:)")
    print(df.clinical_sheet.describe())

    return None



class UnitNormalizer():
    """
    Normalize lab/vital units to consistent scales.
    Applies safe heuristic corrections (only if values clearly out-of-range).
    """
    def __init__(self):
        self.corrections_ = {}

    def fit(self, X, y=None):
        # no learned parameters, but we can store detected corrections for reporting
        df = X.copy()
        self.corrections_ = {}
        
        def record(col, action):
            self.corrections_[col] = action

        if 'HCT' in df:
            if pd.to_numeric(df['HCT'], errors='coerce').quantile(0.95) <= 1.2:
                record('HCT','*100 (fraction→%)')
        if 'HGB' in df:
            if pd.to_numeric(df['HGB'], errors='coerce').quantile(0.95) > 35:
                record('HGB','/10 (g/L→g/dL)')
        if 'WBC' in df:
            if pd.to_numeric(df['WBC'], errors='coerce').quantile(0.99) > 100:
                record('WBC','/1000 (per µL→10^9/L)')
        if 'PLT' in df:
            if pd.to_numeric(df['PLT'], errors='coerce').quantile(0.99) > 2000:
                record('PLT','/1000 (per µL→10^9/L)')
        if 'RBC' in df:
            if pd.to_numeric(df['RBC'], errors='coerce').quantile(0.99) > 20:
                record('RBC','/10 (scale fix)')
        if 'NT_abs' in df:
            if pd.to_numeric(df['NT_abs'], errors='coerce').quantile(0.99) > 50:
                record('NT_abs','/10 or /1000 (likely µm→mm)')
        if 'pH-U' in df:
            if pd.to_numeric(df['pH-U'], errors='coerce').quantile(0.95) <= 1.5:
                record('pH-U','*14 (0–1→0–14 scale)')
        return self

    def transform(self, X):
        df = X.copy()
        for col, action in self.corrections_.items():
            s = pd.to_numeric(df[col], errors='coerce')
            if "HCT" in col and "*100" in action:
                df[col] = s * 100
            if "HGB" in col and "/10" in action:
                df[col] = s / 10.0
            if "WBC" in col and "/1000" in action:
                df[col] = s / 1000.0
            if "PLT" in col and "/1000" in action:
                df[col] = s / 1000.0
            if "RBC" in col and "/10" in action:
                df[col] = s / 10.0
            if "NT_abs" in col:
                if s.quantile(0.99) > 500:
                    df[col] = s / 1000.0
                else:
                    df[col] = s / 10.0
            if "pH-U" in col and "*14" in action:
                df[col] = s * 14.0
        return df

class BinaryFlagsAdding():
  ## adding binary flags using knowing limits for Lab tests
  ## adding age binay col for risk factor age_threshold set to- >38 years
    RANGES = {
        "WBC": (3.0, 30.0),
        "HGB": (10.0, 16.0),
        "HCT": (30.0, 48.0),
        "PLT": (100.0, 800.0),
        "MPV": (7.0, 13.0),
        "MCH": (24.0, 34.0),
        "MCHC": (30.0, 37.0),
        "RDW": (10.0, 20.0),
        "papp_a_MoM": (0.4, None),
        "b_hcg_MoM": (None, 2.0),
        "NT_MoM": (None, 1.5),
        "NT_abs": (None, 3.5),
        "Protein-U": (None, 0.0),
        "pH-U": (4.5, 8.5),
    }

    KEYWORDS = {
        "WBC": ["wbc", "white blood cells"],
        "HGB": ["hgb", "hemoglobin"],
        "HCT": ["hct", "hematocrit"],
        "PLT": ["plt", "platelets"],
        "MPV": ["mpv", "mean platelet volume"],
        "MCH": ["mch", "mean corpuscular hemoglobin"],
        "MCHC": ["mchc", "mean corpuscular hemoglobin concentration"],
        "RDW": ["rdw", "red cell distribution width"],
        "papp_a_MoM": ["papp a mom"],
        "b_hcg_MoM": ["b hcg mom", "bhcg mom"],
        "NT_MoM": ["nt mom", "nuchal translucency mom"],
        "NT_abs": ["nt abs", "nuchal translucency"],
        "Protein-U": ["protein u", "protein-u"],
        "pH-U": ["ph u", "ph-u"],
    }

    def __init__(self, age_threshold=38):
        self.age_threshold = age_threshold

    def add_flags(self, df):
        for canon, (low, high) in self.RANGES.items():
            # find first matching column
            matches = [c for c in df.columns if any(kw in c.lower() for kw in self.KEYWORDS.get(canon, []))]
            if not matches: 
                continue
            s = pd.to_numeric(df[matches[0]], errors="coerce")
            if low is not None:
                df[f"{canon}_low_flag"] = (s < low).astype(float)
            if high is not None:
                df[f"{canon}_high_flag"] = (s > high).astype(float)

        # age
        age_cols = [c for c in df.columns if "age" in c.lower()]
        if age_cols:
            s = pd.to_numeric(df[age_cols[0]], errors="coerce")
            df["age_over_38_flag"] = (s > self.age_threshold).astype(float)

        return df


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
  
    '''
    Adding the following features from Lab data:
    NLR, PLR — Ratios from CBC if neutrophils / lymphocytes / platelets columns are present.
    MAP, pulse_pressure — Derived if mean SBP/DBP values are available.
    hist4_count, hist24_count + hist_4_over_24 + new_vs_chronic_flag — Aggregated counts of diagnosis flags from the past 4 vs 24 months, their ratio, and a flag if new diagnoses appear only in the last 4 months.
    abnormal_lab_count — Count of all existing _low_flag / _high_flag lab abnormality indicators.
    '''
    X = df.copy()
    # --- CBC ratios (NLR, PLR) ---
    neut_cols = [c for c in X.columns if "neutrophils_1" in c.lower()]
    lymph_cols = [c for c in X.columns if "lymphocytes_1" in c.lower()]
    plt_cols  = [c for c in X.columns if "platelets" in c.lower() or "plt" in c.lower()]

    neut_col = neut_cols[0] if neut_cols else None
    lymph_col = lymph_cols[0] if lymph_cols else None
    plt_col = plt_cols[0] if plt_cols else None

    # NLR: neutrophil absolute / lymphocyte absolute
    if neut_col and lymph_col and "NLR" not in X:
        n = pd.to_numeric(X[neut_col], errors="coerce")
        l = pd.to_numeric(X[lymph_col], errors="coerce")
        X["NLR"] = (n / l).replace([np.inf, -np.inf], np.nan)

    # PLR: platelets / lymphocytes absolute
    if plt_col and lymph_col and "PLR" not in X:
        p = pd.to_numeric(X[plt_col], errors="coerce")
        l = pd.to_numeric(X[lymph_col], errors="coerce")
        X["PLR"] = (p / l).replace([np.inf, -np.inf], np.nan)


    # --- BP features (MAP, pulse pressure) ---
    sys_col = 'measure_blood_pressure_sys_mean'
    dia_col = 'measure_blood_pressure_dia_mean'

    if sys_col in X.columns and dia_col in X.columns:
        s = pd.to_numeric(X[sys_col], errors='coerce')
        d = pd.to_numeric(X[dia_col], errors='coerce')
        if 'pulse_pressure' not in X:
            X['pulse_pressure'] = s - d
        if 'MAP' not in X:
            X['MAP'] = d + (s - d) / 3.0

    # --- History aggregates (4_* vs 24_*) ---
    h4  = [c for c in X.columns if str(c).startswith('4_')]
    h24 = [c for c in X.columns if str(c).startswith('24_')]
    if h4 and 'hist4_count' not in X:
        X['hist4_count'] = pd.DataFrame({c: pd.to_numeric(X[c], errors='coerce').clip(0,1) for c in h4}).sum(axis=1)
    if h24 and 'hist24_count' not in X:
        X['hist24_count'] = pd.DataFrame({c: pd.to_numeric(X[c], errors='coerce').clip(0,1) for c in h24}).sum(axis=1)
    if 'hist4_count' in X and 'hist24_count' in X:
        if 'hist_4_over_24' not in X:
            X['hist_4_over_24'] = X['hist4_count'] / (X['hist24_count'] + 1e-6)
        if 'new_vs_chronic_flag' not in X:
            X['new_vs_chronic_flag'] = ((X['hist4_count'] > 0) & (X['hist24_count'] == 0)).astype(float)

    # --- Count existing abnormal lab flags ---
    flag_cols = [c for c in X.columns if c.endswith('_low_flag') or c.endswith('_high_flag')]
    if flag_cols and 'abnormal_lab_count' not in X:
        X['abnormal_lab_count'] = pd.DataFrame(
            {c: pd.to_numeric(X[c], errors='coerce').clip(0,1) for c in flag_cols}
        ).sum(axis=1)


    return X



##### FOR NLP & TEXT column EDA: ##########


# ===== Regex & Canon Definitions =====
SECTION_RE = re.compile(
    r"(?ms)^\s*([^\n:]{1,60})\s*[:\-–—]\s*(.*?)"
    r"(?=^\s*[^\n:]{1,60}\s*[:\-–—]\s*|^\s*שבוע\b.*$|$\Z)"
)
WEEK_RE = re.compile(r"(?m)^\s*שבוע(?:\s*(?:מס'?פר|הריון))?\s*[:\-–—]?\s*(\d{1,2})\s*(?:להריון)?\s*$")
INLINE_WEEK_RE = re.compile(r"\bשבוע\D{0,3}(\d{1,2})\b")

TOP7_RE = {
    "complaints":      [r"(?im)^\s*תלונות\b", r"(?im)^\s*דיווח\b"],
    "risk_factors":    [r"(?im)^\s*גורמי\s*סיכון\b", r"(?im)^\s*סיכונים\b", r"(?im)^\s*סיכון\b"],
    "findings":        [r"(?im)^\s*ממצאים\b", r"(?im)^\s*בדיק(?:ה|ות)\b"],
    "labs_imaging":    [r"(?im)^\s*תוצאות\b", r"(?im)^\s*מעבדה\b", r"(?im)^\s*הדמיה\b"],
    "medications":     [r"(?im)^\s*תרופות\b", r"(?im)^\s*טיפול\s*תרופתי\b", r"(?im)^\s*אספירין\b"],
    "vitals":          [r"(?im)^\s*לחץ\s*דם\b", r"(?im)^\s*דופק\b", r"(?im)^\s*BMI\b", r"(?im)^\s*משקל\b"],
    "recommendations": [r"(?im)^\s*המלצות\b", r"(?im)^\s*המשך\s*טיפול\b", r"(?im)^\s*מעקב\b"],
}
TOP7_RE = {k: [re.compile(p) for p in v] for k, v in TOP7_RE.items()}

HEADER_LINE_RE = re.compile(r"(?m)^\s*([א-תA-Za-z][^:\n]{0,40}?)\s*[:\-–—]\s*$")
SENT_SPLIT_RE = re.compile(r"[\.!\?\n]+")

# ===== Stopwords =====
HEB_STOP = {
    "של","על","עם","בלי","גם","אבל","אם","או","כי","זה","זו","זאת","כל","מאוד","יותר","פחות",
    "יש","אין","לא","כן","הוא","היא","הם","הן","אני","אתה","את","ללא","נגד","בעד","כך","כמו",
    "אך","וכן","וכו","וכו'","וכו’","לפי","מעל","מתחת","בין","עד","אצל","צריך","נדרש","ייתכן","יתכן"
}
EXTRA_STOP = {
    # section/scaffolding
    "תלונות","תלונות מטופלת","גורמי","סיכון","גורמי סיכון","ממצאים","בדיקה","בדיקות",
    "המטופלת","תוצאות","מעבדה","הדמיה","תרופות","טיפול","טיפול תרופתי","המלצות","מעקב","שבוע","להריון",
    # complaint triggers / fillers
    "מתלוננת","מדווחת","אומרת","טוענת","מציינת","מתארת","מספרת","מרגישה","מוסרת","חווה","סובלת",
    "בוקר","בחילות","ללא","אין","כן","לא"
}
STOPWORDS = list(HEB_STOP | EXTRA_STOP)

# ===== Text utils =====
def normalize_he(s: str) -> str:
    """Basic whitespace cleanup."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.replace("\r", "\n")).strip()

def candidate_headers(text):
    """Return list of header-like lines found in the text."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [m.group(1).strip() for m in HEADER_LINE_RE.finditer(text)]

def canon_of(header: str) -> str | None:
    """Map header line to one of the 7 canon families; None if not matched."""
    h = (header or "").strip()
    for c, regs in TOP7_RE.items():
        if any(r.search(h) for r in regs):
            return c
    return None

def parse_canon_sections_all(text: str):
    """
    Return ALL matched sections (no cap):
      [{'raw_header','canon','week','text','ord'}...]
    """
    if not isinstance(text, str) or not text.strip():
        return []
    secs = [(m.group(1).strip(), m.group(2).strip(), m.start())
            for m in SECTION_RE.finditer(text)]
    weeks = [(m.start(), int(m.group(1))) for m in WEEK_RE.finditer(text)]
    out = []
    for i, (hdr, body, pos) in enumerate(secs):
        cn = canon_of(hdr)
        if not cn:
            continue
        # nearest week BEFORE header (1..15)
        wk = None
        for p, w in weeks:
            if p <= pos and 1 <= w <= 15:
                wk = w
            elif p > pos:
                break
        # fallback: week inside body text (e.g., "בשבוע 8 ...")
        if wk is None:
            m = INLINE_WEEK_RE.search(body)
            if m:
                w2 = int(m.group(1))
                if 1 <= w2 <= 15:
                    wk = w2
        out.append({"raw_header": hdr, "canon": cn, "week": wk, "text": body, "ord": i})
    out.sort(key=lambda d: d["ord"])
    return out

def split_to_sentences(sections: pd.DataFrame) -> pd.DataFrame:
    """Explode 'text' into sentences; keep row_id/canon/week/sentence."""
    d = sections[["row_id","canon","week","text"]].copy()
    d["sentence"] = d["text"].astype(str).apply(
        lambda t: [s.strip() for s in SENT_SPLIT_RE.split(t) if s and s.strip()]
    )
    d = d.explode("sentence").dropna(subset=["sentence"]).reset_index(drop=True)
    d = d[d["sentence"].str.len() > 0]
    return d[["row_id","canon","week","sentence"]]

def attach_target(sent_df: pd.DataFrame, df_full: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """Merge TARGET by row_id (assumes row_id == positional index of df_full)."""
    y_map = pd.DataFrame({"row_id": np.arange(len(df_full)),
                          target_col: df_full[target_col].astype(int).values})
    return sent_df.merge(y_map, on="row_id", how="left")

def _post_filter(df_top: pd.DataFrame, keep_bigrams_only: bool = False) -> pd.DataFrame:
    """Drop numeric-only and single-char; optionally keep only bigrams."""
    x = df_top.copy()
    x = x[~x["ngram"].str.fullmatch(r"\d+")]        # remove digits-only
    x = x[x["ngram"].str.len() > 1]                 # remove very short
    if keep_bigrams_only:
        x = x[x["ngram"].str.contains(r"\s")]       # keep bigrams+
    return x

def top_ngrams_per_section(
    df_pos: pd.DataFrame,
    ngram_range=(1,2),
    min_df=3,
    top_k=15,
    keep_bigrams_only=False
) -> dict:
    """Return dict: canon -> DataFrame(['ngram','count']) for positives only."""
    out = {}
    for canon, sub in df_pos.groupby("canon", sort=False):
        texts = sub["sentence"].dropna().tolist()
        if not texts:
            continue
        vec = CountVectorizer(
            preprocessor=normalize_he,
            lowercase=False,
            stop_words=STOPWORDS,
            ngram_range=ngram_range,
            min_df=1 if len(texts) < min_df else min_df,
            max_df=0.95,
        )
        try:
            X = vec.fit_transform(texts)
        except ValueError:
            continue  # empty vocab
        vocab  = np.array(vec.get_feature_names_out())
        counts = np.asarray(X.sum(axis=0)).ravel()
        df_top = pd.DataFrame({"ngram": vocab, "count": counts}).sort_values("count", ascending=False)
        out[canon] = _post_filter(df_top, keep_bigrams_only=keep_bigrams_only).head(top_k).reset_index(drop=True)
    return out

def plot_top_ngrams_per_section(results: dict, k: int = 15) -> None:
    """One horizontal bar chart per section."""
    for canon, df_top in results.items():
        if df_top.empty:
            continue
        data = df_top.head(k).iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(data["ngram"], data["count"])
        plt.title(f"Top {len(data)} n-grams (positives) — {canon}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.show()

def drop_leakage(df: pd.DataFrame, target: str, text_col: str | None) -> pd.DataFrame:
    """
    Keeps target and (optionally) the text column; drops columns whose *names* match LEAK_PATTERNS.
    Expects a compiled regex named LEAK_PATTERNS to exist in the caller scope.
    """
    keep = [target]
    for c in df.columns:
        if c == target:
            continue
        if text_col and c == text_col:
            keep.append(c); continue
        # NOTE: LEAK_PATTERNS must be defined by caller
        if 'LEAK_PATTERNS' in globals() and LEAK_PATTERNS.search(c):
            continue
        keep.append(c)
    return df[keep].copy()

def metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}

def recall_ppv_threshold_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    n = len(y_score)
    if k <= 0:
        return 0, 0.0, 0.0, 1.0
    k = min(k, n)
    order = np.argsort(-y_score)
    top = order[:k]
    thr = float(y_score[order[k-1]])
    tp = int(y_true[top].sum()); pos = int(y_true.sum())
    recall = float(tp/pos) if pos>0 else 0.0
    ppv = float(tp/k)
    return k, recall, ppv, thr


