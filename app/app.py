from pathlib import Path
import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Lab de Predicción",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_CONFIG = {
    "Dólar": {
        "badge": "FX",
        "title": "Predicción del dólar",
        "subtitle": (
            "Simula escenarios macroeconómicos con un panel más claro, visual y "
            "útil para lectura rápida de resultados."
        ),
        "summary": "Proyección sensible a inflación y tasa de interés.",
        "insight": "Ideal para contrastar presiones cambiarias en escenarios cortos.",
        "model_file": "modelo_dolar.pkl",
        "data_file": "dolar_data.csv",
        "target_column": "Precio_Dolar",
        "target_name": "Precio del dólar",
        "unit": "COP",
        "prefix": "$ ",
        "decimals": 2,
        "accent": "#0F766E",
        "accent_strong": "#115E59",
        "accent_soft": "rgba(15, 118, 110, 0.18)",
        "button_label": "Calcular proyección cambiaria",
        "features": [
            {
                "column": "Dia",
                "label": "Día",
                "kind": "int",
                "decimals": 0,
                "step": 1,
                "min": 1,
                "max": 365,
                "format": "%d",
                "help": "Índice temporal del escenario que quieres evaluar.",
            },
            {
                "column": "Inflacion",
                "label": "Inflación",
                "kind": "float",
                "decimals": 3,
                "step": 0.001,
                "min": 0.0,
                "max": 0.2,
                "format": "%.3f",
                "help": "Valor decimal. Ejemplo: 0.025 equivale a 2.5%.",
            },
            {
                "column": "Tasa_interes",
                "label": "Tasa de interés",
                "kind": "float",
                "decimals": 2,
                "step": 0.1,
                "min": 0.0,
                "max": 30.0,
                "format": "%.2f",
                "help": "Tasa de referencia del escenario macroeconómico.",
            },
        ],
    },
    "Glucosa": {
        "badge": "BIO",
        "title": "Predicción de glucosa",
        "subtitle": (
            "Evalúa el nivel esperado de glucosa con una interfaz más profesional, "
            "ordenada y fácil de interpretar."
        ),
        "summary": "Modelo biométrico basado en edad, IMC y actividad física.",
        "insight": "Útil para revisar rápidamente combinaciones de perfil y hábitos.",
        "model_file": "modelo_glucosa.pkl",
        "data_file": "glucosa_data.csv",
        "target_column": "Nivel_Glucosa",
        "target_name": "Nivel de glucosa",
        "unit": "mg/dL",
        "prefix": "",
        "decimals": 2,
        "accent": "#C2410C",
        "accent_strong": "#9A3412",
        "accent_soft": "rgba(194, 65, 12, 0.16)",
        "button_label": "Calcular estimación glucémica",
        "features": [
            {
                "column": "Edad",
                "label": "Edad",
                "kind": "int",
                "decimals": 0,
                "step": 1,
                "min": 1,
                "max": 100,
                "format": "%d",
                "help": "Edad de la persona evaluada.",
            },
            {
                "column": "IMC",
                "label": "IMC",
                "kind": "float",
                "decimals": 2,
                "step": 0.1,
                "min": 10.0,
                "max": 60.0,
                "format": "%.2f",
                "help": "Índice de masa corporal del escenario.",
            },
            {
                "column": "Actividad_Fisica",
                "label": "Actividad física (horas)",
                "kind": "float",
                "decimals": 1,
                "step": 0.5,
                "min": 0.0,
                "max": 24.0,
                "format": "%.1f",
                "help": "Cantidad de horas de actividad física.",
            },
        ],
    },
    "Energía": {
        "badge": "GRID",
        "title": "Predicción de energía",
        "subtitle": (
            "Visualiza consumo esperado con una experiencia más pulida para "
            "explorar temperatura, hora y día de la semana."
        ),
        "summary": "Modelo operativo enfocado en patrones de consumo energético.",
        "insight": "Permite comparar escenarios horarios y climáticos con rapidez.",
        "model_file": "modelo_energia.pkl",
        "data_file": "energia_data.csv",
        "target_column": "Consumo_Energia",
        "target_name": "Consumo de energía",
        "unit": "kWh",
        "prefix": "",
        "decimals": 2,
        "accent": "#1D4ED8",
        "accent_strong": "#1E40AF",
        "accent_soft": "rgba(29, 78, 216, 0.16)",
        "button_label": "Calcular consumo proyectado",
        "features": [
            {
                "column": "Temperatura",
                "label": "Temperatura (°C)",
                "kind": "float",
                "decimals": 1,
                "step": 0.5,
                "min": -10.0,
                "max": 50.0,
                "format": "%.1f",
                "help": "Temperatura ambiente del escenario.",
            },
            {
                "column": "Hora",
                "label": "Hora",
                "kind": "int",
                "decimals": 0,
                "step": 1,
                "min": 1,
                "max": 24,
                "format": "%d",
                "help": "Hora del día en formato 1 a 24.",
            },
            {
                "column": "Dia_Semana",
                "label": "Día de la semana",
                "kind": "int",
                "decimals": 0,
                "step": 1,
                "min": 1,
                "max": 7,
                "format": "%d",
                "help": "1 representa el primer día de la semana del dataset.",
            },
        ],
    },
}


def feature_columns(cfg: dict) -> list[str]:
    return [field["column"] for field in cfg["features"]]


@st.cache_resource
def load_model(model_file: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(MODELS_DIR / model_file)


@st.cache_resource
def rebuild_model(data_file: str, target_column: str, columns: tuple[str, ...]):
    df = load_dataset(data_file)
    model = LinearRegression()
    model.fit(df[list(columns)], df[target_column])
    return model


@st.cache_data
def load_dataset(data_file: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / data_file)


def validate_model(model, cfg: dict, df: pd.DataFrame) -> str | None:
    expected_columns = feature_columns(cfg)
    actual_columns = list(getattr(model, "feature_names_in_", []))

    if actual_columns and actual_columns != expected_columns:
        expected = ", ".join(expected_columns)
        actual = ", ".join(actual_columns)
        return (
            f"El archivo {cfg['model_file']} no coincide con {cfg['title']}. "
            f"Variables esperadas: {expected}. Variables del artefacto: {actual}."
        )

    try:
        model.predict(df[expected_columns].head(1))
    except Exception as exc:
        return f"No se pudo validar {cfg['model_file']}: {exc}"

    return None


def resolve_model(cfg: dict, df: pd.DataFrame):
    fallback_reason = None

    try:
        model = load_model(cfg["model_file"])
    except Exception as exc:
        fallback_reason = f"No se pudo cargar {cfg['model_file']}: {exc}"
    else:
        validation_error = validate_model(model, cfg, df)
        if validation_error is None:
            return model, None
        fallback_reason = validation_error

    fallback_model = rebuild_model(
        cfg["data_file"],
        cfg["target_column"],
        tuple(feature_columns(cfg)),
    )
    return fallback_model, fallback_reason


def format_number(value: float, decimals: int = 2) -> str:
    pattern = f"{{:,.{decimals}f}}"
    text = pattern.format(value)
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def format_value(value: float, cfg: dict, decimals: int | None = None) -> str:
    digits = cfg["decimals"] if decimals is None else decimals
    return f"{cfg['prefix']}{format_number(value, digits)} {cfg['unit']}".strip()


def describe_band(value: float, stats: dict) -> tuple[str, str]:
    if value <= stats["q25"]:
        return "Escenario bajo", "El resultado cae por debajo del tramo central histórico."
    if value >= stats["q75"]:
        return "Escenario alto", "El resultado supera el tramo central histórico."
    return "Escenario estable", "El resultado se mantiene dentro del rango intercuartil del dataset."


def dataset_stats(df: pd.DataFrame, cfg: dict) -> dict:
    target = df[cfg["target_column"]]
    return {
        "records": int(len(df)),
        "mean": float(target.mean()),
        "min": float(target.min()),
        "max": float(target.max()),
        "q25": float(target.quantile(0.25)),
        "q75": float(target.quantile(0.75)),
        "window": max(3, min(14, len(df) // 10 if len(df) > 20 else 4)),
    }


def field_default_value(df: pd.DataFrame, field: dict):
    value = float(df[field["column"]].median())
    if field["kind"] == "int":
        return int(round(value))
    return round(value, field["decimals"])


def field_range_caption(df: pd.DataFrame, field: dict) -> str:
    series = df[field["column"]]
    low = format_number(float(series.min()), field["decimals"])
    high = format_number(float(series.max()), field["decimals"])
    return f"Rango observado: {low} a {high}"


def inject_css(cfg: dict) -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');

        :root {{
            --bg-a: #f8fafc;
            --bg-b: #e9f1fb;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-strong: #ffffff;
            --stroke: rgba(15, 23, 42, 0.08);
            --text: #0f172a;
            --muted: #475569;
            --accent: {cfg["accent"]};
            --accent-strong: {cfg["accent_strong"]};
            --accent-soft: {cfg["accent_soft"]};
        }}

        html, body, [class*="css"]  {{
            font-family: 'Manrope', sans-serif;
        }}

        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 0% 0%, var(--accent-soft) 0%, rgba(255,255,255,0) 32%),
                radial-gradient(circle at 100% 0%, rgba(15,23,42,0.06) 0%, rgba(255,255,255,0) 28%),
                linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 100%);
        }}

        [data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0);
        }}

        .block-container {{
            max-width: 1260px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(15, 23, 42, 0.92) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }}

        [data-testid="stSidebar"] * {{
            color: #e2e8f0;
        }}

        h1, h2, h3 {{
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.03em;
            color: var(--text);
        }}

        .hero-card {{
            position: relative;
            overflow: hidden;
            padding: 2.2rem;
            border-radius: 28px;
            background: linear-gradient(145deg, rgba(255,255,255,0.94), rgba(255,255,255,0.74));
            border: 1px solid var(--stroke);
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.09);
        }}

        .hero-card::after {{
            content: "";
            position: absolute;
            right: -70px;
            bottom: -90px;
            width: 240px;
            height: 240px;
            border-radius: 999px;
            background: radial-gradient(circle, var(--accent-soft) 0%, rgba(255,255,255,0) 68%);
        }}

        .eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }}

        .hero-title {{
            margin: 1rem 0 0.5rem 0;
            font-size: clamp(2.4rem, 4vw, 4rem);
            line-height: 1.02;
            max-width: 760px;
        }}

        .hero-subtitle {{
            max-width: 780px;
            margin: 0;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}

        .hero-pill {{
            padding: 1rem 1.1rem;
            border-radius: 20px;
            border: 1px solid var(--stroke);
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(10px);
        }}

        .hero-pill span,
        .stat-card span,
        .result-chip span,
        .result-meta-item span,
        .guide-card span,
        .sidebar-card span,
        .sidebar-brand span {{
            display: block;
            margin-bottom: 0.35rem;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}

        .hero-pill span {{
            color: var(--muted);
        }}

        .hero-pill strong {{
            font-size: 1rem;
            color: var(--text);
        }}

        .section-title {{
            margin: 1.2rem 0 0.2rem 0;
            font-size: 1.2rem;
        }}

        .section-copy {{
            margin: 0 0 1rem 0;
            color: var(--muted);
            line-height: 1.65;
        }}

        .stat-card {{
            height: 100%;
            min-height: 160px;
            padding: 1.25rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.72));
            border: 1px solid var(--stroke);
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
        }}

        .stat-card span {{
            color: var(--muted);
        }}

        .stat-value {{
            margin: 0.5rem 0 0.65rem 0;
            font-family: 'Sora', sans-serif;
            font-size: 1.75rem;
            line-height: 1.15;
            color: var(--text);
        }}

        .stat-card p {{
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
        }}

        [data-testid="stRadio"] {{
            margin-top: 0.4rem;
        }}

        [data-testid="stRadio"] > div {{
            background: rgba(255,255,255,0.68);
            padding: 0.45rem;
            border-radius: 999px;
            border: 1px solid var(--stroke);
        }}

        div[role="radiogroup"] {{
            gap: 0.55rem;
        }}

        div[role="radiogroup"] label {{
            border-radius: 999px;
            padding: 0.25rem 0.45rem;
            transition: all 0.2s ease;
        }}

        div[role="radiogroup"] label:hover {{
            background: rgba(255,255,255,0.7);
        }}

        div[role="radiogroup"] label:has(input:checked) {{
            background: #ffffff;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        }}

        div[role="radiogroup"] p {{
            font-weight: 700;
            color: var(--text);
        }}

        [data-testid="stForm"] {{
            padding: 1.2rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.74));
            border: 1px solid var(--stroke);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        }}

        div[data-baseweb="input"] {{
            border-radius: 18px;
            border: 1px solid rgba(15, 23, 42, 0.09);
            background: rgba(255,255,255,0.94);
            min-height: 54px;
        }}

        div[data-baseweb="input"]:focus-within {{
            border-color: var(--accent);
            box-shadow: 0 0 0 4px var(--accent-soft);
        }}

        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {{
            width: 100%;
            min-height: 54px;
            border: 0;
            border-radius: 18px;
            background: linear-gradient(135deg, var(--accent), var(--accent-strong));
            color: #ffffff;
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 18px 34px rgba(15, 23, 42, 0.14);
        }}

        .result-card {{
            padding: 1.5rem;
            border-radius: 28px;
            background: linear-gradient(145deg, rgba(15,23,42,0.98), rgba(30,41,59,0.95));
            color: #f8fafc;
            box-shadow: 0 28px 60px rgba(15, 23, 42, 0.22);
        }}

        .result-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: #cbd5e1;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}

        .result-card h3 {{
            margin: 1rem 0 0.45rem 0;
            color: #ffffff;
            font-size: 1.2rem;
        }}

        .result-value {{
            font-family: 'Sora', sans-serif;
            font-size: clamp(2rem, 3.2vw, 3.2rem);
            line-height: 1.08;
            color: #ffffff;
        }}

        .result-body {{
            margin: 0.7rem 0 0 0;
            color: #cbd5e1;
            line-height: 1.7;
        }}

        .result-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1.2rem;
        }}

        .result-chip {{
            padding: 0.9rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.07);
        }}

        .result-chip span {{
            color: #94a3b8;
        }}

        .result-chip strong {{
            font-size: 1rem;
            color: #ffffff;
        }}

        .result-meta {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }}

        .result-meta-item {{
            padding: 0.9rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
        }}

        .result-meta-item span {{
            color: #94a3b8;
        }}

        .result-meta-item strong {{
            color: #ffffff;
            font-size: 1rem;
        }}

        .placeholder-card,
        .guide-card,
        .sidebar-card,
        .sidebar-brand {{
            border-radius: 24px;
            border: 1px solid var(--stroke);
        }}

        .placeholder-card {{
            padding: 1.45rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.7));
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        }}

        .placeholder-card h3,
        .guide-card h3 {{
            margin: 0.5rem 0;
            font-size: 1.1rem;
        }}

        .placeholder-card p,
        .guide-card p {{
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
        }}

        .guide-card {{
            height: 100%;
            padding: 1.2rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.74));
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.05);
        }}

        .guide-card span {{
            color: var(--accent);
        }}

        .guide-card strong {{
            display: block;
            margin-bottom: 0.45rem;
            font-size: 1.1rem;
            color: var(--text);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.55rem;
            padding: 0.35rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.66);
            border: 1px solid var(--stroke);
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 46px;
            border-radius: 999px;
            padding: 0 1rem;
        }}

        .stTabs [aria-selected="true"] {{
            background: #ffffff !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }}

        div[data-testid="stMetric"] {{
            padding: 1rem;
            border-radius: 20px;
            border: 1px solid var(--stroke);
            background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.74));
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.05);
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid var(--stroke);
            background: rgba(255,255,255,0.9);
        }}

        .sidebar-brand,
        .sidebar-card {{
            padding: 1rem;
            background: rgba(255,255,255,0.05);
            border-color: rgba(255,255,255,0.08);
            margin-bottom: 0.9rem;
        }}

        .sidebar-brand strong,
        .sidebar-card strong {{
            display: block;
            color: #f8fafc;
        }}

        .sidebar-brand p,
        .sidebar-card p {{
            margin: 0.2rem 0 0 0;
            color: #cbd5e1;
            line-height: 1.55;
            font-size: 0.92rem;
        }}

        .sidebar-brand span,
        .sidebar-card span {{
            color: #94a3b8;
        }}

        .footer-note {{
            margin-top: 1.5rem;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
            text-align: center;
        }}

        @media (max-width: 980px) {{
            .hero-grid,
            .result-grid,
            .result-meta {{
                grid-template-columns: 1fr;
            }}

            .block-container {{
                padding-top: 1.2rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(title: str, value: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <span>{title}</span>
            <div class="stat-value">{value}</div>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(prediction: float, inputs: dict, cfg: dict, stats: dict) -> None:
    band_title, band_copy = describe_band(prediction, stats)
    mean_delta = ((prediction - stats["mean"]) / stats["mean"] * 100) if stats["mean"] else 0.0
    delta_copy = (
        f"{format_number(abs(mean_delta), 1)}% por encima del promedio"
        if mean_delta >= 0
        else f"{format_number(abs(mean_delta), 1)}% por debajo del promedio"
    )

    input_cards = []
    for field in cfg["features"]:
        raw_value = float(inputs[field["column"]])
        formatted = format_number(raw_value, field["decimals"])
        input_cards.append(
            f'<div class="result-chip"><span>{field["label"]}</span><strong>{formatted}</strong></div>'
        )

    meta_cards = [
        (
            "Promedio histórico",
            format_value(stats["mean"], cfg),
        ),
        (
            "Rango observado",
            f"{format_value(stats['min'], cfg)} - {format_value(stats['max'], cfg)}",
        ),
        (
            "Comparación",
            delta_copy,
        ),
    ]

    input_cards_html = "".join(input_cards)
    meta_html = "".join(
        f'<div class="result-meta-item"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in meta_cards
    )
    result_html = f"""
    <div class="result-card">
        <div class="result-kicker">{cfg["badge"]} | {band_title}</div>
        <h3>{cfg["target_name"]}</h3>
        <div class="result-value">{format_value(prediction, cfg)}</div>
        <p class="result-body">{band_copy} En esta simulación el valor queda {delta_copy.lower()}.</p>
        <div class="result-grid">{input_cards_html}</div>
        <div class="result-meta">{meta_html}</div>
    </div>
    """

    st.markdown(
        result_html,
        unsafe_allow_html=True,
    )


model_names = list(MODEL_CONFIG)
current_model = st.session_state.get("active_model", model_names[0])
inject_css(MODEL_CONFIG.get(current_model, MODEL_CONFIG[model_names[0]]))

st.markdown('<h3 class="section-title">Selecciona el modelo</h3>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Alterna entre escenarios financieros, biométricos y operativos sin cambiar de pantalla.</p>',
    unsafe_allow_html=True,
)

selected_model = st.radio(
    "Modelo",
    model_names,
    key="active_model",
    horizontal=True,
    label_visibility="collapsed",
)
cfg = MODEL_CONFIG[selected_model]
data = load_dataset(cfg["data_file"])
model, model_notice = resolve_model(cfg, data)
stats = dataset_stats(data, cfg)

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <span>Lab de Predicción</span>
            <strong>{cfg["title"]}</strong>
            <p>{cfg["summary"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <span>Modelo activo</span>
            <strong>{cfg["badge"]} | Regresión lineal</strong>
            <p>{cfg["insight"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-card">
            <span>Cómo usar</span>
            <strong>1. Ajusta variables</strong>
            <p>2. Ejecuta la predicción. 3. Compara el resultado contra el histórico del dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <span>Datos</span>
            <strong>{stats["records"]} registros locales</strong>
            <p>Los modelos y datasets se cargan desde las carpetas <code>models</code> y <code>data</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="hero-card">
        <div class="eyebrow">Laboratorio de minería de datos</div>
        <h1 class="hero-title">{cfg["title"]}</h1>
        <p class="hero-subtitle">{cfg["subtitle"]}</p>
        <div class="hero-grid">
            <div class="hero-pill">
                <span>Tipo de modelo</span>
                <strong>{cfg["badge"]} | Regresión lineal</strong>
            </div>
            <div class="hero-pill">
                <span>Unidad objetivo</span>
                <strong>{cfg["unit"]}</strong>
            </div>
            <div class="hero-pill">
                <span>Tamaño del dataset</span>
                <strong>{stats["records"]} observaciones</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h3 class="section-title">Resumen del modelo</h3>', unsafe_allow_html=True)
st.markdown(
    f'<p class="section-copy">{cfg["summary"]}</p>',
    unsafe_allow_html=True,
)

if model_notice:
    st.warning(
        "Se detectó un modelo incompatible con este escenario. "
        "La app reconstruyó un modelo lineal con el dataset local para mantener la predicción operativa."
    )

stat_cols = st.columns(3)
with stat_cols[0]:
    render_stat_card(
        "Promedio histórico",
        format_value(stats["mean"], cfg),
        "Sirve como referencia base para comparar la simulación actual.",
    )
with stat_cols[1]:
    render_stat_card(
        "Zona alta",
        format_value(stats["q75"], cfg),
        "Valores por encima de este punto ya entran en el tramo alto del dataset.",
    )
with stat_cols[2]:
    render_stat_card(
        "Rango observado",
        f"{format_value(stats['min'], cfg)} - {format_value(stats['max'], cfg)}",
        cfg["insight"],
    )

left_col, right_col = st.columns([1.08, 0.92], gap="large")

with left_col:
    st.markdown('<h3 class="section-title">Configura tu escenario</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Ajusta las variables y ejecuta una predicción con una lectura clara del resultado.</p>',
        unsafe_allow_html=True,
    )

    with st.form(f"form_{selected_model}"):
        inputs = {}
        input_cols = st.columns(2)

        for index, field in enumerate(cfg["features"]):
            widget_col = input_cols[index % 2]
            with widget_col:
                default_value = field_default_value(data, field)
                if field["kind"] == "int":
                    value = st.number_input(
                        field["label"],
                        min_value=int(field["min"]),
                        max_value=int(field["max"]),
                        value=int(default_value),
                        step=int(field["step"]),
                        format=field["format"],
                        help=field["help"],
                        key=f"{selected_model}_{field['column']}",
                    )
                else:
                    value = st.number_input(
                        field["label"],
                        min_value=float(field["min"]),
                        max_value=float(field["max"]),
                        value=float(default_value),
                        step=float(field["step"]),
                        format=field["format"],
                        help=field["help"],
                        key=f"{selected_model}_{field['column']}",
                    )
                inputs[field["column"]] = value
                st.caption(field_range_caption(data, field))

        submitted = st.form_submit_button(cfg["button_label"], width="stretch")

    if submitted:
        input_frame = pd.DataFrame(
            [{field["column"]: float(inputs[field["column"]]) for field in cfg["features"]}]
        )
        prediction = float(model.predict(input_frame[feature_columns(cfg)])[0])
        st.session_state[f"prediction_{selected_model}"] = {
            "value": prediction,
            "inputs": inputs,
        }

with right_col:
    st.markdown('<h3 class="section-title">Resultado de la simulación</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Aquí aparece la predicción con contexto histórico y lectura inmediata.</p>',
        unsafe_allow_html=True,
    )

    result_state = st.session_state.get(f"prediction_{selected_model}")
    if result_state:
        render_result(result_state["value"], result_state["inputs"], cfg, stats)
    else:
        st.markdown(
            f"""
            <div class="placeholder-card">
                <span class="eyebrow">Vista previa</span>
                <h3>Ejecuta una simulación para ver el resultado</h3>
                <p>El panel mostrará el valor estimado, su posición frente al histórico y un resumen de las variables que usaste.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Promedio histórico", format_value(stats["mean"], cfg))
    with metric_cols[1]:
        st.metric("Cuartil inferior", format_value(stats["q25"], cfg))
    with metric_cols[2]:
        st.metric("Cuartil superior", format_value(stats["q75"], cfg))

tabs = st.tabs(["Señal histórica", "Muestra de datos", "Variables"])

with tabs[0]:
    st.markdown('<h3 class="section-title">Comportamiento del objetivo</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">La gráfica combina la serie original con una media móvil para facilitar la lectura.</p>',
        unsafe_allow_html=True,
    )
    chart_df = pd.DataFrame(
        {
            cfg["target_name"]: data[cfg["target_column"]],
            "Media móvil": data[cfg["target_column"]].rolling(stats["window"], min_periods=1).mean(),
        }
    )
    st.line_chart(chart_df, width="stretch")
    if st.session_state.get(f"prediction_{selected_model}"):
        st.caption(
            f"Última simulación: {format_value(st.session_state[f'prediction_{selected_model}']['value'], cfg)}"
        )

with tabs[1]:
    st.markdown('<h3 class="section-title">Vista rápida del dataset</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Una muestra inicial de los datos utilizados por el modelo para mantener contexto numérico.</p>',
        unsafe_allow_html=True,
    )
    rename_map = {cfg["target_column"]: cfg["target_name"]}
    for field in cfg["features"]:
        rename_map[field["column"]] = field["label"]

    formatted_preview = data.head(10).rename(columns=rename_map)
    column_config = {
        rename_map[field["column"]]: st.column_config.NumberColumn(
            format=f"%.{field['decimals']}f" if field["decimals"] else "%d"
        )
        for field in cfg["features"]
    }
    column_config[cfg["target_name"]] = st.column_config.NumberColumn(
        format=f"%.{cfg['decimals']}f"
    )

    st.dataframe(
        formatted_preview,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )

with tabs[2]:
    st.markdown('<h3 class="section-title">Guía de variables</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Cada variable incorpora una referencia rápida para facilitar simulaciones más coherentes.</p>',
        unsafe_allow_html=True,
    )
    guide_cols = st.columns(len(cfg["features"]))
    for guide_col, field in zip(guide_cols, cfg["features"]):
        with guide_col:
            median_value = format_number(float(data[field["column"]].median()), field["decimals"])
            st.markdown(
                f"""
                <div class="guide-card">
                    <span>{field["label"]}</span>
                    <strong>Mediana: {median_value}</strong>
                    <p>{field["help"]}</p>
                    <p>{field_range_caption(data, field)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown(
    """
    <div class="footer-note">
        Interfaz rediseñada para mejorar lectura, jerarquía visual y experiencia de simulación sin alterar la lógica principal de los modelos.
    </div>
    """,
    unsafe_allow_html=True,
)
