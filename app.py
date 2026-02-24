import streamlit as st
import pandas as pd
import requests
from datetime import date
import io
import uuid
import json
import hashlib
import altair as alt
from typing import List, Tuple, Optional
from datetime import date as _date
from datetime import datetime

# ---------- PDF (reportlab) ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------- REQUESTS: Session + retries ----------
def build_session():
    s = requests.Session()
    try:
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter

        retry = Retry(
            total=3,
            backoff_factor=0.4,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PATCH", "DELETE"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    except Exception:
        pass
    return s


SESSION = build_session()
TIMEOUT = 15

# ---------- CONFIG SUPABASE ----------
SUPABASE_URL = st.secrets["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

BASE_URL = f"{SUPABASE_URL}/rest/v1"
TABLE_MOV = "movimientos"
TABLE_SALDOS = "saldos_iniciales"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

HEADERS_UPSERT = {
    **HEADERS,
    "Prefer": "return=representation,resolution=merge-duplicates",
}

# ---------- LISTAS FIJAS ----------
CUENTAS = [
    "Principal",
    "Extra",
    "Corto plazo",
    "Medio plazo",
    "Largo plazo",
    "Efectivo",
]

CATS_GASTOS = [
    "Cesta", "Luz", "Agua", "Vivienda", "Gas", "Internet", "Colegio",
    "Suscripciones", "Mantenimiento hogar", "Impuestos", "Seguros",
    "Letra Coche", "Cuidados", "Deporte", "Prestamos", "Comunidad",
    "Combustible", "Adquisiciones", "Restauración", "Viajes", "Eventos",
    "Contabilidad", "Formaciones", "Empresa", "Laborales", "Otros",
]

CATS_INGRESOS = [
    "Nomina Vic",
    "Nomina Sof",
    "Empresa",
    "Inversiones",
    "Ayuda",
    "Donativos",
]

# 👇 columnas esperadas en DB (si no existen, se crean en df local con None)
DB_COLUMNS = ["fecha", "descripcion", "categoria", "cuenta", "cuenta_destino", "importe", "tipo", "mov_hash"]

MESES_NOMBRES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}
DATE_FORMAT = "DD/MM/YYYY"


def nombre_mes(m):
    return MESES_NOMBRES.get(m, str(m))


# ---------- UTIL ----------
def today_ddmmyyyy() -> str:
    return _date.today().strftime("%d/%m/%Y")

def parse_fecha_flexible(v: object) -> Optional[str]:
    """
    Acepta:
    - "dd/mm/yyyy"
    - "yyyy-mm-dd"
    - "d" o "dd" => dia del mes actual/año actual
    - "d/m" o "dd/mm" => año actual
    Devuelve ISO "yyyy-mm-dd" o None si invalida.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # yyyy-mm-dd
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().isoformat()
    except Exception:
        pass

    # dd/mm/yyyy
    try:
        return datetime.strptime(s, "%d/%m/%Y").date().isoformat()
    except Exception:
        pass

    # d o dd => mes/año actual
    if s.isdigit():
        d = int(s)
        try:
            return _date(_date.today().year, _date.today().month, d).isoformat()
        except Exception:
            return None

    # d/m o dd/mm => año actual
    if "/" in s:
        parts = [p for p in s.split("/") if p.strip() != ""]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            d = int(parts[0])
            m = int(parts[1])
            y = _date.today().year
            try:
                return _date(y, m, d).isoformat()
            except Exception:
                return None

    return None
def parse_importe(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(" ", "")
        if s == "":
            return None
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s and "." not in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None
    return None


def normalizar_fecha(valor):
    from datetime import date as _date
    import pandas as _pd
    if isinstance(valor, _pd.Timestamp):
        return valor.date().isoformat()
    if isinstance(valor, _date):
        return valor.isoformat()
    return valor


def safe_str(x) -> str:
    return "" if x is None else str(x)


def sanitize_choice(value: str, options: List[str]) -> Optional[str]:
    v = safe_str(value).strip()
    if not v:
        return None
    return v if v in options else None


def short_json(obj, max_chars=2000):
    s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if len(s) > max_chars:
        return s[:max_chars] + "\n... (truncado)"
    return s


def normalize_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("", "none", "nan", "nat"):
        return ""
    return s


def normalize_tipo(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("", "none", "nan", "nat"):
        return None
    if s in ("gasto", "ingreso", "transferencia"):
        return s
    return None


def money_key(imp: float) -> str:
    try:
        return f"{float(imp):.2f}"
    except Exception:
        return "0.00"


def build_mov_hash(payload: dict) -> str:
    """
    Hash EXACTO del movimiento.
    - Respeta mayúsculas/minúsculas en descripción.
    - Incluye fecha completa (YYYY-MM-DD), tipo, cuenta, destino, categoría, importe.
    """
    fecha = safe_str(payload.get("fecha")).strip()
    tipo = safe_str(payload.get("tipo")).strip().lower()
    cuenta = safe_str(payload.get("cuenta")).strip()
    cuenta_destino = safe_str(payload.get("cuenta_destino")).strip()
    categoria = safe_str(payload.get("categoria")).strip()
    descripcion = safe_str(payload.get("descripcion")).strip()
    importe = money_key(payload.get("importe") or 0.0)

    base = "||".join([fecha, tipo, cuenta, cuenta_destino, categoria, importe, descripcion])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# ---------- Debug / error helper ----------
def show_http_error(action: str, r: requests.Response, sample_payload=None):
    st.error(f"{action} ERROR: {r.status_code}")
    try:
        st.code(r.text[:4000])
    except Exception:
        st.code("<sin cuerpo>")
    if sample_payload is not None:
        st.caption("Payload (muestra):")
        st.code(short_json(sample_payload, 2500))


# ---------- SUPABASE ----------
def fetch_movimientos():
    url = f"{BASE_URL}/{TABLE_MOV}"
    # ✅ Mostramos lo más reciente primero para que lo que insertas aparezca arriba (página 1)
    r = SESSION.get(
        url,
        headers=HEADERS,
        params={"select": "*", "order": "fecha.asc,created_at.asc"},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def fetch_saldos():
    url = f"{BASE_URL}/{TABLE_SALDOS}"
    r = SESSION.get(url, headers=HEADERS, params={"select": "*"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def upsert_movimientos_by_hash(rows: List[dict]):
    """
    UPSERT por mov_hash (requiere UNIQUE en la columna mov_hash en Supabase).
    Si se intenta insertar el mismo movimiento dos veces, NO duplica: mergea.
    """
    if not rows:
        return []

    url = f"{BASE_URL}/{TABLE_MOV}?on_conflict=mov_hash"
    r = SESSION.post(url, headers=HEADERS_UPSERT, json=rows, timeout=TIMEOUT)

    if r.status_code >= 400:
        show_http_error("UPSERT(mov_hash)", r, sample_payload=rows[:3])
        return None

    if not r.text:
        return []
    try:
        return r.json()
    except Exception:
        st.code(r.text[:4000])
        return []


def upsert_movimientos_by_id(rows: List[dict]):
    """
    UPSERT por id (para ediciones/borrados).
    """
    if not rows:
        return []
    url = f"{BASE_URL}/{TABLE_MOV}?on_conflict=id"
    r = SESSION.post(url, headers=HEADERS_UPSERT, json=rows, timeout=TIMEOUT)
    if r.status_code >= 400:
        show_http_error("UPSERT(id)", r, sample_payload=rows[:3])
        return None
    if not r.text:
        return []
    try:
        return r.json()
    except Exception:
        st.code(r.text[:4000])
        return []


def delete_movimientos_bulk(ids):
    ids = [str(i) for i in ids if i]
    if not ids:
        return True
    url = f"{BASE_URL}/{TABLE_MOV}"
    params = {"id": f"in.({','.join(ids)})"}
    r = SESSION.delete(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    if r.status_code >= 400:
        show_http_error("DELETE", r, sample_payload={"ids": ids[:25]})
        return False
    return True


def update_saldo_inicial_upsert(cuenta, saldo):
    url = f"{BASE_URL}/{TABLE_SALDOS}"
    r0 = SESSION.get(url, headers=HEADERS, params={"select": "*", "cuenta": f"eq.{cuenta}"}, timeout=TIMEOUT)
    r0.raise_for_status()
    existe = len(r0.json()) > 0

    if existe:
        r = SESSION.patch(
            url, headers=HEADERS, params={"cuenta": f"eq.{cuenta}"},
            json={"saldo_inicial": float(saldo)}, timeout=TIMEOUT
        )
    else:
        r = SESSION.post(
            url, headers=HEADERS,
            json={"cuenta": cuenta, "saldo_inicial": float(saldo)}, timeout=TIMEOUT
        )

    if r.status_code >= 400:
        show_http_error("SALDOS", r, sample_payload={"cuenta": cuenta, "saldo": saldo})
        r.raise_for_status()


# ---------- DATA ----------
def preparar_dataframe_base(rows):
    df = pd.DataFrame(rows) if rows else pd.DataFrame([])

    # Asegurar columnas esperadas
    for col in ["id", "created_at"] + DB_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Timestamps
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Fecha principal
    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["fecha"] = df["fecha_dt"].dt.date
    df["anio"] = df["fecha_dt"].dt.year
    df["mes"] = df["fecha_dt"].dt.month

    # Normalizaciones básicas
    df["descripcion"] = df["descripcion"].fillna("").astype(str)
    df["categoria"] = df["categoria"].fillna("").astype(str)
    df["cuenta"] = df["cuenta"].fillna("").astype(str)
    df["cuenta_destino"] = df["cuenta_destino"].where(df["cuenta_destino"].notna(), None)

    # Importe a float
    def _to_float(x):
        if x is None:
            return float("nan")
        try:
            return float(x)
        except Exception:
            v = parse_importe(str(x))
            return float(v) if v is not None else float("nan")

    df["importe"] = df["importe"].apply(_to_float)

    # tipo normalizado (fuente de verdad)
    df["tipo"] = df["tipo"].apply(normalize_tipo)

    # mov_hash: si no existe en DB (filas legacy) lo calculamos localmente si hay tipo
    def _ensure_hash(row):
        if row.get("mov_hash") not in (None, "", " "):
            return str(row.get("mov_hash"))

        t = row.get("tipo")
        if t not in ("gasto", "ingreso", "transferencia"):
            return None

        payload = {
            "fecha": row.get("fecha_dt").date().isoformat()
                    if pd.notna(row.get("fecha_dt")) else safe_str(row.get("fecha")),
            "tipo": t,
            "cuenta": row.get("cuenta"),
            "cuenta_destino": row.get("cuenta_destino") or "",
            "categoria": row.get("categoria"),
            "importe": float(row.get("importe") or 0.0),
            "descripcion": row.get("descripcion"),
        }
        return build_mov_hash(payload)

    df["mov_hash"] = df.apply(lambda r: _ensure_hash(r), axis=1)

    # ✅ ORDEN ASC (como antes): lo más antiguo arriba
    if "created_at_dt" in df.columns:
        df = df.sort_values(["fecha_dt", "created_at_dt", "id"], ascending=[True, True, True])
    else:
        df = df.sort_values(["fecha_dt", "id"], ascending=[True, True])

    return df


def get_saldos_iniciales_from_rows(rows):
    saldos = {c: 0.0 for c in CUENTAS}
    if not rows:
        return saldos
    df = pd.DataFrame(rows)
    for _, row in df.iterrows():
        c = row.get("cuenta")
        s = float(row.get("saldo_inicial") or 0)
        if c in saldos:
            saldos[c] = s
    return saldos


def calcular_saldos_por_cuenta(df, saldos_iniciales: dict):
    saldos = {k: float(v) for k, v in saldos_iniciales.items()}
    for _, row in df.iterrows():
        imp = float(row.get("importe") or 0)
        origen = row.get("cuenta")
        destino = row.get("cuenta_destino")
        tipo = row.get("tipo")

        if tipo == "ingreso":
            if origen in saldos:
                saldos[origen] += imp
        elif tipo == "gasto":
            if origen in saldos:
                saldos[origen] -= imp
        elif tipo == "transferencia":
            if origen in saldos:
                saldos[origen] -= imp
            if destino in saldos:
                saldos[destino] += imp
        else:
            pass
    return saldos


# ---------- Editor helpers ----------
def aplicar_defaults_df_editor(df_visible, default_cuenta="Principal"):
    df2 = df_visible.copy()
    if "cuenta" not in df2.columns:
        return df2
    for i in range(len(df2)):
        rid = normalize_id(df2.at[i, "id"]) if "id" in df2.columns else ""
        cuenta = df2.at[i, "cuenta"]
        es_nueva = (rid == "")
        cuenta_vacia = (cuenta is None) or (str(cuenta).strip() == "")
        if es_nueva and cuenta_vacia:
            df2.at[i, "cuenta"] = default_cuenta
    return df2


def build_editor_df(df_src: pd.DataFrame, visible_cols: List[str], default_cuenta: str) -> pd.DataFrame:
    dfv = df_src.copy()

    for c in ["id"] + visible_cols:
        if c not in dfv.columns:
            dfv[c] = None

    dfv = dfv[["id"] + visible_cols].copy()
    dfv = dfv.reset_index(drop=True)

    # fecha como texto dd/mm/yyyy (y si viene date, convertir)
    if "fecha" in dfv.columns:
        def _fmt_fecha(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ""
            try:
                if isinstance(x, _date):
                    return x.strftime("%d/%m/%Y")
            except Exception:
                pass
            return str(x)
        dfv["fecha"] = dfv["fecha"].apply(_fmt_fecha)

    if "importe" in dfv.columns:
        dfv["importe"] = dfv["importe"].apply(lambda x: "" if pd.isna(x) else str(x))

    dfv["🗑 Eliminar"] = False
    dfv = aplicar_defaults_df_editor(dfv, default_cuenta=default_cuenta)
    return dfv


def add_duplicate_last_row(df_visible: pd.DataFrame, cols_to_dup: List[str]) -> pd.DataFrame:
    df2 = df_visible.copy().reset_index(drop=True)
    if df2.empty:
        return df2

    last_row = None
    for i in range(len(df2) - 1, -1, -1):
        row = df2.iloc[i]
        if any(str(row.get(c, "")).strip() for c in cols_to_dup):
            last_row = row
            break
    if last_row is None:
        last_row = df2.iloc[-1]

    new = {c: last_row.get(c, "") for c in df2.columns}
    if "id" in new:
        new["id"] = ""
    if "🗑 Eliminar" in new:
        new["🗑 Eliminar"] = False

    df2 = pd.concat([df2, pd.DataFrame([new])], ignore_index=True)
    return df2


def paginate_df(df: pd.DataFrame, page_key: str, page_size: int):
    total = len(df)
    if total == 0:
        return df, 1, 1, total

    pages = max(1, (total + page_size - 1) // page_size)
    page = st.session_state.get(page_key, 1)
    page = min(max(1, page), pages)
    st.session_state[page_key] = page

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    return df.iloc[start:end].copy(), page, pages, total


# ---------- PREPARAR PAYLOAD ----------
def validar_y_preparar_payload_desde_editor(df_edit, modo) -> Tuple[List[str], List[dict], List[dict], List[str]]:
    """
    - tipo se AUTOFIJA por la pestaña (modo).
    - rows_insert: filas sin id (nuevas)
    - rows_upsert: filas con id (ediciones)
    """
    ids_borrar, rows_upsert, rows_insert, avisos = [], [], [], []

    tipo_forzado = {"gastos": "gasto", "ingresos": "ingreso", "transferencias": "transferencia"}.get(modo)
    if tipo_forzado is None:
        tipo_forzado = None

    for idx, r in df_edit.iterrows():
        rid = normalize_id(r.get("id"))
        es_nueva = (rid == "")

        eliminar = bool(r.get("🗑 Eliminar", False))
        if eliminar:
            if not es_nueva:
                ids_borrar.append(rid)
            continue

        fecha_iso = parse_fecha_flexible(r.get("fecha"))
        if not fecha_iso:
            continue

        imp = parse_importe(r.get("importe"))
        if imp is None or imp == 0:
            continue

        desc = safe_str(r.get("descripcion")).strip()
        cuenta = sanitize_choice(r.get("cuenta"), CUENTAS)
        if not cuenta:
            avisos.append(f"Fila {idx+1}: Cuenta inválida o vacía.")
            continue

        payload = {
            "fecha": fecha_iso,
            "descripcion": desc,
            "cuenta": cuenta,
            "importe": float(imp),
            "tipo": tipo_forzado,  # ✅ SIEMPRE según pestaña
        }

        if modo == "gastos":
            categoria = sanitize_choice(r.get("categoria"), CATS_GASTOS)
            if not categoria:
                avisos.append(f"Fila {idx+1}: Categoría inválida o vacía.")
                continue
            payload["categoria"] = categoria
            payload["cuenta_destino"] = None

        elif modo == "ingresos":
            categoria = sanitize_choice(r.get("categoria"), CATS_INGRESOS)
            if not categoria:
                avisos.append(f"Fila {idx+1}: Categoría inválida o vacía.")
                continue
            payload["categoria"] = categoria
            payload["cuenta_destino"] = None

        elif modo == "transferencias":
            cuenta_destino = sanitize_choice(r.get("cuenta_destino"), CUENTAS)
            if not cuenta_destino:
                avisos.append(f"Fila {idx+1}: Cuenta destino inválida o vacía.")
                continue
            if cuenta_destino == cuenta:
                avisos.append(f"Fila {idx+1}: Cuenta destino no puede ser igual a origen.")
                continue
            payload["categoria"] = "Transferencia"
            payload["cuenta_destino"] = cuenta_destino

        payload["mov_hash"] = build_mov_hash(payload)

        if not es_nueva:
            payload_up = dict(payload)
            payload_up["id"] = rid
            rows_upsert.append(payload_up)
        else:
            rows_insert.append(payload)

    def _dedup_rows(rows):
        out = []
        seen = set()
        for x in rows:
            h = x.get("mov_hash")
            if not h:
                out.append(x)
                continue
            if h in seen:
                continue
            seen.add(h)
            out.append(x)
        return out

    rows_insert = _dedup_rows(rows_insert)
    rows_upsert = _dedup_rows(rows_upsert)

    return ids_borrar, rows_upsert, rows_insert, avisos


# ---------- Fingerprint (cambios sin guardar) ----------
def df_fingerprint(df: pd.DataFrame, cols: List[str]) -> str:
    try:
        sub = df[cols].copy()
    except Exception:
        sub = df.copy()
    sub = sub.fillna("")
    payload = sub.astype(str).to_dict(orient="records")
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def unsaved_banner(tab_key: str, df_edit: pd.DataFrame, cols: List[str]):
    fp = df_fingerprint(df_edit, cols)
    saved_fp_key = f"{tab_key}_saved_fp"
    current_fp_key = f"{tab_key}_current_fp"
    st.session_state[current_fp_key] = fp

    saved_fp = st.session_state.get(saved_fp_key)
    if saved_fp and saved_fp != fp:
        st.warning("⚠️ Tienes cambios sin guardar en esta pestaña.")
    elif not saved_fp:
        st.session_state[saved_fp_key] = fp


def mark_saved(tab_key: str, df_edit: pd.DataFrame, cols: List[str]):
    st.session_state[f"{tab_key}_saved_fp"] = df_fingerprint(df_edit, cols)


# ---------- EXPORTAR ----------
def df_to_excel_bytes(df, sheet_name="Datos"):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output) as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        output.seek(0)
        return output
    except Exception:
        return None


def df_to_pdf_bytes(df, title="Datos"):
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, title)
    y -= 20
    c.setFont("Helvetica", 8)

    cols = list(df.columns)
    x_start = 40

    x = x_start
    for col in cols:
        c.drawString(x, y, str(col)[:15])
        x += 80
    y -= 15

    for _, row in df.iterrows():
        x = x_start
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 8)
        for col in cols:
            c.drawString(x, y, str(row[col])[:15])
            x += 80
        y -= 12

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ---------- CARGA EN MEMORIA ----------
def load_data_once():
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False

    if not st.session_state["data_loaded"]:
        rows_mov = fetch_movimientos()
        rows_saldos = fetch_saldos()
        st.session_state["rows_mov"] = rows_mov
        st.session_state["rows_saldos"] = rows_saldos
        st.session_state["df_base"] = preparar_dataframe_base(rows_mov)
        st.session_state["saldos_init"] = get_saldos_iniciales_from_rows(rows_saldos)
        st.session_state["data_loaded"] = True


def invalidate_data():
    st.session_state["data_loaded"] = False
    for k in ["rows_mov", "rows_saldos", "df_base", "saldos_init"]:
        st.session_state.pop(k, None)


# ---------- APP ----------
st.set_page_config(page_title="Finanzas Familiares", layout="wide")
col_title, col_reset = st.columns([6,1])

col_title, col_reset = st.columns([6,1])

with col_title:
    st.title("Finanzas familiares")

with col_reset:
    if st.button("🧯 Reset locks"):
        for k in list(st.session_state.keys()):
            if k.startswith("saving_") or k.startswith("autosave_lock_"):
                st.session_state[k] = False
        st.toast("Locks reseteados", icon="🧯")

st.markdown("""
<style>
div[data-testid="stDataEditor"] [data-column="id"],
div[data-testid="stDataEditor"] [data-testid="stDataEditorColumnHeader"][data-column="id"]{
  width: 0px !important;
  min-width: 0px !important;
  max-width: 0px !important;
  padding: 0 !important;
  margin: 0 !important;
  opacity: 0 !important;
  border: 0 !important;
}
</style>
""", unsafe_allow_html=True)

default_cuenta = st.sidebar.selectbox("Cuenta por defecto (filas nuevas)", CUENTAS, index=0)
modo_movil = st.sidebar.checkbox("📱 Modo móvil compacto", value=False)
modo_debug = st.sidebar.checkbox("🧪 Debug", value=False)

# ✅ Auto-guardado DESACTIVADO (y eliminado de tabs). Solo guardas con botón.
autosave_activo = st.sidebar.checkbox("💾 Auto-guardado inserts", value=False)

if "saving" not in st.session_state:
    st.session_state["saving"] = False

if st.sidebar.button("🔄 Refrescar datos"):
    invalidate_data()
    st.rerun()

try:
    load_data_once()
    df_base = st.session_state["df_base"]
    saldos_init = st.session_state["saldos_init"]
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")
    st.stop()

anios_disponibles = sorted([int(a) for a in df_base["anio"].dropna().unique()], reverse=True) or [date.today().year]
meses_disponibles = list(range(1, 13))

tab_gastos, tab_ingresos, tab_transf, tab_balances, tab_hist, tab_config = st.tabs(
    ["💸 Gastos", "💰 Ingresos", "🔁 Transferencias", "📊 Balances", "📚 Histórico", "⚙️ Config"]
)

missing_tipo = int(df_base["tipo"].isna().sum()) if "tipo" in df_base.columns else 0
if missing_tipo > 0:
    st.caption(
        f"ℹ️ Hay {missing_tipo} movimiento(s) antiguos sin 'tipo'. "
        "No se clasificarán bien hasta que los actualices (puedes editarlos desde Histórico o rehacerlos)."
    )


def filtros_anio_mes_texto(prefix, modo_movil_local):
    if modo_movil_local:
        anio = st.selectbox("Año", anios_disponibles, key=f"anio_{prefix}_m")
        mes = st.selectbox(
            "Mes",
            ["Todos"] + meses_disponibles,
            key=f"mes_{prefix}_m",
            format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x),
        )
        texto = st.text_input("Buscar en descripción", key=f"busca_{prefix}_m")
        return anio, mes, texto
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            anio = st.selectbox("Año", anios_disponibles, key=f"anio_{prefix}")
        with c2:
            mes = st.selectbox(
                "Mes",
                ["Todos"] + meses_disponibles,
                key=f"mes_{prefix}",
                format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x),
            )
        with c3:
            texto = st.text_input("Buscar en descripción", key=f"busca_{prefix}")
        return anio, mes, texto

def guardar_cambios_robusto(tab_key: str, df_edit: pd.DataFrame, modo: str, cols_fingerprint: List[str]):
    lock_key = f"saving_{tab_key}"

    # lock por pestaña (evita dobles clicks / reruns raros)
    if st.session_state.get(lock_key, False):
        st.warning("Ya hay un guardado en curso en esta pestaña...")
        return

    st.session_state[lock_key] = True
    try:
        ids_borrar, rows_upsert, rows_insert, avisos = validar_y_preparar_payload_desde_editor(df_edit, modo=modo)

        if avisos:
            for a in avisos[:10]:
                st.warning(a)
            if len(avisos) > 10:
                st.caption(f"... +{len(avisos)-10} avisos mas")

        if not rows_insert and not rows_upsert and not ids_borrar:
            st.info("No hay cambios validos para guardar.")
            return

        if modo_debug:
            st.json({
                "borrar": len(ids_borrar),
                "updates_id": len(rows_upsert),
                "inserts_hash": len(rows_insert),
                "sample_insert": rows_insert[:2],
                "sample_update": rows_upsert[:2],
            })

        # 1) inserts: UPSERT por mov_hash (anti duplicados real si hay UNIQUE)
        res_in = upsert_movimientos_by_hash(rows_insert)
        if res_in is None:
            st.error("Fallo el guardado de inserts (UPSERT mov_hash). Revisa indice UNIQUE y columnas.")
            return

        # 2) updates: UPSERT por id
        res_up = upsert_movimientos_by_id(rows_upsert)
        if res_up is None:
            st.error("Fallo el guardado de updates (UPSERT id).")
            return

        # 3) deletes
        ok_del = delete_movimientos_bulk(ids_borrar)
        if not ok_del:
            st.error("Guardado parcial: se inserto/actualizo, pero fallo el borrado.")
        else:
            st.success(f"Guardado OK (inserts: {len(res_in)} | updates: {len(res_up)} | borrados: {len(ids_borrar)})")

        mark_saved(tab_key, df_edit, cols_fingerprint)

        invalidate_data()
        
        # 👇 SALTAR A LA ÚLTIMA PÁGINA SEGÚN PESTAÑA
        if tab_key == "gastos":
            st.session_state["page_g"] = 10**9
        elif tab_key == "ingresos":
            st.session_state["page_i"] = 10**9
        elif tab_key == "transf":
            st.session_state["page_t"] = 10**9
        
        st.rerun()

    except Exception as e:
        # CLAVE: si algo revienta, no dejes el lock pillado
        st.error(f"Error guardando: {e}")
        if modo_debug:
            import traceback
            st.code(traceback.format_exc())
    finally:
        st.session_state[lock_key] = False


# ---------- TAB GASTOS ----------
with tab_gastos:
    st.subheader("Gastos")
    anio_g, mes_g, texto_g = filtros_anio_mes_texto("g", modo_movil)

    df_g = df_base.copy()
    df_g = df_g[df_g["tipo"] == "gasto"]
    df_g = df_g[df_g["anio"] == anio_g]
    if mes_g != "Todos":
        df_g = df_g[df_g["mes"] == mes_g]
    if texto_g:
        df_g = df_g[df_g["descripcion"].str.contains(texto_g, case=False, na=False)]
    df_g = df_g.reset_index(drop=True)

    fp_g = (anio_g, mes_g, (texto_g or "").strip().lower())
    if st.session_state.get("fp_gastos") != fp_g:
        st.session_state["fp_gastos"] = fp_g
        st.session_state["page_g"] = 1

    st.caption(f"Movimientos encontrados: {len(df_g)}")

    # TOTAL DEL FILTRO (no de la pagina)
    total_filtrado_g = float(df_g["importe"].fillna(0).sum())
    st.metric("Total gastos (filtro)", f"{total_filtrado_g:,.2f} €")

    page_size_g = st.selectbox("Filas por pagina", [50, 100, 200, 500], index=1, key="page_size_g")
    df_g_page, page_g, pages_g, total_g_rows = paginate_df(df_g, "page_g", page_size_g)
    df_g_page = df_g_page.reset_index(drop=True)

    if df_g_page.empty:
        df_g_page = pd.DataFrame([{
            "id": "",
            "fecha": today_ddmmyyyy(),   # HOY por defecto (texto)
            "descripcion": "",
            "categoria": "",
            "cuenta": default_cuenta,
            "cuenta_destino": None,
            "importe": "",
        }])
    else:
        # asegura que filas nuevas que añadas queden con fecha HOY por defecto al duplicar/crear
        pass

    visible_cols_g = ["fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_g_editor = build_editor_df(df_g_page, visible_cols_g, default_cuenta=default_cuenta)

    ctop1, _ = st.columns([1, 3])
    with ctop1:
        if st.button("Duplicar ultima fila", key="dup_g"):
            df_g_editor = add_duplicate_last_row(df_g_editor, cols_to_dup=visible_cols_g)
            # si la fila duplicada queda sin fecha, pon HOY
            if df_g_editor.at[len(df_g_editor)-1, "fecha"] in ("", None):
                df_g_editor.at[len(df_g_editor)-1, "fecha"] = today_ddmmyyyy()

    df_g_edit = st.data_editor(
        df_g_editor,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_gastos",
        column_order=["fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar", "id"],
        column_config={
            "id": st.column_config.TextColumn("", disabled=True, width="small"),
            "fecha": st.column_config.TextColumn("Fecha", help="dd/mm/aaaa. Puedes poner solo el dia (ej: 5) o dia/mes (ej: 5/2)"),
            "descripcion": st.column_config.TextColumn("Descripcion"),
            "categoria": st.column_config.SelectboxColumn("Categoria", options=CATS_GASTOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn(""),
        },
    )

    # paginacion
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️", key="prev_g", disabled=(page_g <= 1)):
            st.session_state["page_g"] -= 1
            st.rerun()
    with nav2:
        st.caption(f"Pagina {page_g}/{pages_g} — {total_g_rows} filas")
    with nav3:
        if st.button("➡️", key="next_g", disabled=(page_g >= pages_g)):
            st.session_state["page_g"] += 1
            st.rerun()

    unsaved_banner("gastos", df_g_edit, cols=["id", "fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])

    # GUARDA (form para evitar clicks fantasma)
    with st.form("form_save_gastos", clear_on_submit=False):
        submitted_g = st.form_submit_button("Guardar cambios")

    if submitted_g:
        guardar_cambios_robusto(
            "gastos", df_g_edit, modo="gastos",
            cols_fingerprint=["id", "fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"]
        )


# ---------- TAB INGRESOS ----------
with tab_ingresos:
    st.subheader("Ingresos")
    anio_i, mes_i, texto_i = filtros_anio_mes_texto("i", modo_movil)

    df_i = df_base.copy()
    df_i = df_i[df_i["tipo"] == "ingreso"]
    df_i = df_i[df_i["anio"] == anio_i]
    if mes_i != "Todos":
        df_i = df_i[df_i["mes"] == mes_i]
    if texto_i:
        df_i = df_i[df_i["descripcion"].str.contains(texto_i, case=False, na=False)]
    df_i = df_i.reset_index(drop=True)

    fp_i = (anio_i, mes_i, (texto_i or "").strip().lower())
    if st.session_state.get("fp_ingresos") != fp_i:
        st.session_state["fp_ingresos"] = fp_i
        st.session_state["page_i"] = 1

    st.caption(f"Movimientos encontrados: {len(df_i)}")

    # TOTAL DEL FILTRO
    total_filtrado_i = float(df_i["importe"].fillna(0).sum())
    st.metric("Total ingresos (filtro)", f"{total_filtrado_i:,.2f} €")

    page_size_i = st.selectbox("Filas por pagina", [50, 100, 200, 500], index=1, key="page_size_i")
    df_i_page, page_i, pages_i, total_i_rows = paginate_df(df_i, "page_i", page_size_i)
    df_i_page = df_i_page.reset_index(drop=True)

    if df_i_page.empty:
        df_i_page = pd.DataFrame([{
            "id": "",
            "fecha": today_ddmmyyyy(),
            "descripcion": "",
            "categoria": "",
            "cuenta": default_cuenta,
            "cuenta_destino": None,
            "importe": "",
        }])

    visible_cols_i = ["fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_i_editor = build_editor_df(df_i_page, visible_cols_i, default_cuenta=default_cuenta)

    if st.button("Duplicar ultima fila", key="dup_i"):
        df_i_editor = add_duplicate_last_row(df_i_editor, cols_to_dup=visible_cols_i)
        if df_i_editor.at[len(df_i_editor)-1, "fecha"] in ("", None):
            df_i_editor.at[len(df_i_editor)-1, "fecha"] = today_ddmmyyyy()

    df_i_edit = st.data_editor(
        df_i_editor,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_ingresos",
        column_order=["fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar", "id"],
        column_config={
            "id": st.column_config.TextColumn("", disabled=True, width="small"),
            "fecha": st.column_config.TextColumn("Fecha", help="dd/mm/aaaa. Puedes poner solo el dia (ej: 5) o dia/mes (ej: 5/2)"),
            "descripcion": st.column_config.TextColumn("Descripcion"),
            "categoria": st.column_config.SelectboxColumn("Categoria", options=CATS_INGRESOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn(""),
        },
    )

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️", key="prev_i", disabled=(page_i <= 1)):
            st.session_state["page_i"] -= 1
            st.rerun()
    with nav2:
        st.caption(f"Pagina {page_i}/{pages_i} — {total_i_rows} filas")
    with nav3:
        if st.button("➡️", key="next_i", disabled=(page_i >= pages_i)):
            st.session_state["page_i"] += 1
            st.rerun()

    unsaved_banner("ingresos", df_i_edit, cols=["id", "fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])

    with st.form("form_save_ingresos", clear_on_submit=False):
        submitted_i = st.form_submit_button("Guardar cambios")

    if submitted_i:
        guardar_cambios_robusto(
            "ingresos", df_i_edit, modo="ingresos",
            cols_fingerprint=["id", "fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"]
        )


# ---------- TAB TRANSFERENCIAS ----------
with tab_transf:
    st.subheader("Transferencias")
    anio_t, mes_t, texto_t = filtros_anio_mes_texto("t", modo_movil)

    df_t = df_base.copy()
    df_t = df_t[df_t["tipo"] == "transferencia"]
    df_t = df_t[df_t["anio"] == anio_t]
    if mes_t != "Todos":
        df_t = df_t[df_t["mes"] == mes_t]
    if texto_t:
        df_t = df_t[df_t["descripcion"].str.contains(texto_t, case=False, na=False)]
    df_t = df_t.reset_index(drop=True)

    fp_t = (anio_t, mes_t, (texto_t or "").strip().lower())
    if st.session_state.get("fp_transf") != fp_t:
        st.session_state["fp_transf"] = fp_t
        st.session_state["page_t"] = 1

    st.caption(f"Movimientos encontrados: {len(df_t)}")

    # TOTAL DEL FILTRO (en transferencias normalmente no hace falta, pero lo dejo)
    total_filtrado_t = float(df_t["importe"].fillna(0).sum())
    st.metric("Total transferencias (filtro)", f"{total_filtrado_t:,.2f} €")

    page_size_t = st.selectbox("Filas por pagina", [50, 100, 200, 500], index=1, key="page_size_t")
    df_t_page, page_t, pages_t, total_t_rows = paginate_df(df_t, "page_t", page_size_t)
    df_t_page = df_t_page.reset_index(drop=True)

    if df_t_page.empty:
        df_t_page = pd.DataFrame([{
            "id": "",
            "fecha": today_ddmmyyyy(),
            "descripcion": "",
            "cuenta": default_cuenta,
            "cuenta_destino": "",
            "importe": "",
        }])

    visible_cols_t = ["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]
    df_t_editor = build_editor_df(df_t_page, visible_cols_t, default_cuenta=default_cuenta)

    if st.button("Duplicar ultima fila", key="dup_t"):
        df_t_editor = add_duplicate_last_row(df_t_editor, cols_to_dup=visible_cols_t)
        if df_t_editor.at[len(df_t_editor)-1, "fecha"] in ("", None):
            df_t_editor.at[len(df_t_editor)-1, "fecha"] = today_ddmmyyyy()

    df_t_edit = st.data_editor(
        df_t_editor,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_transf",
        column_order=["fecha", "descripcion", "cuenta", "cuenta_destino", "importe", "🗑 Eliminar", "id"],
        column_config={
            "id": st.column_config.TextColumn("", disabled=True, width="small"),
            "fecha": st.column_config.TextColumn("Fecha", help="dd/mm/aaaa. Puedes poner solo el dia (ej: 5) o dia/mes (ej: 5/2)"),
            "descripcion": st.column_config.TextColumn("Descripcion"),
            "cuenta": st.column_config.SelectboxColumn("Cuenta origen", options=CUENTAS),
            "cuenta_destino": st.column_config.SelectboxColumn("Cuenta destino", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn(""),
        },
    )

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️", key="prev_t", disabled=(page_t <= 1)):
            st.session_state["page_t"] -= 1
            st.rerun()
    with nav2:
        st.caption(f"Pagina {page_t}/{pages_t} — {total_t_rows} filas")
    with nav3:
        if st.button("➡️", key="next_t", disabled=(page_t >= pages_t)):
            st.session_state["page_t"] += 1
            st.rerun()

    unsaved_banner("transf", df_t_edit, cols=["id", "fecha", "descripcion", "cuenta", "cuenta_destino", "importe", "🗑 Eliminar"])

    with st.form("form_save_transf", clear_on_submit=False):
        submitted_t = st.form_submit_button("Guardar cambios")

    if submitted_t:
        guardar_cambios_robusto(
            "transf", df_t_edit, modo="transferencias",
            cols_fingerprint=["id", "fecha", "descripcion", "cuenta", "cuenta_destino", "importe", "🗑 Eliminar"]
        )


# ---------- TAB BALANCES ----------
with tab_balances:
    st.subheader("📊 Balances")

    anio_b = st.selectbox("Año", anios_disponibles, key="anio_bal")
    meses_sel = st.multiselect("Meses", options=meses_disponibles, default=meses_disponibles, format_func=nombre_mes)

    df_v = df_base.copy()
    df_v = df_v[(df_v["anio"] == anio_b) & (df_v["mes"].isin(meses_sel))].copy()

    df_gv = df_v[df_v["tipo"] == "gasto"].copy()
    df_iv = df_v[df_v["tipo"] == "ingreso"].copy()

    total_g = float(df_gv["importe"].fillna(0).sum())
    total_i = float(df_iv["importe"].fillna(0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Ingresos", f"{total_i:,.2f} €")
    c2.metric("Gastos", f"{total_g:,.2f} €")
    c3.metric("Ahorro", f"{(total_i - total_g):,.2f} €")

    st.markdown("---")
    st.markdown("**Saldos por cuenta (hasta fin del año)**")
    saldos = calcular_saldos_por_cuenta(df_base[df_base["anio"] <= anio_b], saldos_iniciales=saldos_init)
    df_saldos = pd.DataFrame([{"Cuenta": c, "Saldo": saldos.get(c, 0.0)} for c in CUENTAS])
    st.dataframe(df_saldos, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📈 Análisis visual (pro)")

    orden_meses = meses_disponibles[:]

    def _mes_label(m):
        try:
            return nombre_mes(int(m))[:3]
        except Exception:
            return str(m)

    for _df in (df_v, df_gv, df_iv):
        _df["mes_num"] = pd.to_numeric(_df["mes"], errors="coerce")
        _df["mes_label"] = _df["mes_num"].apply(_mes_label)

    meses_en_vista = sorted([int(m) for m in df_v["mes_num"].dropna().unique().tolist()])
    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown("**Ingresos vs Gastos vs Ahorro**")
        g_mes = df_gv.groupby(["mes_num", "mes_label"])["importe"].sum().reset_index()
        i_mes = df_iv.groupby(["mes_num", "mes_label"])["importe"].sum().reset_index()

        base = pd.DataFrame({"mes_num": meses_en_vista})
        base["mes_label"] = base["mes_num"].apply(_mes_label)

        base = base.merge(i_mes[["mes_num", "importe"]].rename(columns={"importe": "Ingresos"}), on="mes_num", how="left")
        base = base.merge(g_mes[["mes_num", "importe"]].rename(columns={"importe": "Gastos"}), on="mes_num", how="left")

        base["Ingresos"] = base["Ingresos"].fillna(0.0)
        base["Gastos"] = base["Gastos"].fillna(0.0)
        base["Ahorro"] = base["Ingresos"] - base["Gastos"]

        long = base.melt(["mes_num", "mes_label"], var_name="tipo", value_name="eur")

        if len(meses_en_vista) <= 1:
            chart = (
                alt.Chart(long)
                .mark_bar()
                .encode(
                    x=alt.X("tipo:N", title=""),
                    y=alt.Y("eur:Q", title="€"),
                    tooltip=[alt.Tooltip("tipo:N", title="Tipo"), alt.Tooltip("eur:Q", title="€", format=",.2f")],
                )
                .properties(height=320)
            )
        else:
            chart = (
                alt.Chart(long)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "mes_label:O",
                        title="Mes",
                        sort=[_mes_label(m) for m in orden_meses if m in meses_en_vista],
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("eur:Q", title="€"),
                    color=alt.Color("tipo:N", title=""),
                    tooltip=[
                        alt.Tooltip("mes_label:O", title="Mes"),
                        alt.Tooltip("tipo:N", title="Tipo"),
                        alt.Tooltip("eur:Q", title="€", format=",.2f"),
                    ],
                )
                .properties(height=320)
                .interactive()
            )
        st.altair_chart(chart, use_container_width=True)

    with cc2:
        st.markdown("**Gastos por categoría**")
        if df_gv.empty:
            st.info("Sin gastos en el rango.")
        else:
            top_cats = (
                df_gv.groupby("categoria")["importe"].sum()
                .sort_values(ascending=False).head(8).index.tolist()
            )
            g_cat = df_gv.copy()
            g_cat["cat2"] = g_cat["categoria"].where(g_cat["categoria"].isin(top_cats), "Otros")
            g_cat_mes = g_cat.groupby(["mes_num", "mes_label", "cat2"])["importe"].sum().reset_index()

            if len(meses_en_vista) <= 1:
                tot = g_cat_mes.groupby("cat2")["importe"].sum().sort_values(ascending=False).reset_index()
                chart2 = (
                    alt.Chart(tot)
                    .mark_bar()
                    .encode(
                        y=alt.Y("cat2:N", sort="-x", title=""),
                        x=alt.X("importe:Q", title="€"),
                        tooltip=[
                            alt.Tooltip("cat2:N", title="Categoría"),
                            alt.Tooltip("importe:Q", title="€", format=",.2f"),
                        ],
                    )
                    .properties(height=320)
                )
            else:
                chart2 = (
                    alt.Chart(g_cat_mes)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "mes_label:O",
                            title="Mes",
                            sort=[_mes_label(m) for m in orden_meses if m in meses_en_vista],
                            axis=alt.Axis(labelAngle=0),
                        ),
                        y=alt.Y("importe:Q", title="€", stack="zero"),
                        color=alt.Color("cat2:N", title="Categoría"),
                        tooltip=[
                            alt.Tooltip("mes_label:O", title="Mes"),
                            alt.Tooltip("cat2:N", title="Categoría"),
                            alt.Tooltip("importe:Q", title="€", format=",.2f"),
                        ],
                    )
                    .properties(height=320)
                    .interactive()
                )
            st.altair_chart(chart2, use_container_width=True)


# ---------- TAB HISTÓRICO ----------
with tab_hist:
    st.subheader("📚 Histórico")

    c1, c2, c3 = st.columns(3)
    with c1:
        anio_h = st.selectbox("Año", ["Todos"] + list(anios_disponibles), key="anio_hist")
    with c2:
        mes_h = st.selectbox(
            "Mes",
            ["Todos"] + meses_disponibles,
            key="mes_hist",
            format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x),
        )
    with c3:
        texto_h = st.text_input("Buscar", key="busca_hist", placeholder="Descripción, categoría, cuenta...")

    df_h = df_base.copy()

    if anio_h != "Todos":
        df_h = df_h[df_h["anio"] == anio_h]
    if mes_h != "Todos":
        df_h = df_h[df_h["mes"] == mes_h]
    if texto_h:
        df_h = df_h[df_h.apply(lambda r: texto_h.lower() in str(r).lower(), axis=1)]

    st.write("Movimientos encontrados:", len(df_h))
    columnas = ["fecha", "tipo", "descripcion", "categoria", "cuenta", "cuenta_destino", "importe", "mov_hash"]
    df_hist = df_h[columnas].sort_values("fecha", ascending=False)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    excel_bytes_h = df_to_excel_bytes(df_hist, sheet_name="Historico")
    if excel_bytes_h:
        st.download_button(
            "⬇️ Exportar histórico a Excel",
            data=excel_bytes_h,
            file_name="historico.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    pdf_bytes_h = df_to_pdf_bytes(df_hist, title="Histórico completo")
    if pdf_bytes_h:
        st.download_button(
            "⬇️ Exportar histórico a PDF",
            data=pdf_bytes_h,
            file_name="historico.pdf",
            mime="application/pdf",
        )


# ---------- TAB CONFIG ----------
with tab_config:
    st.subheader("⚙️ Configuración saldos iniciales")

    df_conf = pd.DataFrame({
        "cuenta": CUENTAS,
        "saldo_inicial": [saldos_init.get(c, 0.0) for c in CUENTAS],
    })

    df_conf_edit = st.data_editor(
        df_conf,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        key="editor_saldos",
        column_config={
            "cuenta": st.column_config.TextColumn("Cuenta", disabled=True),
            "saldo_inicial": st.column_config.NumberColumn("Saldo inicial", format="%.2f"),
        },
    )

    if st.button("💾 Guardar saldos", key="save_saldos", disabled=st.session_state["saving"]):
        st.session_state["saving"] = True
        try:
            for _, row in df_conf_edit.iterrows():
                update_saldo_inicial_upsert(row["cuenta"], row["saldo_inicial"] or 0.0)
            st.success("Saldos guardados ✅")
            invalidate_data()
            st.rerun()
        finally:
            st.session_state["saving"] = False
