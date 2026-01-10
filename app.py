import streamlit as st
import pandas as pd
import requests
from datetime import date
import io
import uuid
import json
import hashlib
from typing import List, Tuple, Optional

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

DB_COLUMNS = ["fecha", "descripcion", "categoria", "cuenta", "cuenta_destino", "importe"]

MESES_NOMBRES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

DATE_FORMAT = "DD/MM/YYYY"

def nombre_mes(m):
    return MESES_NOMBRES.get(m, str(m))

# ---------- UTIL: coma/punto + normalización ----------
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

# ---------- SUPABASE (sin llamadas en cada edición) ----------
def supa_get(table: str, params: dict):
    url = f"{BASE_URL}/{table}"
    r = SESSION.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def fetch_movimientos():
    return supa_get(TABLE_MOV, {"select": "*", "order": "fecha.asc"})

def fetch_saldos():
    return supa_get(TABLE_SALDOS, {"select": "*"})

def insert_movimientos_bulk(rows, generar_uuid: bool):
    if not rows:
        return []
    if generar_uuid:
        for row in rows:
            row.setdefault("id", str(uuid.uuid4()))
    url = f"{BASE_URL}/{TABLE_MOV}"
    r = SESSION.post(url, headers=HEADERS, json=rows, timeout=TIMEOUT)
    if r.status_code >= 400:
        show_http_error("INSERT", r, sample_payload=rows[:3])
        return None
    if not r.text:
        return []
    return r.json()

def upsert_movimientos_bulk(rows):
    if not rows:
        return []
    url = f"{BASE_URL}/{TABLE_MOV}?on_conflict=id"
    r = SESSION.post(url, headers=HEADERS_UPSERT, json=rows, timeout=TIMEOUT)
    if r.status_code >= 400:
        show_http_error("UPSERT", r, sample_payload=rows[:3])
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

# ---------- DATA PREP ----------
def preparar_dataframe_base(rows):
    df = pd.DataFrame(rows) if rows else pd.DataFrame([])
    for col in ["id"] + DB_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["fecha"] = df["fecha_dt"].dt.date
    df["anio"] = df["fecha_dt"].dt.year
    df["mes"] = df["fecha_dt"].dt.month

    df["descripcion"] = df["descripcion"].fillna("").astype(str)
    df["categoria"] = df["categoria"].fillna("").astype(str)
    df["cuenta"] = df["cuenta"].fillna("").astype(str)
    df["cuenta_destino"] = df["cuenta_destino"].where(df["cuenta_destino"].notna(), None)

    def _to_float(x):
        if x is None:
            return float("nan")
        try:
            return float(x)
        except Exception:
            v = parse_importe(str(x))
            return float(v) if v is not None else float("nan")

    df["importe"] = df["importe"].apply(_to_float)
    return df

def get_saldos_iniciales_from_rows(rows):
    if not rows:
        return {c: 0.0 for c in CUENTAS}
    df = pd.DataFrame(rows)
    saldos = {c: 0.0 for c in CUENTAS}
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
        cat = row.get("categoria")

        if (destino in (None, "", " ")) and (cat in CATS_INGRESOS):
            if origen in saldos:
                saldos[origen] += imp
        elif (destino in (None, "", " ")) and (cat in CATS_GASTOS):
            if origen in saldos:
                saldos[origen] -= imp
        elif destino not in (None, "", " "):
            if origen in saldos:
                saldos[origen] -= imp
            if destino in saldos:
                saldos[destino] += imp
    return saldos

# ---------- DEFAULTS EN FILAS NUEVAS ----------
def aplicar_defaults_df_editor(df_visible, default_cuenta="Principal"):
    df2 = df_visible.copy()
    if "cuenta" not in df2.columns:
        return df2

    for i in range(len(df2)):
        cuenta = df2.at[i, "cuenta"] if "cuenta" in df2.columns else ""
        cuenta_vacia = (cuenta is None) or (str(cuenta).strip() == "")
        if cuenta_vacia:
            df2.at[i, "cuenta"] = default_cuenta
    return df2

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

def unsaved_banner(tab_key: str, df_edit: pd.DataFrame, cols_fingerprint: List[str]):
    fp = df_fingerprint(df_edit, cols_fingerprint)
    saved_fp_key = f"{tab_key}_saved_fp"
    current_fp_key = f"{tab_key}_current_fp"
    st.session_state[current_fp_key] = fp

    saved_fp = st.session_state.get(saved_fp_key)
    if saved_fp and saved_fp != fp:
        st.warning("⚠️ Tienes cambios sin guardar en esta pestaña.")
    elif not saved_fp:
        st.session_state[saved_fp_key] = fp

def mark_saved(tab_key: str, df_edit: pd.DataFrame, cols_fingerprint: List[str]):
    st.session_state[f"{tab_key}_saved_fp"] = df_fingerprint(df_edit, cols_fingerprint)

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

# ---------- Editor helpers (ID oculto sin columna visible) ----------
def build_editor_df(df_src: pd.DataFrame, visible_cols: List[str], default_cuenta: str) -> pd.DataFrame:
    """
    Devuelve un df para st.data_editor:
    - NO muestra id
    - usa __row_id como índice oculto
    - filas nuevas usan __new__... para evitar índices duplicados
    """
    dfv = df_src.copy()

    # Asegurar columnas
    for c in ["id"] + visible_cols:
        if c not in dfv.columns:
            dfv[c] = None

    dfv = dfv[["id"] + visible_cols].copy()
    dfv["importe"] = dfv["importe"].apply(lambda x: "" if pd.isna(x) else str(x))

    dfv["🗑 Eliminar"] = False

    # defaults (sobre vista con range index)
    dfv = aplicar_defaults_df_editor(dfv, default_cuenta=default_cuenta)

    # índice oculto estable
    dfv["id"] = dfv["id"].fillna("").astype(str)
    row_ids = []
    for i, rid in enumerate(dfv["id"].tolist()):
        if rid.strip():
            row_ids.append(rid.strip())
        else:
            row_ids.append(f"__new__{i}_{uuid.uuid4().hex[:8]}")
    dfv["__row_id"] = row_ids

    # quitamos id visible
    dfv = dfv.drop(columns=["id"])
    dfv = dfv.set_index("__row_id")
    dfv.index.name = ""  # evita el título del índice
    return dfv

def add_duplicate_last_row(df_editor: pd.DataFrame, cols_to_dup: List[str]) -> pd.DataFrame:
    """
    Duplica la última fila útil en un df_editor (con índice oculto).
    """
    df2 = df_editor.copy()
    if df2.empty:
        return df2

    # trabajamos sobre reset para copiar valores
    tmp = df2.reset_index(drop=True)

    last_row = None
    for i in range(len(tmp) - 1, -1, -1):
        row = tmp.iloc[i]
        if any(str(row.get(c, "")).strip() for c in cols_to_dup):
            last_row = row
            break
    if last_row is None:
        last_row = tmp.iloc[-1]

    new = {c: last_row.get(c, "") for c in tmp.columns}
    if "🗑 Eliminar" in new:
        new["🗑 Eliminar"] = False

    tmp = pd.concat([tmp, pd.DataFrame([new])], ignore_index=True)

    # reconstruimos índice oculto (manteniendo los existentes)
    row_ids = []
    for i in range(len(tmp)):
        row_ids.append(f"__new__dup{i}_{uuid.uuid4().hex[:8]}")
    tmp.index = row_ids
    tmp.index.name = ""
    return tmp

# ---------- PREPARAR PAYLOAD (split upsert/insert) ----------
def validar_y_preparar_payload_desde_editor(df_edit: pd.DataFrame, modo: str) -> Tuple[List[str], List[dict], List[dict], List[str]]:
    ids_borrar = []
    rows_upsert = []
    rows_insert = []
    avisos = []

    # índice oculto = id real si no empieza por __new__
    for ridx, r in df_edit.iterrows():
        rid = str(ridx).strip() if ridx is not None else ""

        eliminar = bool(r.get("🗑 Eliminar", False))
        es_nueva = (not rid) or rid.startswith("__new__")

        if eliminar:
            if (not es_nueva) and rid:
                ids_borrar.append(rid)
            continue

        fecha = r.get("fecha")
        if pd.isna(fecha) or fecha in (None, "", "NaT"):
            continue

        imp = parse_importe(r.get("importe"))
        if imp is None or imp == 0:
            continue

        desc = safe_str(r.get("descripcion")).strip()
        cuenta = sanitize_choice(r.get("cuenta"), CUENTAS)
        if not cuenta:
            avisos.append("Cuenta inválida o vacía.")
            continue

        payload = {
            "fecha": normalizar_fecha(fecha),
            "descripcion": desc,
            "cuenta": cuenta,
            "importe": float(imp),
        }

        if modo in ("gastos", "ingresos"):
            opciones = CATS_GASTOS if modo == "gastos" else CATS_INGRESOS
            categoria = sanitize_choice(r.get("categoria"), opciones)
            if not categoria:
                avisos.append("Categoría inválida o vacía.")
                continue
            payload["categoria"] = categoria
            payload["cuenta_destino"] = None

        elif modo == "transferencias":
            cuenta_destino = sanitize_choice(r.get("cuenta_destino"), CUENTAS)
            if not cuenta_destino:
                avisos.append("Cuenta destino inválida o vacía.")
                continue
            if cuenta_destino == cuenta:
                avisos.append("Cuenta destino no puede ser igual a origen.")
                continue
            payload["categoria"] = "Transferencia"
            payload["cuenta_destino"] = cuenta_destino

        if (not es_nueva) and rid:
            payload_up = dict(payload)
            payload_up["id"] = rid
            rows_upsert.append(payload_up)
        else:
            rows_insert.append(payload)

    return ids_borrar, rows_upsert, rows_insert, avisos

def total_importe_col(df_edit, col="importe"):
    return sum(float(parse_importe(x) or 0) for x in df_edit[col].tolist())

# ---------- CARGA EN MEMORIA (anti parpadeo) ----------
def load_data_once():
    """
    Carga datos una vez por sesión.
    Para refrescar: invalidate_data() + st.rerun()
    """
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
    st.session_state.pop("rows_mov", None)
    st.session_state.pop("rows_saldos", None)
    st.session_state.pop("df_base", None)
    st.session_state.pop("saldos_init", None)

# ---------- APP ----------
st.set_page_config(page_title="Finanzas Familiares", layout="wide")
st.title("Finanzas familiares")

# Sidebar
default_cuenta = st.sidebar.selectbox("Cuenta por defecto (filas nuevas)", CUENTAS, index=0)
generar_uuid_inserts = st.sidebar.checkbox("Generar UUID en filas nuevas (recomendado)", value=True)
modo_movil = st.sidebar.checkbox("📱 Modo móvil compacto", value=False)
modo_debug = st.sidebar.checkbox("🧪 Debug", value=False)

# Lock anti doble guardado
if "saving" not in st.session_state:
    st.session_state["saving"] = False

# Botón opcional de refresco manual (por si quieres)
if st.sidebar.button("🔄 Refrescar datos"):
    invalidate_data()
    st.rerun()

# Cargar datos una sola vez (no se re-pide al editar celdas)
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

# ---------- helper filtros ----------
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

# ---------- GUARDADO ROBUSTO ----------
def guardar_cambios_robusto(tab_key: str, df_edit: pd.DataFrame, modo: str, cols_fingerprint: List[str]):
    if st.session_state["saving"]:
        st.info("Guardando…")
        return

    st.session_state["saving"] = True
    try:
        ids_borrar, rows_upsert, rows_insert, avisos = validar_y_preparar_payload_desde_editor(df_edit, modo=modo)

        if avisos:
            for a in avisos[:10]:
                st.warning(a)
            if len(avisos) > 10:
                st.caption(f"... +{len(avisos)-10} avisos más")

        if modo_debug:
            st.write("IDs a borrar:", ids_borrar)
            st.write("Upserts:", len(rows_upsert), "Inserts:", len(rows_insert))
            st.json({"upsert_sample": rows_upsert[:3], "insert_sample": rows_insert[:3]})

        # Orden seguro
        res_in = insert_movimientos_bulk(rows_insert, generar_uuid=generar_uuid_inserts)
        if res_in is None:
            st.error("No se pudo guardar (falló INSERT). No se ha borrado nada.")
            return

        res_up = upsert_movimientos_bulk(rows_upsert)
        if res_up is None:
            st.error("No se pudo guardar (falló UPSERT). No se ha borrado nada.")
            return

        ok_del = delete_movimientos_bulk(ids_borrar)
        if not ok_del:
            st.error("Guardado parcial: se insertó/actualizó, pero falló el borrado.")
        else:
            st.success(f"Guardado ✅ (inserts: {len(res_in)} | updates: {len(res_up)} | borrados: {len(ids_borrar)})")

        # Marcar como guardado + refrescar datos SOLO aquí
        mark_saved(tab_key, df_edit, cols_fingerprint)
        invalidate_data()
        st.rerun()

    finally:
        st.session_state["saving"] = False

# ---------- TAB GASTOS ----------
with tab_gastos:
    st.subheader("Gastos")
    anio_g, mes_g, texto_g = filtros_anio_mes_texto("g", modo_movil)

    df_g = df_base.copy()
    df_g = df_g[df_g["cuenta_destino"].isin([None, "", " "])]
    df_g = df_g[df_g["categoria"].isin(CATS_GASTOS) | df_g["categoria"].isna() | (df_g["categoria"] == "")]
    df_g = df_g[df_g["anio"] == anio_g]
    if mes_g != "Todos":
        df_g = df_g[df_g["mes"] == mes_g]
    if texto_g:
        df_g = df_g[df_g["descripcion"].str.contains(texto_g, case=False, na=False)]
    df_g = df_g.reset_index(drop=True)

    if df_g.empty:
        df_g = pd.DataFrame([{
            "id": "",
            "fecha": None,
            "descripcion": "",
            "categoria": "",
            "cuenta": default_cuenta,
            "cuenta_destino": None,
            "importe": "",
        }])

    visible_cols_g = ["fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_g_editor = build_editor_df(df_g, visible_cols_g, default_cuenta=default_cuenta)

    ctop1, _ = st.columns([1, 3])
    with ctop1:
        if st.button("➕ Duplicar última fila", key="dup_g"):
            df_g_editor = add_duplicate_last_row(df_g_editor, cols_to_dup=["fecha", "descripcion", "categoria", "cuenta", "importe"])

    df_g_edit = st.data_editor(
        df_g_editor,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_gastos",
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "descripcion": st.column_config.TextColumn("Descripción"),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_GASTOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    unsaved_banner("gastos", df_g_edit, cols_fingerprint=["fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])
    st.metric("Total gastos (vista actual)", f"{total_importe_col(df_g_edit):,.2f} €")

    if st.button("💾 Guardar cambios", key="save_gastos", disabled=st.session_state["saving"]):
        guardar_cambios_robusto("gastos", df_g_edit, modo="gastos", cols_fingerprint=["fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])

# ---------- TAB INGRESOS ----------
with tab_ingresos:
    st.subheader("Ingresos")
    anio_i, mes_i, texto_i = filtros_anio_mes_texto("i", modo_movil)

    df_i = df_base.copy()
    df_i = df_i[df_i["cuenta_destino"].isin([None, "", " "])]
    df_i = df_i[df_i["categoria"].isin(CATS_INGRESOS) | df_i["categoria"].isna() | (df_i["categoria"] == "")]
    df_i = df_i[df_i["anio"] == anio_i]
    if mes_i != "Todos":
        df_i = df_i[df_i["mes"] == mes_i]
    if texto_i:
        df_i = df_i[df_i["descripcion"].str.contains(texto_i, case=False, na=False)]
    df_i = df_i.reset_index(drop=True)

    if df_i.empty:
        df_i = pd.DataFrame([{
            "id": "",
            "fecha": None,
            "descripcion": "",
            "categoria": "",
            "cuenta": default_cuenta,
            "cuenta_destino": None,
            "importe": "",
        }])

    visible_cols_i = ["fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_i_editor = build_editor_df(df_i, visible_cols_i, default_cuenta=default_cuenta)

    if st.button("➕ Duplicar última fila", key="dup_i"):
        df_i_editor = add_duplicate_last_row(df_i_editor, cols_to_dup=["fecha", "descripcion", "categoria", "cuenta", "importe"])

    df_i_edit = st.data_editor(
        df_i_editor,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_ingresos",
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "descripcion": st.column_config.TextColumn("Descripción"),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_INGRESOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    unsaved_banner("ingresos", df_i_edit, cols_fingerprint=["fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])
    st.metric("Total ingresos (vista actual)", f"{total_importe_col(df_i_edit):,.2f} €")

    if st.button("💾 Guardar cambios", key="save_ingresos", disabled=st.session_state["saving"]):
        guardar_cambios_robusto("ingresos", df_i_edit, modo="ingresos", cols_fingerprint=["fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])

# ---------- TAB TRANSFERENCIAS ----------
with tab_transf:
    st.subheader("Transferencias")
    anio_t, mes_t, texto_t = filtros_anio_mes_texto("t", modo_movil)

    df_t = df_base.copy()
    df_t = df_t[~df_t["cuenta_destino"].isin([None, "", " "])]
    df_t = df_t[df_t["anio"] == anio_t]
    if mes_t != "Todos":
        df_t = df_t[df_t["mes"] == mes_t]
    if texto_t:
        df_t = df_t[df_t["descripcion"].str.contains(texto_t, case=False, na=False)]
    df_t = df_t.reset_index(drop=True)

    if df_t.empty:
        df_t = pd.DataFrame([{
            "id": "",
            "fecha": None,
            "descripcion": "",
            "cuenta": default_cuenta,
            "cuenta_destino": "",
            "importe": "",
        }])

    visible_cols_t = ["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]
    df_t_editor = build_editor_df(df_t, visible_cols_t, default_cuenta=default_cuenta)

    if st.button("➕ Duplicar última fila", key="dup_t"):
        df_t_editor = add_duplicate_last_row(df_t_editor, cols_to_dup=["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"])

    df_t_edit = st.data_editor(
        df_t_editor,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_transf",
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "descripcion": st.column_config.TextColumn("Descripción"),
            "cuenta": st.column_config.SelectboxColumn("Cuenta origen", options=CUENTAS),
            "cuenta_destino": st.column_config.SelectboxColumn("Cuenta destino", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    unsaved_banner("transf", df_t_edit, cols_fingerprint=["fecha", "descripcion", "cuenta", "cuenta_destino", "importe", "🗑 Eliminar"])

    if st.button("💾 Guardar cambios", key="save_transf", disabled=st.session_state["saving"]):
        guardar_cambios_robusto("transf", df_t_edit, modo="transferencias", cols_fingerprint=["fecha", "descripcion", "cuenta", "cuenta_destino", "importe", "🗑 Eliminar"])

# ---------- TAB BALANCES ----------
with tab_balances:
    st.subheader("📊 Balances")
    anio_b = st.selectbox("Año", anios_disponibles, key="anio_bal")
    meses_sel = st.multiselect("Meses", options=meses_disponibles, default=meses_disponibles, format_func=nombre_mes)

    df_b = df_base[(df_base["anio"] == anio_b) & (df_base["mes"].isin(meses_sel))].copy()

    df_g_b = df_b[(df_b["cuenta_destino"].isin([None, "", " "])) & (df_b["categoria"].isin(CATS_GASTOS))]
    df_i_b = df_b[(df_b["cuenta_destino"].isin([None, "", " "])) & (df_b["categoria"].isin(CATS_INGRESOS))]

    total_g = float(df_g_b["importe"].fillna(0).sum())
    total_i = float(df_i_b["importe"].fillna(0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Ingresos", f"{total_i:,.2f} €")
    c2.metric("Gastos", f"{total_g:,.2f} €")
    c3.metric("Ahorro", f"{(total_i - total_g):,.2f} €")

    st.markdown("**Saldos por cuenta (hasta fin del año)**")
    saldos = calcular_saldos_por_cuenta(df_base[df_base["anio"] <= anio_b], saldos_iniciales=saldos_init)
    df_saldos = pd.DataFrame([{"Cuenta": c, "Saldo": saldos.get(c, 0.0)} for c in CUENTAS])
    st.dataframe(df_saldos, use_container_width=True, hide_index=True)

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

    def detectar_tipo(row):
        if row["cuenta_destino"] not in (None, "", " "):
            return "Transferencia"
        if row["categoria"] in CATS_INGRESOS:
            return "Ingreso"
        if row["categoria"] in CATS_GASTOS:
            return "Gasto"
        return "Otro"

    df_h["tipo"] = df_h.apply(detectar_tipo, axis=1)

    if anio_h != "Todos":
        df_h = df_h[df_h["anio"] == anio_h]
    if mes_h != "Todos":
        df_h = df_h[df_h["mes"] == mes_h]
    if texto_h:
        df_h = df_h[df_h.apply(lambda r: texto_h.lower() in str(r).lower(), axis=1)]

    st.write("Movimientos encontrados:", len(df_h))
    columnas = ["fecha", "tipo", "descripcion", "categoria", "cuenta", "cuenta_destino", "importe"]
    df_hist = df_h[columnas].sort_values("fecha")
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
