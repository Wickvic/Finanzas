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
        # Si urllib3/Retry no está disponible por lo que sea, seguimos sin reintentos.
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
        # 1.234,56 -> 1234.56
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        # 12,34 -> 12.34
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


# ---------- SUPABASE (cacheadas) ----------
@st.cache_data(ttl=10)
def get_table_cached(table: str):
    url = f"{BASE_URL}/{table}"
    r = SESSION.get(url, headers=HEADERS, params={"select": "*"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def get_movimientos_cached():
    url = f"{BASE_URL}/{TABLE_MOV}"
    r = SESSION.get(url, headers=HEADERS, params={"select": "*", "order": "fecha.asc"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def cache_clear_all():
    try:
        st.cache_data.clear()
    except Exception:
        pass


def insert_movimientos_bulk(rows, generar_uuid: bool):
    """INSERT: filas nuevas. Si id en tu tabla no tiene default, ponemos UUID."""
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
    """UPSERT: filas con id (updates)."""
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


# ---------- DATA ----------
def preparar_dataframe_base():
    rows = get_movimientos_cached()
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


def get_saldos_iniciales():
    try:
        rows = get_table_cached(TABLE_SALDOS)
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
    except Exception:
        return {c: 0.0 for c in CUENTAS}


def calcular_saldos_por_cuenta(df):
    saldos = get_saldos_iniciales()
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
        rid = df2.at[i, "id"] if "id" in df2.columns else ""
        cuenta = df2.at[i, "cuenta"]
        es_nueva = (rid is None) or (str(rid).strip() == "")
        cuenta_vacia = (cuenta is None) or (str(cuenta).strip() == "")
        if es_nueva and cuenta_vacia:
            df2.at[i, "cuenta"] = default_cuenta
    return df2


# ---------- PREPARAR PAYLOAD (split upsert/insert) ----------
def validar_y_preparar_payload_desde_editor(df_edit, modo) -> Tuple[List[str], List[dict], List[dict], List[str]]:
    ids_borrar = []
    rows_upsert = []   # con id
    rows_insert = []   # sin id
    avisos = []

    for idx, r in df_edit.iterrows():
        rid = r.get("id", None)
        eliminar = bool(r.get("🗑 Eliminar", False))

        if eliminar:
            if pd.notna(rid) and str(rid).strip() != "":
                ids_borrar.append(str(rid))
            continue

        fecha = r.get("fecha")
        if pd.isna(fecha) or fecha in (None, "", "NaT"):
            # fila incompleta => no se guarda
            continue

        imp = parse_importe(r.get("importe"))
        if imp is None or imp == 0:
            # regla: guardamos solo cuando hay importe válido
            continue

        desc = safe_str(r.get("descripcion")).strip()

        cuenta = sanitize_choice(r.get("cuenta"), CUENTAS)
        if not cuenta:
            avisos.append(f"Fila {idx+1}: Cuenta inválida o vacía.")
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

        if pd.notna(rid) and str(rid).strip() != "":
            payload_up = dict(payload)
            payload_up["id"] = str(rid).strip()
            rows_upsert.append(payload_up)
        else:
            rows_insert.append(payload)

    return ids_borrar, rows_upsert, rows_insert, avisos


def total_importe_col(df_edit, col="importe"):
    return sum(float(parse_importe(x) or 0) for x in df_edit[col].tolist())


def df_fingerprint(df: pd.DataFrame, cols: List[str]) -> str:
    """Hash estable para detectar cambios sin guardar (ignora orden de columnas extra)."""
    try:
        sub = df[cols].copy()
    except Exception:
        sub = df.copy()
    sub = sub.fillna("")
    # normalizamos importes a string tal cual están (para detectar cambios)
    payload = sub.astype(str).to_dict(orient="records")
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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


# ---------- APP ----------
st.set_page_config(page_title="Finanzas Familiares", layout="wide")
st.title("Finanzas familiares")

# (8) UI: cuenta por defecto + opción UUID inserts
default_cuenta = st.sidebar.selectbox("Cuenta por defecto (filas nuevas)", CUENTAS, index=0)
generar_uuid_inserts = st.sidebar.checkbox("Generar UUID en filas nuevas (recomendado)", value=True)
modo_movil = st.sidebar.checkbox("📱 Modo móvil compacto", value=False)
modo_debug = st.sidebar.checkbox("🧪 Debug", value=False)

# (2) lock anti-doble guardado
if "saving" not in st.session_state:
    st.session_state["saving"] = False

try:
    df_base = preparar_dataframe_base()
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


# (5) cambios sin guardar: helper banner
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


# (8) UI: duplicar última fila útil
def add_duplicate_last_row(df_visible: pd.DataFrame, cols_to_dup: List[str]) -> pd.DataFrame:
    df2 = df_visible.copy()
    if df2.empty:
        return df2
    # buscamos última fila con algún dato
    last_row = None
    for i in range(len(df2) - 1, -1, -1):
        row = df2.iloc[i]
        if any(str(row.get(c, "")).strip() for c in cols_to_dup):
            last_row = row
            break
    if last_row is None:
        last_row = df2.iloc[-1]
    new = {c: last_row.get(c, "") for c in df2.columns}
    # fila nueva => id vacío, no borrar
    if "id" in new:
        new["id"] = ""
    if "🗑 Eliminar" in new:
        new["🗑 Eliminar"] = False
    df2 = pd.concat([df2, pd.DataFrame([new])], ignore_index=True)
    return df2


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

        # (1) Orden seguro: primero INSERT/UPSERT; luego DELETE
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

        # (7) clear cache + refrescar
        cache_clear_all()
        mark_saved(tab_key, df_edit, cols_fingerprint)
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

    visible_cols = ["id", "fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_g_visible = df_g[visible_cols].copy()
    df_g_visible["importe"] = df_g_visible["importe"].apply(lambda x: "" if pd.isna(x) else str(x))
    df_g_visible["🗑 Eliminar"] = False
    df_g_visible = aplicar_defaults_df_editor(df_g_visible, default_cuenta=default_cuenta)

    ctop1, ctop2 = st.columns([1, 3])
    with ctop1:
        if st.button("➕ Duplicar última fila", key="dup_g"):
            df_g_visible = add_duplicate_last_row(df_g_visible, cols_to_dup=["fecha", "descripcion", "categoria", "cuenta", "importe"])

    df_g_edit = st.data_editor(
        df_g_visible,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_gastos",
        column_config={
            "id": st.column_config.TextColumn("", disabled=True, width="small"),
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_GASTOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    unsaved_banner("gastos", df_g_edit, cols_fingerprint=["id", "fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])
    st.metric("Total gastos (vista actual)", f"{total_importe_col(df_g_edit):,.2f} €")

        if st.button("💾 Guardar cambios", key="save_gastos_real", disabled=st.session_state["saving"]):
        guardar_cambios_robusto(
            "gastos",
            df_g_edit,
            modo="gastos",
            cols_fingerprint=[
                "id",
                "fecha",
                "descripcion",
                "categoria",
                "cuenta",
                "importe",
                "🗑 Eliminar",
            ],
        )



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

    visible_cols = ["id", "fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_i_visible = df_i[visible_cols].copy()
    df_i_visible["importe"] = df_i_visible["importe"].apply(lambda x: "" if pd.isna(x) else str(x))
    df_i_visible["🗑 Eliminar"] = False
    df_i_visible = aplicar_defaults_df_editor(df_i_visible, default_cuenta=default_cuenta)

    if st.button("➕ Duplicar última fila", key="dup_i"):
        df_i_visible = add_duplicate_last_row(df_i_visible, cols_to_dup=["fecha", "descripcion", "categoria", "cuenta", "importe"])

    df_i_edit = st.data_editor(
        df_i_visible,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_ingresos",
        column_config={
            "id": st.column_config.TextColumn("", disabled=True, width="small"),
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_INGRESOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    unsaved_banner("ingresos", df_i_edit, cols_fingerprint=["id", "fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])
    st.metric("Total ingresos (vista actual)", f"{total_importe_col(df_i_edit):,.2f} €")

    if st.button("💾 Guardar cambios", key="save_ingresos_real", disabled=st.session_state["saving"]):
        guardar_cambios_robusto("ingresos", df_i_edit, modo="ingresos", cols_fingerprint=["id", "fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"])


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
            "categoria": "Transferencia",
            "cuenta": default_cuenta,
            "cuenta_destino": "",
            "importe": "",
        }])

    visible_cols = ["id", "fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]
    df_t_visible = df_t[visible_cols].copy()
    df_t_visible["importe"] = df_t_visible["importe"].apply(lambda x: "" if pd.isna(x) else str(x))
    df_t_visible["🗑 Eliminar"] = False
    df_t_visible = aplicar_defaults_df_editor(df_t_visible, default_cuenta=default_cuenta)

    if st.button("➕ Duplicar última fila", key="dup_t"):
        df_t_visible = add_duplicate_last_row(df_t_visible, cols_to_dup=["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"])

    df_t_edit = st.data_editor(
        df_t_visible,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_transf",
        column_config={
            "id": st.column_config.TextColumn("", disabled=True, width="small"),
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "cuenta": st.column_config.SelectboxColumn("Cuenta origen", options=CUENTAS),
            "cuenta_destino": st.column_config.SelectboxColumn("Cuenta destino", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    unsaved_banner("transf", df_t_edit, cols_fingerprint=["id", "fecha", "descripcion", "cuenta", "cuenta_destino", "importe", "🗑 Eliminar"])

    if st.button("💾 Guardar cambios", key="save_transf_real", disabled=st.session_state["saving"]):
        guardar_cambios_robusto("transf", df_t_edit, modo="transferencias", cols_fingerprint=["id", "fecha", "descripcion", "cuenta", "cuenta_destino", "importe", "🗑 Eliminar"])


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
    saldos = calcular_saldos_por_cuenta(df_base[df_base["anio"] <= anio_b])
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

    # export
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

    saldos_init = get_saldos_iniciales()
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
            cache_clear_all()
            st.success("Saldos guardados ✅")
            st.rerun()
        finally:
            st.session_state["saving"] = False
