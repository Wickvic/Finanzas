import streamlit as st
import pandas as pd
import requests
from datetime import date
import io

# ---------- PDF (reportlab) ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

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
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

def nombre_mes(m):
    return MESES_NOMBRES.get(m, str(m))

# ---------- FUNCIONES SUPABASE ----------
def get_table(table):
    url = f"{BASE_URL}/{table}"
    params = {"select": "*"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

def get_movimientos():
    url = f"{BASE_URL}/{TABLE_MOV}"
    params = {"select": "*", "order": "fecha.asc"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

def insert_movimiento(data):
    url = f"{BASE_URL}/{TABLE_MOV}"
    r = requests.post(url, headers=HEADERS, json=data)
    if r.status_code >= 400:
        st.error(f"Error al insertar en Supabase ({r.status_code}): {r.text}")
        return None
    return r.json()

def update_movimiento(row_id, data):
    url = f"{BASE_URL}/{TABLE_MOV}"
    params = {"id": f"eq.{row_id}"}
    r = requests.patch(url, headers=HEADERS, params=params, json=data)
    if r.status_code >= 400:
        st.error(f"Error al actualizar en Supabase ({r.status_code}): {r.text}")
        return None
    return r.json()

def delete_movimiento(row_id):
    url = f"{BASE_URL}/{TABLE_MOV}"
    params = {"id": f"eq.{row_id}"}
    r = requests.delete(url, headers=HEADERS, params=params)
    if r.status_code >= 400:
        st.error(f"Error al borrar en Supabase ({r.status_code}): {r.text}")
        return False
    return True

def update_saldo_inicial(cuenta, saldo):
    url = f"{BASE_URL}/{TABLE_SALDOS}"
    params = {"cuenta": f"eq.{cuenta}"}
    data = {"saldo_inicial": float(saldo)}
    r = requests.patch(url, headers=HEADERS, params=params, json=data)
    if r.status_code not in (200, 204):
        r.raise_for_status()

# ---------- UTILIDADES DATA ----------
def preparar_dataframe_base():
    rows = get_movimientos()
    df = pd.DataFrame(rows) if rows else pd.DataFrame([])

    for col in ["id"] + DB_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    df["fecha_dt"] = pd.to_datetime(df["fecha"])
    df["anio"] = df["fecha_dt"].dt.year
    df["mes"] = df["fecha_dt"].dt.month
    return df

def get_saldos_iniciales():
    try:
        rows = get_table(TABLE_SALDOS)
        if not rows:
            return {c: 0.0 for c in CUENTAS}
        df = pd.DataFrame(rows)
        saldos = {c: 0.0 for c in CUENTAS}
        for _, row in df.iterrows():
            cuenta = row.get("cuenta")
            saldo = float(row.get("saldo_inicial") or 0)
            if cuenta in saldos:
                saldos[cuenta] = saldo
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

def normalizar_fecha(valor):
    from datetime import date as _date
    import pandas as _pd

    if isinstance(valor, _pd.Timestamp):
        return valor.date().isoformat()
    if isinstance(valor, _date):
        return valor.isoformat()
    return valor

def guardar_cambios(df_edit_full, ids_originales, modo, ids_marcados_borrar=None):
    import pandas as _pd

    if ids_marcados_borrar is None:
        ids_marcados_borrar = set()

    ids_editados = set(df_edit_full["id"].dropna())
    ids_a_borrar = (ids_originales - ids_editados) | set(ids_marcados_borrar)

    for borrar_id in ids_a_borrar:
        if borrar_id:
            delete_movimiento(borrar_id)

    for _, row in df_edit_full.iterrows():
        row_dict = row.to_dict()
        row_id = row_dict.get("id", None)

        if row_id in ids_marcados_borrar:
            continue

        if modo == "transferencias":
            fecha_ok = row_dict.get("fecha") not in (None, "", "NaT")
            cuenta_ok = bool(row_dict.get("cuenta"))
            cuenta_dest_ok = bool(row_dict.get("cuenta_destino"))
            importe_ok = row_dict.get("importe") not in (None, "", 0, 0.0)
            if not (fecha_ok and cuenta_ok and cuenta_dest_ok and importe_ok):
                continue
        else:
            fecha_ok = row_dict.get("fecha") not in (None, "", "NaT")
            categoria_ok = bool(row_dict.get("categoria"))
            cuenta_ok = bool(row_dict.get("cuenta"))
            importe_ok = row_dict.get("importe") not in (None, "", 0, 0.0)

            if (
                row_dict.get("fecha") in (None, "", "NaT")
                and not row_dict.get("categoria")
                and not row_dict.get("descripcion")
                and not row_dict.get("cuenta")
                and not row_dict.get("importe")
            ):
                continue

            if not (fecha_ok and categoria_ok and cuenta_ok and importe_ok):
                continue

        row_dict["fecha"] = normalizar_fecha(row_dict.get("fecha"))
        try:
            row_dict["importe"] = float(row_dict.get("importe") or 0)
        except Exception:
            row_dict["importe"] = 0.0

        if modo in ("gastos", "ingresos"):
            row_dict["cuenta_destino"] = None
        elif modo == "transferencias":
            row_dict["categoria"] = row_dict.get("categoria") or "Transferencia"

        data = {}
        for col in DB_COLUMNS:
            v = row_dict.get(col, None)
            if _pd.isna(v):
                v = None
            data[col] = v

        if row_id and _pd.notna(row_id):
            update_movimiento(row_id, data)
        else:
            insert_movimiento(data)

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

# ---------- SESSION HELPERS ----------
def ensure_editor_df(state_key: str, df_default: pd.DataFrame):
    """Guarda un df en session_state para evitar que el rerun pise lo que estás escribiendo."""
    if state_key not in st.session_state:
        st.session_state[state_key] = df_default.copy()

def add_empty_row(state_key: str, modo: str):
    df = st.session_state[state_key].copy()
    base = {
        "fecha": None,
        "descripcion": "",
        "categoria": "" if modo in ("gastos", "ingresos") else "Transferencia",
        "cuenta": "",
        "cuenta_destino": None if modo in ("gastos", "ingresos") else "",
        "importe": 0.0,
        "🗑 Eliminar": False,
    }
    df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
    st.session_state[state_key] = df

# ---------- APP ----------
st.set_page_config(page_title="Finanzas Familiares", layout="wide")
st.title("Finanzas familiares")

modo_movil = st.sidebar.checkbox("📱 Modo móvil compacto", value=False)

try:
    df_base = preparar_dataframe_base()
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")
    st.stop()

anios_disponibles = sorted(df_base["anio"].dropna().unique(), reverse=True) or [date.today().year]
meses_disponibles = list(range(1, 13))

tab_gastos, tab_ingresos, tab_transf, tab_balances, tab_hist, tab_config = st.tabs(
    ["💸 Gastos", "💰 Ingresos", "🔁 Transferencias", "📊 Balances", "📚 Histórico completo", "⚙️ Configuración"]
)

DATE_COL = st.column_config.DateColumn("Fecha", format="DD/MM/YYYY")

# ---------- TAB GASTOS ----------
with tab_gastos:
    st.subheader("Gastos")

    if modo_movil:
        anio_g = st.selectbox("Año", anios_disponibles, key="anio_g_m")
        mes_g = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_g_m",
                             format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x))
        texto_g = st.text_input("Buscar en descripción", key="busca_g_m")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            anio_g = st.selectbox("Año", anios_disponibles, key="anio_g")
        with col_f2:
            mes_g = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_g",
                                 format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x))
        with col_f3:
            texto_g = st.text_input("Buscar en descripción", key="busca_g")

    df_g = df_base.copy()
    df_g = df_g[df_g["cuenta_destino"].isin([None, "", " "])]
    df_g = df_g[df_g["categoria"].isin(CATS_GASTOS) | df_g["categoria"].isna() | (df_g["categoria"] == "")]
    df_g = df_g[df_g["anio"] == anio_g]
    if mes_g != "Todos":
        df_g = df_g[df_g["mes"] == mes_g]
    if texto_g:
        df_g = df_g[df_g["descripcion"].str.contains(texto_g, case=False, na=False)]

    # Visible en editor
    visible_g = ["fecha", "descripcion", "categoria", "cuenta", "importe", "🗑 Eliminar"]
    if df_g.empty:
        df_g_visible = pd.DataFrame([{
            "fecha": None, "descripcion": "", "categoria": "", "cuenta": "", "importe": 0.0, "🗑 Eliminar": False
        }])
        df_g_orig = pd.DataFrame([{
            "id": None, "fecha": None, "descripcion": "", "categoria": "", "cuenta": "", "cuenta_destino": None, "importe": 0.0
        }])
    else:
        df_g_orig = df_g.copy()
        df_g_visible = df_g_orig[["fecha", "descripcion", "categoria", "cuenta", "importe"]].copy()
        df_g_visible["🗑 Eliminar"] = False

    ids_originales_g = set(df_g_orig["id"].dropna())

    state_key_g = "df_editor_gastos"
    ensure_editor_df(state_key_g, df_g_visible)

    colb1, colb2 = st.columns([1, 1])
    with colb1:
        if st.button("➕ Añadir fila", key="add_row_gastos"):
            add_empty_row(state_key_g, modo="gastos")
    with colb2:
        btn_guardar_g = st.button("💾 Guardar cambios", key="save_gastos")

    df_g_edit_visible = st.data_editor(
        st.session_state[state_key_g],
        num_rows="dynamic",
        use_container_width=True,
        key="editor_gastos",
        hide_index=True,  # ✅ adiós numerito
        column_config={
            "fecha": DATE_COL,
            "descripcion": st.column_config.TextColumn("Descripción"),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_GASTOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.NumberColumn("Importe", format="%.2f"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑 Eliminar"),
        },
    )

    # Persistimos lo editado (evita pérdidas al Enter/Tab)
    st.session_state[state_key_g] = df_g_edit_visible.copy()

    # Reconstrucción full (con id oculto detrás)
    rows_full = []
    for idx, row in df_g_edit_visible.iterrows():
        if idx in df_g_orig.index:
            base = df_g_orig.loc[idx, ["id"] + DB_COLUMNS].to_dict()
        else:
            base = {"id": None}
            for col in DB_COLUMNS:
                base.setdefault(col, None)
        for col in ["fecha", "descripcion", "categoria", "cuenta", "importe"]:
            base[col] = row.get(col)
        base["eliminar"] = bool(row.get("🗑 Eliminar", False))
        rows_full.append(base)

    df_g_full = pd.DataFrame(rows_full)
    ids_marcados_borrar_g = set(df_g_full.loc[df_g_full["eliminar"], "id"].dropna())
    df_g_full_db = df_g_full.drop(columns=["eliminar"])

    if btn_guardar_g:
        guardar_cambios(df_g_full_db, ids_originales_g, modo="gastos", ids_marcados_borrar=ids_marcados_borrar_g)
        st.success("Guardado ✅")
        st.rerun()

    total_g = float(df_g_full_db["importe"].fillna(0).sum())
    st.metric("Total gastos (vista actual)", f"{total_g:,.2f} €")

    export_g = df_g_full_db[["fecha", "descripcion", "categoria", "cuenta", "importe"]].copy()

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        mes_nombre_g = "todos" if mes_g == "Todos" else nombre_mes(mes_g).lower()
        excel_bytes = df_to_excel_bytes(export_g, sheet_name="Gastos")
        if excel_bytes:
            st.download_button("⬇️ Exportar gastos a Excel", data=excel_bytes,
                               file_name=f"gastos_{anio_g}_{mes_nombre_g}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.caption("Para exportar a Excel, instala 'openpyxl' o 'xlsxwriter'.")

    with col_e2:
        mes_nombre_g = "todos" if mes_g == "Todos" else nombre_mes(mes_g).lower()
        pdf_bytes = df_to_pdf_bytes(export_g, title="Gastos filtrados")
        if pdf_bytes:
            st.download_button("⬇️ Exportar gastos a PDF", data=pdf_bytes,
                               file_name=f"gastos_{anio_g}_{mes_nombre_g}.pdf",
                               mime="application/pdf")
        else:
            st.caption("Para exportar a PDF, instala 'reportlab'.")

# ---------- TAB INGRESOS ----------
with tab_ingresos:
    st.subheader("Ingresos")

    if modo_movil:
        anio_i = st.selectbox("Año", anios_disponibles, key="anio_i_m")
        mes_i = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_i_m",
                             format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x))
        texto_i = st.text_input("Buscar en descripción", key="busca_i_m")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            anio_i = st.selectbox("Año", anios_disponibles, key="anio_i")
        with col_f2:
            mes_i = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_i",
                                 format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x))
        with col_f3:
            texto_i = st.text_input("Buscar en descripción", key="busca_i")

    df_i = df_base.copy()
    df_i = df_i[df_i["cuenta_destino"].isin([None, "", " "])]
    df_i = df_i[df_i["categoria"].isin(CATS_INGRESOS) | df_i["categoria"].isna() | (df_i["categoria"] == "")]
    df_i = df_i[df_i["anio"] == anio_i]
    if mes_i != "Todos":
        df_i = df_i[df_i["mes"] == mes_i]
    if texto_i:
        df_i = df_i[df_i["descripcion"].str.contains(texto_i, case=False, na=False)]

    if df_i.empty:
        df_i_visible = pd.DataFrame([{
            "fecha": None, "descripcion": "", "categoria": "", "cuenta": "", "importe": 0.0, "🗑 Eliminar": False
        }])
        df_i_orig = pd.DataFrame([{
            "id": None, "fecha": None, "descripcion": "", "categoria": "", "cuenta": "", "cuenta_destino": None, "importe": 0.0
        }])
    else:
        df_i_orig = df_i.copy()
        df_i_visible = df_i_orig[["fecha", "descripcion", "categoria", "cuenta", "importe"]].copy()
        df_i_visible["🗑 Eliminar"] = False

    ids_originales_i = set(df_i_orig["id"].dropna())

    state_key_i = "df_editor_ingresos"
    ensure_editor_df(state_key_i, df_i_visible)

    colb1, colb2 = st.columns([1, 1])
    with colb1:
        if st.button("➕ Añadir fila", key="add_row_ingresos"):
            add_empty_row(state_key_i, modo="ingresos")
    with colb2:
        btn_guardar_i = st.button("💾 Guardar cambios", key="save_ingresos")

    df_i_edit_visible = st.data_editor(
        st.session_state[state_key_i],
        num_rows="dynamic",
        use_container_width=True,
        key="editor_ingresos",
        hide_index=True,
        column_config={
            "fecha": DATE_COL,
            "descripcion": st.column_config.TextColumn("Descripción"),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_INGRESOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.NumberColumn("Importe", format="%.2f"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑 Eliminar"),
        },
    )
    st.session_state[state_key_i] = df_i_edit_visible.copy()

    rows_full_i = []
    for idx, row in df_i_edit_visible.iterrows():
        if idx in df_i_orig.index:
            base = df_i_orig.loc[idx, ["id"] + DB_COLUMNS].to_dict()
        else:
            base = {"id": None}
            for col in DB_COLUMNS:
                base.setdefault(col, None)
        for col in ["fecha", "descripcion", "categoria", "cuenta", "importe"]:
            base[col] = row.get(col)
        base["eliminar"] = bool(row.get("🗑 Eliminar", False))
        rows_full_i.append(base)

    df_i_full = pd.DataFrame(rows_full_i)
    ids_marcados_borrar_i = set(df_i_full.loc[df_i_full["eliminar"], "id"].dropna())
    df_i_full_db = df_i_full.drop(columns=["eliminar"])

    if btn_guardar_i:
        guardar_cambios(df_i_full_db, ids_originales_i, modo="ingresos", ids_marcados_borrar=ids_marcados_borrar_i)
        st.success("Guardado ✅")
        st.rerun()

    total_i = float(df_i_full_db["importe"].fillna(0).sum())
    st.metric("Total ingresos (vista actual)", f"{total_i:,.2f} €")

    export_i = df_i_full_db[["fecha", "descripcion", "categoria", "cuenta", "importe"]].copy()

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        mes_nombre_i = "todos" if mes_i == "Todos" else nombre_mes(mes_i).lower()
        excel_bytes_i = df_to_excel_bytes(export_i, sheet_name="Ingresos")
        if excel_bytes_i:
            st.download_button("⬇️ Exportar ingresos a Excel", data=excel_bytes_i,
                               file_name=f"ingresos_{anio_i}_{mes_nombre_i}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.caption("Para exportar a Excel, instala 'openpyxl' o 'xlsxwriter'.")

    with col_e2:
        mes_nombre_i = "todos" if mes_i == "Todos" else nombre_mes(mes_i).lower()
        pdf_bytes_i = df_to_pdf_bytes(export_i, title="Ingresos filtrados")
        if pdf_bytes_i:
            st.download_button("⬇️ Exportar ingresos a PDF", data=pdf_bytes_i,
                               file_name=f"ingresos_{anio_i}_{mes_nombre_i}.pdf",
                               mime="application/pdf")
        else:
            st.caption("Para exportar a PDF, instala 'reportlab'.")

# ---------- TAB TRANSFERENCIAS ----------
with tab_transf:
    st.subheader("Transferencias entre cuentas")

    if modo_movil:
        anio_t = st.selectbox("Año", anios_disponibles, key="anio_t_m")
        mes_t = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_t_m",
                             format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x))
        texto_t = st.text_input("Buscar en descripción", key="busca_t_m")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            anio_t = st.selectbox("Año", anios_disponibles, key="anio_t")
        with col_f2:
            mes_t = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_t",
                                 format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x))
        with col_f3:
            texto_t = st.text_input("Buscar en descripción", key="busca_t")

    df_t = df_base.copy()
    df_t = df_t[~df_t["cuenta_destino"].isin([None, "", " "])]
    df_t = df_t[df_t["anio"] == anio_t]
    if mes_t != "Todos":
        df_t = df_t[df_t["mes"] == mes_t]
    if texto_t:
        df_t = df_t[df_t["descripcion"].str.contains(texto_t, case=False, na=False)]

    if df_t.empty:
        df_t_visible = pd.DataFrame([{
            "fecha": None, "descripcion": "", "cuenta": "", "cuenta_destino": "", "importe": 0.0, "🗑 Eliminar": False
        }])
        df_t_orig = pd.DataFrame([{
            "id": None, "fecha": None, "descripcion": "", "categoria": "Transferencia",
            "cuenta": "", "cuenta_destino": "", "importe": 0.0
        }])
    else:
        df_t_orig = df_t.copy()
        df_t_visible = df_t_orig[["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]].copy()
        df_t_visible["🗑 Eliminar"] = False

    ids_originales_t = set(df_t_orig["id"].dropna())

    state_key_t = "df_editor_transferencias"
    ensure_editor_df(state_key_t, df_t_visible)

    colb1, colb2 = st.columns([1, 1])
    with colb1:
        if st.button("➕ Añadir fila", key="add_row_transf"):
            add_empty_row(state_key_t, modo="transferencias")
    with colb2:
        btn_guardar_t = st.button("💾 Guardar cambios", key="save_transf")

    df_t_edit_visible = st.data_editor(
        st.session_state[state_key_t],
        num_rows="dynamic",
        use_container_width=True,
        key="editor_transf",
        hide_index=True,
        column_config={
            "fecha": DATE_COL,
            "descripcion": st.column_config.TextColumn("Descripción"),
            "cuenta": st.column_config.SelectboxColumn("Cuenta origen", options=CUENTAS),
            "cuenta_destino": st.column_config.SelectboxColumn("Cuenta destino", options=CUENTAS),
            "importe": st.column_config.NumberColumn("Importe", format="%.2f"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑 Eliminar"),
        },
    )
    st.session_state[state_key_t] = df_t_edit_visible.copy()

    rows_full_t = []
    for idx, row in df_t_edit_visible.iterrows():
        if idx in df_t_orig.index:
            base = df_t_orig.loc[idx, ["id"] + DB_COLUMNS].to_dict()
        else:
            base = {"id": None}
            for col in DB_COLUMNS:
                base.setdefault(col, None)
        base["categoria"] = "Transferencia"
        for col in ["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]:
            base[col] = row.get(col)
        base["eliminar"] = bool(row.get("🗑 Eliminar", False))
        rows_full_t.append(base)

    df_t_full = pd.DataFrame(rows_full_t)
    ids_marcados_borrar_t = set(df_t_full.loc[df_t_full["eliminar"], "id"].dropna())
    df_t_full_db = df_t_full.drop(columns=["eliminar"])

    if btn_guardar_t:
        guardar_cambios(df_t_full_db, ids_originales_t, modo="transferencias", ids_marcados_borrar=ids_marcados_borrar_t)
        st.success("Guardado ✅")
        st.rerun()

    export_t = df_t_full_db[["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]].copy()

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        mes_nombre_t = "todos" if mes_t == "Todos" else nombre_mes(mes_t).lower()
        excel_bytes_t = df_to_excel_bytes(export_t, sheet_name="Transferencias")
        if excel_bytes_t:
            st.download_button("⬇️ Exportar transferencias a Excel", data=excel_bytes_t,
                               file_name=f"transferencias_{anio_t}_{mes_nombre_t}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.caption("Para exportar a Excel, instala 'openpyxl' o 'xlsxwriter'.")

    with col_e2:
        mes_nombre_t = "todos" if mes_t == "Todos" else nombre_mes(mes_t).lower()
        pdf_bytes_t = df_to_pdf_bytes(export_t, title="Transferencias filtradas")
        if pdf_bytes_t:
            st.download_button("⬇️ Exportar transferencias a PDF", data=pdf_bytes_t,
                               file_name=f"transferencias_{anio_t}_{mes_nombre_t}.pdf",
                               mime="application/pdf")
        else:
            st.caption("Para exportar a PDF, instala 'reportlab'.")

# ---------- TAB BALANCES ----------
with tab_balances:
    st.subheader("📊 Balances por año y saldos por cuenta")

    anios_balances = sorted([a for a in anios_disponibles if a >= 2026], reverse=True) or anios_disponibles
    anio_b = st.selectbox("Año para análisis", anios_balances, key="anio_bal")
    meses_sel_bal = st.multiselect(
        "Meses a visualizar",
        options=meses_disponibles,
        default=meses_disponibles,
        key="meses_bal",
        format_func=nombre_mes,
    )

    df_b = df_base[(df_base["anio"] == anio_b) & (df_base["mes"].isin(meses_sel_bal))].copy()

    df_g_b = df_b[(df_b["cuenta_destino"].isin([None, "", " "])) & (df_b["categoria"].isin(CATS_GASTOS))]
    df_i_b = df_b[(df_b["cuenta_destino"].isin([None, "", " "])) & (df_b["categoria"].isin(CATS_INGRESOS))]

    total_g_anual = float(df_g_b["importe"].fillna(0).sum())
    total_i_anual = float(df_i_b["importe"].fillna(0).sum())
    ahorro_anual = total_i_anual - total_g_anual

    if modo_movil:
        st.metric("Ingresos (meses seleccionados)", f"{total_i_anual:,.2f} €")
        st.metric("Gastos (meses seleccionados)", f"{total_g_anual:,.2f} €")
        st.metric("Ahorro (meses seleccionados)", f"{ahorro_anual:,.2f} €")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Ingresos (meses seleccionados)", f"{total_i_anual:,.2f} €")
        c2.metric("Gastos (meses seleccionados)", f"{total_g_anual:,.2f} €")
        c3.metric("Ahorro (meses seleccionados)", f"{ahorro_anual:,.2f} €")

    st.markdown("**Saldos por cuenta (hasta el final del año seleccionado)**")
    df_saldos_base = df_base[df_base["anio"] <= anio_b].copy()
    saldos = calcular_saldos_por_cuenta(df_saldos_base)
    df_saldos = pd.DataFrame([{"Cuenta": c, "Saldo": sal} for c, sal in saldos.items()])

    if modo_movil:
        st.dataframe(df_saldos, use_container_width=True, hide_index=True)
        st.bar_chart(df_saldos.set_index("Cuenta")["Saldo"])
    else:
        col_s1, col_s2 = st.columns([2, 1])
        col_s1.dataframe(df_saldos, use_container_width=True, hide_index=True)
        col_s2.bar_chart(df_saldos.set_index("Cuenta")["Saldo"])

    st.markdown("---")

    if modo_movil:
        st.markdown("**Gasto por categoría**")
        if not df_g_b.empty:
            st.bar_chart(df_g_b.groupby("categoria")["importe"].sum().sort_values(ascending=False))
        else:
            st.info("No hay gastos para el filtro seleccionado.")

        st.markdown("**Ingresos por categoría**")
        if not df_i_b.empty:
            st.bar_chart(df_i_b.groupby("categoria")["importe"].sum().sort_values(ascending=False))
        else:
            st.info("No hay ingresos para el filtro seleccionado.")
    else:
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.markdown("**Gasto por categoría**")
            if not df_g_b.empty:
                st.bar_chart(df_g_b.groupby("categoria")["importe"].sum().sort_values(ascending=False))
            else:
                st.info("No hay gastos para el filtro seleccionado.")
        with col_graf2:
            st.markdown("**Ingresos por categoría**")
            if not df_i_b.empty:
                st.bar_chart(df_i_b.groupby("categoria")["importe"].sum().sort_values(ascending=False))
            else:
                st.info("No hay ingresos para el filtro seleccionado.")

# ---------- TAB HISTÓRICO COMPLETO ----------
with tab_hist:
    st.subheader("📚 Histórico completo de movimientos")

    colh1, colh2, colh3 = st.columns(3)
    with colh1:
        anio_h = st.selectbox("Año", ["Todos"] + list(anios_disponibles), key="anio_hist")
    with colh2:
        mes_h = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_hist",
                             format_func=lambda x: "Todos" if x == "Todos" else nombre_mes(x))
    with colh3:
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

    columnas_hist = ["fecha", "tipo", "descripcion", "categoria", "cuenta", "cuenta_destino", "importe"]
    df_hist_visible = df_h[columnas_hist].sort_values("fecha").copy()

    # Mostrar fechas dd/mm/aaaa también aquí (solo visual)
    df_hist_visible["fecha"] = pd.to_datetime(df_hist_visible["fecha"]).dt.strftime("%d/%m/%Y")

    st.dataframe(df_hist_visible, use_container_width=True, hide_index=True)

    excel_bytes_h = df_to_excel_bytes(df_hist_visible, sheet_name="Historico")
    if excel_bytes_h:
        st.download_button("⬇️ Exportar histórico a Excel", data=excel_bytes_h,
                           file_name="historico.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Para exportar histórico a Excel, instala 'openpyxl' o 'xlsxwriter'.")

    pdf_bytes_h = df_to_pdf_bytes(df_hist_visible, title="Histórico completo")
    if pdf_bytes_h:
        st.download_button("⬇️ Exportar histórico a PDF", data=pdf_bytes_h,
                           file_name="historico.pdf", mime="application/pdf")
    else:
        st.caption("Para exportar a PDF, instala 'reportlab'.")

# ---------- TAB CONFIGURACIÓN ----------
with tab_config:
    st.subheader("⚙️ Configuración de saldos iniciales")

    saldos_init = get_saldos_iniciales()
    df_conf = pd.DataFrame({
        "cuenta": CUENTAS,
        "saldo_inicial": [saldos_init[c] for c in CUENTAS],
    })

    df_conf_edit = st.data_editor(
        df_conf,
        num_rows="fixed",
        use_container_width=True,
        key="editor_saldos_iniciales",
        hide_index=True,
        column_config={
            "cuenta": st.column_config.TextColumn("Cuenta", disabled=True),
            "saldo_inicial": st.column_config.NumberColumn("Saldo inicial", format="%.2f"),
        },
    )

    if st.button("💾 Guardar saldos iniciales", key="save_saldos_iniciales"):
        for _, row in df_conf_edit.iterrows():
            cuenta = row["cuenta"]
            saldo = row["saldo_inicial"] or 0.0
            update_saldo_inicial(cuenta, saldo)
        st.success("Saldos iniciales actualizados ✅")
