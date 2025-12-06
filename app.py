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
    # En vez de hacer raise_for_status directamente, mostramos el mensaje de Supabase
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
    """Actualiza el saldo inicial de una cuenta (tabla saldos_iniciales)."""
    url = f"{BASE_URL}/{TABLE_SALDOS}"
    params = {"cuenta": f"eq.{cuenta}"}
    data = {"saldo_inicial": float(saldo)}
    r = requests.patch(url, headers=HEADERS, params=params, json=data)
    if r.status_code not in (200, 204):
        r.raise_for_status()


# ---------- UTILIDADES DATA ----------
def preparar_dataframe_base():
    rows = get_movimientos()
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame([])

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
    saldos = get_saldos_iniciales()  # saldo inicial

    for _, row in df.iterrows():
        imp = float(row.get("importe") or 0)
        origen = row.get("cuenta")
        destino = row.get("cuenta_destino")
        cat = row.get("categoria")

        # ingresos
        if (destino in (None, "", " ")) and (cat in CATS_INGRESOS):
            if origen in saldos:
                saldos[origen] += imp
        # gastos
        elif (destino in (None, "", " ")) and (cat in CATS_GASTOS):
            if origen in saldos:
                saldos[origen] -= imp
        # transferencias
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
    """Sincroniza cambios con Supabase, incluyendo borrados marcados con la papelera.
       Además limpia NaN/NaT para que el JSON sea válido.
    """
    import pandas as _pd

    if ids_marcados_borrar is None:
        ids_marcados_borrar = set()

    # ---- Borrados (filas que desaparecen + marcadas con papelera) ----
    ids_editados = set(df_edit_full["id"].dropna())
    ids_a_borrar = (ids_originales - ids_editados) | set(ids_marcados_borrar)

    for borrar_id in ids_a_borrar:
        if borrar_id:
            delete_movimiento(borrar_id)

    # ---- Altas / updates ----
    for _, row in df_edit_full.iterrows():
        row_dict = row.to_dict()
        row_id = row_dict.get("id", None)

        # si está marcada para borrar, la ignoramos (ya borrada arriba)
        if row_id in ids_marcados_borrar:
            continue

        # REGLA PARA TRANSFERENCIAS: campos obligatorios mínimos
        if modo == "transferencias":
            if (
                row_dict.get("fecha") in (None, "", "NaT")
                or not row_dict.get("cuenta")
                or not row_dict.get("cuenta_destino")
            ):
                continue
        else:
            # gastos / ingresos: saltamos filas totalmente vacías
            if (
                row_dict.get("fecha") in (None, "", "NaT")
                and not row_dict.get("categoria")
                and not row_dict.get("descripcion")
                and not row_dict.get("importe")
            ):
                continue

        # Normalizar fecha
        row_dict["fecha"] = normalizar_fecha(row_dict.get("fecha"))

        # Importe seguro
        try:
            row_dict["importe"] = float(row_dict.get("importe") or 0)
        except Exception:
            row_dict["importe"] = 0.0

        # Ajustes según modo
        if modo in ("gastos", "ingresos"):
            row_dict["cuenta_destino"] = None
        elif modo == "transferencias":
            row_dict["categoria"] = row_dict.get("categoria") or "Transferencia"

        # Construimos el diccionario final SIN NaN / NaT
        data = {}
        for col in DB_COLUMNS:
            v = row_dict.get(col, None)
            # pd.isna() detecta NaN, NaT, pd.NA, etc.
            if _pd.isna(v):
                v = None
            data[col] = v

        # Update o insert
        if row_id and _pd.notna(row_id):
            update_movimiento(row_id, data)
        else:
            insert_movimiento(data)



# ---------- EXPORTAR ----------
def df_to_excel_bytes(df, sheet_name="Datos"):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output) as writer:  # engine auto (openpyxl/xlsxwriter)
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

# Modo móvil compacto
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

# ---------- TAB GASTOS ----------
with tab_gastos:
    st.subheader("Gastos")

    if modo_movil:
        anio_g = st.selectbox("Año", anios_disponibles, key="anio_g_m")
        mes_g = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_g_m")
        texto_g = st.text_input("Buscar en descripción", key="busca_g_m")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            anio_g = st.selectbox("Año", anios_disponibles, key="anio_g")
        with col_f2:
            mes_g = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_g")
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

    if df_g.empty:
        df_g = pd.DataFrame([{
            "id": None,
            "fecha": None,
            "descripcion": "",
            "categoria": "",
            "cuenta": "",
            "cuenta_destino": None,
            "importe": 0.0,
        }])

    df_g_orig = df_g.copy()
    ids_originales_g = set(df_g_orig["id"].dropna())

    visible_g = ["fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_g_visible = df_g_orig[visible_g].copy()
    df_g_visible["🗑 Eliminar"] = False

    df_g_edit_visible = st.data_editor(
        df_g_visible,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_gastos",
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_GASTOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.NumberColumn("Importe", format="%.2f"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑 Eliminar"),
        },
    )

    rows_full = []
    for idx, row in df_g_edit_visible.iterrows():
        if idx in df_g_orig.index:
            base = df_g_orig.loc[idx, ["id"] + DB_COLUMNS].to_dict()
        else:
            base = {"id": None}
            for col in DB_COLUMNS:
                base.setdefault(col, None)
        for col in visible_g:
            base[col] = row.get(col)
        base["eliminar"] = bool(row.get("🗑 Eliminar", False))
        rows_full.append(base)

    df_g_full = pd.DataFrame(rows_full)
    ids_marcados_borrar_g = set(df_g_full.loc[df_g_full["eliminar"], "id"].dropna())
    df_g_full_db = df_g_full.drop(columns=["eliminar"])

    guardar_cambios(df_g_full_db, ids_originales_g, modo="gastos", ids_marcados_borrar=ids_marcados_borrar_g)

    total_g = float(df_g_full_db["importe"].fillna(0).sum())
    st.metric("Total gastos (vista actual)", f"{total_g:,.2f} €")

    export_g = df_g_full_db[["fecha", "descripcion", "categoria", "cuenta", "importe"]].copy()

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        excel_bytes = df_to_excel_bytes(export_g, sheet_name="Gastos")
        if excel_bytes:
            st.download_button(
                "⬇️ Exportar gastos a Excel",
                data=excel_bytes,
                file_name=f"gastos_{anio_g}_{mes_g if mes_g!='Todos' else 'todos'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.caption("Para exportar a Excel, instala 'openpyxl' o 'xlsxwriter' en el entorno de Streamlit.")

    with col_e2:
        pdf_bytes = df_to_pdf_bytes(export_g, title="Gastos filtrados")
        if pdf_bytes:
            st.download_button(
                "⬇️ Exportar gastos a PDF",
                data=pdf_bytes,
                file_name=f"gastos_{anio_g}_{mes_g if mes_g!='Todos' else 'todos'}.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("Para exportar a PDF, instala 'reportlab' en el entorno de Streamlit.")


# ---------- TAB INGRESOS ----------
with tab_ingresos:
    st.subheader("Ingresos")

    if modo_movil:
        anio_i = st.selectbox("Año", anios_disponibles, key="anio_i_m")
        mes_i = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_i_m")
        texto_i = st.text_input("Buscar en descripción", key="busca_i_m")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            anio_i = st.selectbox("Año", anios_disponibles, key="anio_i")
        with col_f2:
            mes_i = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_i")
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
        df_i = pd.DataFrame([{
            "id": None,
            "fecha": None,
            "descripcion": "",
            "categoria": "",
            "cuenta": "",
            "cuenta_destino": None,
            "importe": 0.0,
        }])

    df_i_orig = df_i.copy()
    ids_originales_i = set(df_i_orig["id"].dropna())

    visible_i = ["fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_i_visible = df_i_orig[visible_i].copy()
    df_i_visible["🗑 Eliminar"] = False

    df_i_edit_visible = st.data_editor(
        df_i_visible,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_ingresos",
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_INGRESOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.NumberColumn("Importe", format="%.2f"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑 Eliminar"),
        },
    )

    rows_full_i = []
    for idx, row in df_i_edit_visible.iterrows():
        if idx in df_i_orig.index:
            base = df_i_orig.loc[idx, ["id"] + DB_COLUMNS].to_dict()
        else:
            base = {"id": None}
            for col in DB_COLUMNS:
                base.setdefault(col, None)
        for col in visible_i:
            base[col] = row.get(col)
        base["eliminar"] = bool(row.get("🗑 Eliminar", False))
        rows_full_i.append(base)

    df_i_full = pd.DataFrame(rows_full_i)
    ids_marcados_borrar_i = set(df_i_full.loc[df_i_full["eliminar"], "id"].dropna())
    df_i_full_db = df_i_full.drop(columns=["eliminar"])

    guardar_cambios(df_i_full_db, ids_originales_i, modo="ingresos", ids_marcados_borrar=ids_marcados_borrar_i)

    total_i = float(df_i_full_db["importe"].fillna(0).sum())
    st.metric("Total ingresos (vista actual)", f"{total_i:,.2f} €")

    export_i = df_i_full_db[["fecha", "descripcion", "categoria", "cuenta", "importe"]].copy()

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        excel_bytes_i = df_to_excel_bytes(export_i, sheet_name="Ingresos")
        if excel_bytes_i:
            st.download_button(
                "⬇️ Exportar ingresos a Excel",
                data=excel_bytes_i,
                file_name=f"ingresos_{anio_i}_{mes_i if mes_i!='Todos' else 'todos'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.caption("Para exportar a Excel, instala 'openpyxl' o 'xlsxwriter' en el entorno de Streamlit.")

    with col_e2:
        pdf_bytes_i = df_to_pdf_bytes(export_i, title="Ingresos filtrados")
        if pdf_bytes_i:
            st.download_button(
                "⬇️ Exportar ingresos a PDF",
                data=pdf_bytes_i,
                file_name=f"ingresos_{anio_i}_{mes_i if mes_i!='Todos' else 'todos'}.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("Para exportar a PDF, instala 'reportlab' en el entorno de Streamlit.")


# ---------- TAB TRANSFERENCIAS ----------
with tab_transf:
    st.subheader("Transferencias entre cuentas")

    if modo_movil:
        anio_t = st.selectbox("Año", anios_disponibles, key="anio_t_m")
        mes_t = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_t_m")
        texto_t = st.text_input("Buscar en descripción", key="busca_t_m")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            anio_t = st.selectbox("Año", anios_disponibles, key="anio_t")
        with col_f2:
            mes_t = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_t")
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
        df_t = pd.DataFrame([{
            "id": None,
            "fecha": None,
            "descripcion": "",
            "categoria": "Transferencia",
            "cuenta": "",
            "cuenta_destino": "",
            "importe": 0.0,
        }])

    df_t_orig = df_t.copy()
    ids_originales_t = set(df_t_orig["id"].dropna())

    visible_t = ["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]
    df_t_visible = df_t_orig[visible_t].copy()
    df_t_visible["🗑 Eliminar"] = False

    df_t_edit_visible = st.data_editor(
        df_t_visible,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_transf",
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
            "cuenta": st.column_config.SelectboxColumn("Cuenta origen", options=CUENTAS),
            "cuenta_destino": st.column_config.SelectboxColumn("Cuenta destino", options=CUENTAS),
            "importe": st.column_config.NumberColumn("Importe", format="%.2f"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑 Eliminar"),
        },
    )

    rows_full_t = []
    for idx, row in df_t_edit_visible.iterrows():
        if idx in df_t_orig.index:
            base = df_t_orig.loc[idx, ["id"] + DB_COLUMNS].to_dict()
        else:
            base = {"id": None}
            for col in DB_COLUMNS:
                base.setdefault(col, None)
        for col in visible_t:
            base[col] = row.get(col)
        base["eliminar"] = bool(row.get("🗑 Eliminar", False))
        rows_full_t.append(base)

    df_t_full = pd.DataFrame(rows_full_t)
    df_t_full["categoria"] = "Transferencia"
    ids_marcados_borrar_t = set(df_t_full.loc[df_t_full["eliminar"], "id"].dropna())
    df_t_full_db = df_t_full.drop(columns=["eliminar"])

    guardar_cambios(df_t_full_db, ids_originales_t, modo="transferencias", ids_marcados_borrar=ids_marcados_borrar_t)

    export_t = df_t_full_db[["fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]].copy()

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        excel_bytes_t = df_to_excel_bytes(export_t, sheet_name="Transferencias")
        if excel_bytes_t:
            st.download_button(
                "⬇️ Exportar transferencias a Excel",
                data=excel_bytes_t,
                file_name=f"transferencias_{anio_t}_{mes_t if mes_t!='Todos' else 'todos'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.caption("Para exportar a Excel, instala 'openpyxl' o 'xlsxwriter' en el entorno de Streamlit.")

    with col_e2:
        pdf_bytes_t = df_to_pdf_bytes(export_t, title="Transferencias filtradas")
        if pdf_bytes_t:
            st.download_button(
                "⬇️ Exportar transferencias a PDF",
                data=pdf_bytes_t,
                file_name=f"transferencias_{anio_t}_{mes_t if mes_t!='Todos' else 'todos'}.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("Para exportar a PDF, instala 'reportlab' en el entorno de Streamlit.")


# ---------- TAB BALANCES ----------
with tab_balances:
    st.subheader("📊 Balances por año y saldos por cuenta")

    # años de balances a partir de 2026 (si no hay, usamos todos)
    anios_balances = sorted([a for a in anios_disponibles if a >= 2026], reverse=True) or anios_disponibles
    anio_b = st.selectbox("Año para análisis", anios_balances, key="anio_bal")
    meses_sel_bal = st.multiselect(
        "Meses a visualizar",
        options=meses_disponibles,
        default=meses_disponibles,
        key="meses_bal",
    )

    df_b = df_base[(df_base["anio"] == anio_b) & (df_base["mes"].isin(meses_sel_bal))].copy()

    df_g_b = df_b[
        (df_b["cuenta_destino"].isin([None, "", " "])) &
        (df_b["categoria"].isin(CATS_GASTOS))
    ]
    df_i_b = df_b[
        (df_b["cuenta_destino"].isin([None, "", " "])) &
        (df_b["categoria"].isin(CATS_INGRESOS))
    ]

    total_g_anual = float(df_g_b["importe"].fillna(0).sum())
    total_i_anual = float(df_i_b["importe"].fillna(0).sum())
    ahorro_anual = total_i_anual - total_g_anual

    if modo_movil:
        st.metric("Ingresos (meses seleccionados)", f"{total_i_anual:,.2f} €")
        st.metric("Gastos (meses seleccionados)", f"{total_g_anual:,.2f} €")
        st.metric("Ahorro (meses seleccionados)", f"{ahorro_anual:,.2f} €")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Ingresos (meses seleccionados)", f"{total_i_anual:,.2f} €")
        with c2:
            st.metric("Gastos (meses seleccionados)", f"{total_g_anual:,.2f} €")
        with c3:
            st.metric("Ahorro (meses seleccionados)", f"{ahorro_anual:,.2f} €")

    # Serie mensual (periodo YYYY-MM)
    data_mes = []
    for m in sorted(meses_sel_bal):
        gi = df_i_b[df_i_b["mes"] == m]["importe"].sum()
        gg = df_g_b[df_g_b["mes"] == m]["importe"].sum()
        data_mes.append({
            "periodo": f"{anio_b}-{m:02d}",
            "Ingresos": gi,
            "Gastos": gg,
            "Ahorro": gi - gg,
        })
    if data_mes:
        df_mes = pd.DataFrame(data_mes).set_index("periodo")
        st.markdown("**Evolución mensual (Ingresos, Gastos, Ahorro)**")
        st.line_chart(df_mes)

    if modo_movil:
        st.markdown("**Gasto por categoría**")
        if not df_g_b.empty:
            gastos_cat = df_g_b.groupby("categoria")["importe"].sum().sort_values(ascending=False)
            st.bar_chart(gastos_cat)
        else:
            st.info("No hay gastos para el filtro seleccionado.")

        st.markdown("**Ingresos por categoría**")
        if not df_i_b.empty:
            ingresos_cat = df_i_b.groupby("categoria")["importe"].sum().sort_values(ascending=False)
            st.bar_chart(ingresos_cat)
        else:
            st.info("No hay ingresos para el filtro seleccionado.")
    else:
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.markdown("**Gasto por categoría**")
            if not df_g_b.empty:
                gastos_cat = df_g_b.groupby("categoria")["importe"].sum().sort_values(ascending=False)
                st.bar_chart(gastos_cat)
            else:
                st.info("No hay gastos para el filtro seleccionado.")
        with col_graf2:
            st.markdown("**Ingresos por categoría**")
            if not df_i_b.empty:
                ingresos_cat = df_i_b.groupby("categoria")["importe"].sum().sort_values(ascending=False)
                st.bar_chart(ingresos_cat)
            else:
                st.info("No hay ingresos para el filtro seleccionado.")

    st.markdown("---")
    st.markdown("**Saldos por cuenta (hasta el final del año seleccionado)**")

    df_saldos_base = df_base[df_base["anio"] <= anio_b].copy()
    saldos = calcular_saldos_por_cuenta(df_saldos_base)
    df_saldos = pd.DataFrame(
        [{"Cuenta": c, "Saldo": sal} for c, sal in saldos.items()]
    )

    if modo_movil:
        st.dataframe(df_saldos, use_container_width=True)
        st.bar_chart(df_saldos.set_index("Cuenta")["Saldo"])
    else:
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            st.dataframe(df_saldos, use_container_width=True)
        with col_s2:
            st.bar_chart(df_saldos.set_index("Cuenta")["Saldo"])


# ---------- TAB HISTÓRICO COMPLETO ----------
with tab_hist:
    st.subheader("📚 Histórico completo de movimientos")

    colh1, colh2, colh3 = st.columns(3)
    with colh1:
        anio_h = st.selectbox("Año", ["Todos"] + list(anios_disponibles), key="anio_hist")
    with colh2:
        mes_h = st.selectbox("Mes", ["Todos"] + meses_disponibles, key="mes_hist")
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

    columnas_hist = [
        "fecha", "tipo", "descripcion", "categoria", "cuenta", "cuenta_destino", "importe"
    ]
    df_hist_visible = df_h[columnas_hist].sort_values("fecha")

    st.dataframe(df_hist_visible, use_container_width=True)

    excel_bytes_h = df_to_excel_bytes(df_hist_visible, sheet_name="Historico")
    if excel_bytes_h:
        st.download_button(
            "⬇️ Exportar histórico a Excel",
            data=excel_bytes_h,
            file_name=f"historico.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.caption("Para exportar histórico a Excel, instala 'openpyxl' o 'xlsxwriter' en el entorno de Streamlit.")

    pdf_bytes_h = df_to_pdf_bytes(df_hist_visible, title="Histórico completo")
    if pdf_bytes_h:
        st.download_button(
            "⬇️ Exportar histórico a PDF",
            data=pdf_bytes_h,
            file_name="historico.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("Para exportar histórico a PDF, instala 'reportlab' en el entorno de Streamlit.")


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
        column_config={
            "cuenta": st.column_config.TextColumn("Cuenta", disabled=True),
            "saldo_inicial": st.column_config.NumberColumn("Saldo inicial", format="%.2f"),
        },
    )

    for _, row in df_conf_edit.iterrows():
        cuenta = row["cuenta"]
        saldo = row["saldo_inicial"] or 0.0
        update_saldo_inicial(cuenta, saldo)

    st.success("Saldos iniciales actualizados ✅")
