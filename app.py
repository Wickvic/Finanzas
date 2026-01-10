import streamlit as st
import pandas as pd
import requests
from datetime import date
import io
import uuid

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


# ---------- UTIL: coma/punto ----------
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


# ---------- SUPABASE ----------
def get_table(table):
    url = f"{BASE_URL}/{table}"
    r = requests.get(url, headers=HEADERS, params={"select": "*"}, timeout=30)
    r.raise_for_status()
    return r.json()


def get_movimientos():
    url = f"{BASE_URL}/{TABLE_MOV}"
    r = requests.get(url, headers=HEADERS, params={"select": "*", "order": "fecha.asc"}, timeout=30)
    r.raise_for_status()
    return r.json()


def insert_movimientos_bulk(rows):
    """INSERT: filas nuevas SIN id. Si tu tabla no tiene default para id, generamos UUID aquí."""
    if not rows:
        return []
    # Si tu tabla no genera id, descomenta estas 2 líneas:
    # for row in rows:
    #     row.setdefault("id", str(uuid.uuid4()))

    url = f"{BASE_URL}/{TABLE_MOV}"
    r = requests.post(url, headers=HEADERS, json=rows, timeout=30)

    st.write("INSERT status:", r.status_code)
    if r.status_code >= 400:
        st.error(f"INSERT ERROR: {r.status_code}")
        st.code(r.text)
        return None

    if not r.text:
        st.warning("INSERT OK pero sin body")
        return []
    return r.json()


def upsert_movimientos_bulk(rows):
    """UPSERT: solo filas con id (updates)."""
    if not rows:
        return []
    url = f"{BASE_URL}/{TABLE_MOV}?on_conflict=id"
    r = requests.post(url, headers=HEADERS_UPSERT, json=rows, timeout=30)

    st.write("UPSERT status:", r.status_code)
    if r.status_code >= 400:
        st.error(f"UPSERT ERROR: {r.status_code}")
        st.code(r.text)
        return None

    if not r.text:
        st.warning("UPSERT OK pero sin body (r.text vacío)")
        return []

    try:
        return r.json()
    except Exception:
        st.warning("UPSERT OK pero respuesta no es JSON")
        st.code(r.text)
        return []


def delete_movimientos_bulk(ids):
    ids = [str(i) for i in ids if i]
    if not ids:
        return True
    url = f"{BASE_URL}/{TABLE_MOV}"
    params = {"id": f"in.({','.join(ids)})"}
    r = requests.delete(url, headers=HEADERS, params=params, timeout=30)
    st.write("DELETE status:", r.status_code)
    if r.status_code >= 400:
        st.error(f"DELETE ERROR: {r.status_code}")
        st.code(r.text)
        return False
    return True


def update_saldo_inicial_upsert(cuenta, saldo):
    url = f"{BASE_URL}/{TABLE_SALDOS}"
    r0 = requests.get(url, headers=HEADERS, params={"select": "*", "cuenta": f"eq.{cuenta}"}, timeout=30)
    r0.raise_for_status()
    existe = len(r0.json()) > 0

    if existe:
        r = requests.patch(
            url, headers=HEADERS, params={"cuenta": f"eq.{cuenta}"},
            json={"saldo_inicial": float(saldo)}, timeout=30
        )
    else:
        r = requests.post(
            url, headers=HEADERS,
            json={"cuenta": cuenta, "saldo_inicial": float(saldo)}, timeout=30
        )

    if r.status_code >= 400:
        st.error(f"SALDOS ERROR: {r.status_code}")
        st.code(r.text)
        r.raise_for_status()


# ---------- DATA ----------
def preparar_dataframe_base():
    rows = get_movimientos()
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
        rows = get_table(TABLE_SALDOS)
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


# ---------- COPIAR FECHAS ----------
def rellenar_fechas_huecas_con_anterior(df_edit, fecha_col="fecha"):
    """Rellena fechas vacías con la última fecha válida anterior (útil para copiar fecha fila a fila)."""
    df2 = df_edit.copy()
    last = None
    for i in range(len(df2)):
        v = df2.at[i, fecha_col] if fecha_col in df2.columns else None
        if v in (None, "", "NaT") or (pd.isna(v) if not isinstance(v, date) else False):
            if last is not None:
                df2.at[i, fecha_col] = last
        else:
            last = v
    return df2


# ---------- PREPARAR PAYLOAD (split upsert/insert) ----------
def validar_y_preparar_payload_desde_editor(df_edit, modo):
    ids_borrar = []
    rows_upsert = []   # con id
    rows_insert = []   # sin id

    for _, r in df_edit.iterrows():
        rid = r.get("id", None)
        eliminar = bool(r.get("🗑 Eliminar", False))

        if eliminar:
            if pd.notna(rid) and str(rid).strip() != "":
                ids_borrar.append(rid)
            continue

        fecha = r.get("fecha")
        if pd.isna(fecha) or fecha in (None, "", "NaT"):
            continue

        imp = parse_importe(r.get("importe"))
        if imp is None or imp == 0:
            continue

        desc = (r.get("descripcion") or "").strip()
        cuenta = (r.get("cuenta") or "").strip()
        if not cuenta:
            continue

        payload = {
            "fecha": normalizar_fecha(fecha),
            "descripcion": desc,
            "cuenta": cuenta,
            "importe": float(imp),
        }

        if modo in ("gastos", "ingresos"):
            categoria = (r.get("categoria") or "").strip()
            if not categoria:
                continue
            payload["categoria"] = categoria
            payload["cuenta_destino"] = None

        elif modo == "transferencias":
            cuenta_destino = (r.get("cuenta_destino") or "").strip()
            if not cuenta_destino:
                continue
            payload["categoria"] = "Transferencia"
            payload["cuenta_destino"] = cuenta_destino

        if pd.notna(rid) and str(rid).strip() != "":
            payload_up = dict(payload)
            payload_up["id"] = rid
            rows_upsert.append(payload_up)
        else:
            rows_insert.append(payload)

    return ids_borrar, rows_upsert, rows_insert


# ---------- APP ----------
st.set_page_config(page_title="Finanzas Familiares", layout="wide")
st.title("Finanzas familiares")

modo_movil = st.sidebar.checkbox("📱 Modo móvil compacto", value=False)
modo_debug = st.sidebar.checkbox("🧪 Modo debug guardado", value=False)

try:
    df_base = preparar_dataframe_base()
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")
    st.stop()

anios_disponibles = sorted([int(a) for a in df_base["anio"].dropna().unique()], reverse=True) or [date.today().year]
meses_disponibles = list(range(1, 13))

tab_gastos, tab_ingresos, tab_transf, tab_balances, tab_hist, tab_config, tab_debug = st.tabs(
    ["💸 Gastos", "💰 Ingresos", "🔁 Transferencias", "📊 Balances", "📚 Histórico", "⚙️ Config", "🧪 Debug"]
)

# ---------- TAB DEBUG ----------
with tab_debug:
    st.subheader("🧪 Debug Supabase")
    st.caption("Si el INSERT falla por id NOT NULL, activa el uuid en tabla o descomenta el generador UUID en insert_movimientos_bulk().")

    if st.button("🧪 Test INSERT directo (1€ en Principal/Cesta)"):
        test = [{
            "fecha": date.today().isoformat(),
            "descripcion": "TEST",
            "categoria": "Cesta",
            "cuenta": "Principal",
            "cuenta_destino": None,
            "importe": 1.23,
        }]
        res = insert_movimientos_bulk(test)
        st.write("Respuesta:", res)

    st.markdown("---")
    st.write("Primeras 10 filas cargadas desde Supabase:")
    st.dataframe(df_base.head(10), use_container_width=True, hide_index=True)


# ---------- HELPERS UI ----------
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


def total_importe_col(df_edit, col="importe"):
    return sum(float(parse_importe(x) or 0) for x in df_edit[col].tolist())


def aplicar_defaults_nuevas_filas(df_visible, modo):
    """
    - Cuenta por defecto: Principal en filas nuevas (id vacío) y cuenta vacía.
    """
    df2 = df_visible.copy()
    if "cuenta" in df2.columns:
        for i in range(len(df2)):
            rid = df2.at[i, "id"] if "id" in df2.columns else ""
            cuenta = df2.at[i, "cuenta"]
            if (rid in (None, "", "nan")) and (cuenta in (None, "", "nan")):
                df2.at[i, "cuenta"] = "Principal"
    return df2


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
            "cuenta": "Principal",
            "cuenta_destino": None,
            "importe": "",
        }])

    visible = ["id", "fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_g_visible = df_g[visible].copy()
    df_g_visible["importe"] = df_g_visible["importe"].apply(lambda x: "" if pd.isna(x) else str(x))
    df_g_visible["🗑 Eliminar"] = False
    df_g_visible = aplicar_defaults_nuevas_filas(df_g_visible, modo="gastos")

    st.caption("💡 Puedes copiar/pegar celdas (Ctrl+C / Ctrl+V). Botón de abajo rellena fechas vacías con la de arriba.")
    colb1, colb2 = st.columns([1, 2])
    with colb1:
        if st.button("📅 Rellenar fechas vacías", key="fill_dates_g"):
            st.session_state["editor_gastos"] = rellenar_fechas_huecas_con_anterior(df_g_visible)

    df_g_edit = st.data_editor(
        df_g_visible,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_gastos",
        column_config={
            "id": st.column_config.TextColumn("", disabled=True),  # casi oculto
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_GASTOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    st.metric("Total gastos (vista actual)", f"{total_importe_col(df_g_edit):,.2f} €")

    if st.button("💾 Guardar cambios (Gastos)", key="save_gastos"):
        ids_borrar, rows_upsert, rows_insert = validar_y_preparar_payload_desde_editor(df_g_edit, modo="gastos")

        if modo_debug:
            st.write("Filas en editor:", len(df_g_edit))
            st.write("IDs a borrar:", ids_borrar)
            st.write("Upserts (con id):", len(rows_upsert))
            st.write("Inserts (sin id):", len(rows_insert))
            st.json({"upsert_first": rows_upsert[:5], "insert_first": rows_insert[:5]})

        ok_del = delete_movimientos_bulk(ids_borrar)
        res_up = upsert_movimientos_bulk(rows_upsert)
        res_in = insert_movimientos_bulk(rows_insert)

        if (res_up is None) or (res_in is None):
            st.error("No se pudo guardar (mira el error arriba).")
        else:
            st.success(f"Guardado ✅ (updates: {len(res_up)} | inserts: {len(res_in)})")
            st.rerun()


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
            "cuenta": "Principal",
            "cuenta_destino": None,
            "importe": "",
        }])

    visible = ["id", "fecha", "descripcion", "categoria", "cuenta", "importe"]
    df_i_visible = df_i[visible].copy()
    df_i_visible["importe"] = df_i_visible["importe"].apply(lambda x: "" if pd.isna(x) else str(x))
    df_i_visible["🗑 Eliminar"] = False
    df_i_visible = aplicar_defaults_nuevas_filas(df_i_visible, modo="ingresos")

    st.caption("💡 Puedes copiar/pegar celdas (Ctrl+C / Ctrl+V). Botón de abajo rellena fechas vacías con la de arriba.")
    if st.button("📅 Rellenar fechas vacías", key="fill_dates_i"):
        st.session_state["editor_ingresos"] = rellenar_fechas_huecas_con_anterior(df_i_visible)

    df_i_edit = st.data_editor(
        df_i_visible,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_ingresos",
        column_config={
            "id": st.column_config.TextColumn("", disabled=True),
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=CATS_INGRESOS),
            "cuenta": st.column_config.SelectboxColumn("Cuenta", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    st.metric("Total ingresos (vista actual)", f"{total_importe_col(df_i_edit):,.2f} €")

    if st.button("💾 Guardar cambios (Ingresos)", key="save_ingresos"):
        ids_borrar, rows_upsert, rows_insert = validar_y_preparar_payload_desde_editor(df_i_edit, modo="ingresos")

        if modo_debug:
            st.write("Filas en editor:", len(df_i_edit))
            st.write("IDs a borrar:", ids_borrar)
            st.write("Upserts (con id):", len(rows_upsert))
            st.write("Inserts (sin id):", len(rows_insert))
            st.json({"upsert_first": rows_upsert[:5], "insert_first": rows_insert[:5]})

        ok_del = delete_movimientos_bulk(ids_borrar)
        res_up = upsert_movimientos_bulk(rows_upsert)
        res_in = insert_movimientos_bulk(rows_insert)

        if (res_up is None) or (res_in is None):
            st.error("No se pudo guardar (mira el error arriba).")
        else:
            st.success(f"Guardado ✅ (updates: {len(res_up)} | inserts: {len(res_in)})")
            st.rerun()


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
            "cuenta": "Principal",
            "cuenta_destino": "",
            "importe": "",
        }])

    visible = ["id", "fecha", "descripcion", "cuenta", "cuenta_destino", "importe"]
    df_t_visible = df_t[visible].copy()
    df_t_visible["importe"] = df_t_visible["importe"].apply(lambda x: "" if pd.isna(x) else str(x))
    df_t_visible["🗑 Eliminar"] = False
    df_t_visible = aplicar_defaults_nuevas_filas(df_t_visible, modo="transferencias")

    st.caption("💡 Puedes copiar/pegar celdas (Ctrl+C / Ctrl+V). Botón de abajo rellena fechas vacías con la de arriba.")
    if st.button("📅 Rellenar fechas vacías", key="fill_dates_t"):
        st.session_state["editor_transf"] = rellenar_fechas_huecas_con_anterior(df_t_visible)

    df_t_edit = st.data_editor(
        df_t_visible,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_transf",
        column_config={
            "id": st.column_config.TextColumn("", disabled=True),
            "fecha": st.column_config.DateColumn("Fecha", format=DATE_FORMAT),
            "cuenta": st.column_config.SelectboxColumn("Cuenta origen", options=CUENTAS),
            "cuenta_destino": st.column_config.SelectboxColumn("Cuenta destino", options=CUENTAS),
            "importe": st.column_config.TextColumn("Importe"),
            "🗑 Eliminar": st.column_config.CheckboxColumn("🗑"),
        },
    )

    if st.button("💾 Guardar cambios (Transferencias)", key="save_transf"):
        ids_borrar, rows_upsert, rows_insert = validar_y_preparar_payload_desde_editor(df_t_edit, modo="transferencias")

        if modo_debug:
            st.write("Filas en editor:", len(df_t_edit))
            st.write("IDs a borrar:", ids_borrar)
            st.write("Upserts (con id):", len(rows_upsert))
            st.write("Inserts (sin id):", len(rows_insert))
            st.json({"upsert_first": rows_upsert[:5], "insert_first": rows_insert[:5]})

        ok_del = delete_movimientos_bulk(ids_borrar)
        res_up = upsert_movimientos_bulk(rows_upsert)
        res_in = insert_movimientos_bulk(rows_insert)

        if (res_up is None) or (res_in is None):
            st.error("No se pudo guardar (mira el error arriba).")
        else:
            st.success(f"Guardado ✅ (updates: {len(res_up)} | inserts: {len(res_in)})")
            st.rerun()


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
    st.dataframe(df_h[columnas].sort_values("fecha"), use_container_width=True, hide_index=True)


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

    if st.button("💾 Guardar saldos", key="save_saldos"):
        for _, row in df_conf_edit.iterrows():
            update_saldo_inicial_upsert(row["cuenta"], row["saldo_inicial"] or 0.0)
        st.success("Saldos guardados ✅")
        st.rerun()
