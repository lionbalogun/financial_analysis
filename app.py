# app.py (with DCF module)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from io import BytesIO
import math

st.set_page_config(page_title="Pride Advisory — Fundamental Analyzer w/ DCF", layout="wide", page_icon="🦁")

# ===== Branding header =====
st.markdown(
    """
    <div style="display:flex; align-items:center;">
      <h1 style='margin-right:16px; color:#A67C00;'>Pride Advisory — Fundamental Analyzer</h1>
      <div style="padding:6px 10px; border-radius:6px; background:#000; color:#A67C00;">Black & Gold • v2.0 (with DCF)</div>
    </div>
    <p>Upload multi-period financial statements (1..n years), map rows once, compute ratios, and run a DCF valuation with sensitivity analysis.</p>
    """,
    unsafe_allow_html=True
)

# ===== Helpers: safe math & market fetch =====
def safe_div(a, b):
    try:
        return a / b if b is not None and b != 0 and not (pd.isna(b)) else np.nan
    except Exception:
        return np.nan

def fetch_market_data(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info if hasattr(t, "info") else {}
        price = info.get('regularMarketPrice') or info.get('previousClose') or None
        shares = info.get('sharesOutstanding') or None
        market_cap = info.get('marketCap') or (price * shares if (price and shares) else None)
        financials = None
        balance = None
        cashflow = None
        try:
            financials = t.financials if hasattr(t, "financials") else None
            balance = t.balance_sheet if hasattr(t, "balance_sheet") else None
            cashflow = t.cashflow if hasattr(t, "cashflow") else None
        except Exception:
            financials, balance, cashflow = None, None, None

        return {"price": price, "shares_outstanding": shares, "market_cap": market_cap,
                "raw_info": info, "financials_df": financials, "balance_df": balance, "cashflow_df": cashflow}
    except Exception as e:
        return {"error": str(e)}

# ===== Multi-period helpers =====
def normalize_period_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_row(df, label):
    if label is None or label == "":
        return pd.Series(dtype=float)
    if label in df.index:
        s = df.loc[label].astype(float)
        s.index = [str(i).strip() for i in s.index]
        return s
    matches = [r for r in df.index if str(r).strip().lower() == str(label).strip().lower()]
    if matches:
        s = df.loc[matches[0]].astype(float)
        s.index = [str(i).strip() for i in s.index]
        return s
    matches2 = [r for r in df.index if str(label).strip().lower() in str(r).strip().lower()]
    if matches2:
        s = df.loc[matches2[0]].astype(float)
        s.index = [str(i).strip() for i in s.index]
        return s
    return pd.Series(dtype=float)

def build_financials_df(income_df, bal_df, cf_df, mapping):
    income_df = normalize_period_cols(income_df) if isinstance(income_df, pd.DataFrame) else pd.DataFrame()
    bal_df = normalize_period_cols(bal_df) if isinstance(bal_df, pd.DataFrame) else pd.DataFrame()
    cf_df = normalize_period_cols(cf_df) if isinstance(cf_df, pd.DataFrame) else pd.DataFrame()

    periods = []
    for df in (income_df, bal_df, cf_df):
        if isinstance(df, pd.DataFrame):
            for c in df.columns:
                if c not in periods:
                    periods.append(c)

    if not periods:
        return pd.DataFrame()

    metrics = list(mapping.keys())
    financials = pd.DataFrame(index=metrics, columns=periods, dtype=float)

    for key, row_label in mapping.items():
        if key in ("revenue", "net_income", "ebitda", "ebit", "interest_expense", "tax_expense"):
            s = pick_row(income_df, row_label) if not income_df.empty else pd.Series(dtype=float)
        elif key in ("total_assets", "current_assets", "inventories", "current_liabilities",
                     "total_liabilities", "total_debt", "cash_and_equivalents",
                     "shareholders_equity", "shares_outstanding", "ppe_gross", "accum_depr"):
            s = pick_row(bal_df, row_label) if not bal_df.empty else pd.Series(dtype=float)
        elif key in ("operating_cf", "capex", "free_cash_flow"):
            s = pick_row(cf_df, row_label) if not cf_df.empty else pd.Series(dtype=float)
        else:
            s = pick_row(income_df, row_label).reindex(periods).combine_first(
                pick_row(bal_df, row_label).reindex(periods)).combine_first(pick_row(cf_df, row_label).reindex(periods))

        for p in periods:
            if p in s.index:
                try:
                    financials.loc[key, p] = float(s.loc[p])
                except Exception:
                    financials.loc[key, p] = np.nan
            else:
                financials.loc[key, p] = np.nan

    return financials

def apply_missing_value_strategy(df, strategy):
    if df.empty:
        return df
    if strategy in (None, "none"):
        return df
    if strategy == "ffill":
        return df.fillna(method='ffill', axis=1)
    if strategy == "bfill":
        return df.fillna(method='bfill', axis=1)
    if strategy == "interpolate":
        return df.interpolate(axis=1, method='linear', limit_direction='both')
    return df

def compute_time_series_ratios(fin_df):
    if fin_df.empty:
        return pd.DataFrame()
    f = fin_df.copy().astype(float)
    periods = f.columns.tolist()

    def s(key):
        return f.loc[key] if key in f.index else pd.Series(index=periods, dtype=float)

    eps = s('net_income').div(s('shares_outstanding'))

    ratios = {
        'Revenue': s('revenue'),
        'Net_Income': s('net_income'),
        'EPS': eps,
        'Net_Margin': s('net_income').div(s('revenue')),
        'ROE': s('net_income').div(s('shareholders_equity')),
        'ROA': s('net_income').div(s('total_assets')),
        'EBITDA_Margin': s('ebitda').div(s('revenue')),
        'Debt/Equity': s('total_debt').div(s('shareholders_equity')),
        'Net_Debt': s('total_debt').fillna(0) - s('cash_and_equivalents').fillna(0),
        'NetDebt/EBITDA': (s('total_debt').fillna(0) - s('cash_and_equivalents').fillna(0)).div(s('ebitda')),
        'Interest_Coverage': s('ebit').div(s('interest_expense')),
        'Current_Ratio': s('current_assets').div(s('current_liabilities')),
        'Quick_Ratio': (s('current_assets') - s('inventories')).div(s('current_liabilities')),
        'Cash_Ratio': s('cash_and_equivalents').div(s('current_liabilities')),
        'Capex/OpCF': s('capex').div(s('operating_cf')),
        'Asset_Depreciation_Ratio': s('accum_depr').div(s('ppe_gross'))
    }

    ratios_df = pd.DataFrame(ratios).T
    ratios_df = ratios_df.reindex(columns=periods)
    return ratios_df

# ===== DCF helpers =====
def compute_dcf_from_fcf(last_fcf, fcf_growth_rates, wacc, terminal_growth):
    """
    Project FCFs given last_fcf and a list/array of growth rates for each forecast year.
    Returns DataFrame with columns: 'Year', 'FCF', 'DiscountFactor', 'PV_FCF'
    """
    years = len(fcf_growth_rates)
    fcfs = []
    current = last_fcf
    for g in fcf_growth_rates:
        next_fcf = current * (1 + g)
        fcfs.append(next_fcf)
        current = next_fcf

    df = pd.DataFrame({
        "Period": [i+1 for i in range(years)],
        "FCF": fcfs
    })
    df["DiscountFactor"] = [(1 + wacc) ** (i+1) for i in range(years)]
    df["PV_FCF"] = df["FCF"] / df["DiscountFactor"]

    # Terminal value (at end of last forecast year) using Gordon growth
    last_proj_fcf = df["FCF"].iloc[-1]
    if wacc <= terminal_growth:
        tv = np.nan
    else:
        tv = last_proj_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    df.loc["Terminal"] = ["Terminal", tv, (1 + wacc) ** years, tv / ((1 + wacc) ** years)]
    return df

def compute_dcf_revenue_based(last_revenue, revenue_growth_rates, last_fcf_margin, fcf_margin_change_rates, wacc, terminal_growth):
    """
    Project revenue, then derive FCF = Revenue * FCF_margin, where fcf_margin can change per year.
    fcf_margin_change_rates are growth rates for the margin (e.g., 0 means margin stays the same).
    """
    years = len(revenue_growth_rates)
    revenues = []
    margins = []
    fcfs = []
    current_rev = last_revenue
    current_margin = last_fcf_margin
    for i in range(years):
        g_rev = revenue_growth_rates[i]
        g_margin = fcf_margin_change_rates[i] if i < len(fcf_margin_change_rates) else 0.0
        current_rev = current_rev * (1 + g_rev)
        current_margin = current_margin * (1 + g_margin)
        revenues.append(current_rev)
        margins.append(current_margin)
        fcfs.append(current_rev * current_margin)

    df = pd.DataFrame({
        "Period": [i+1 for i in range(years)],
        "Revenue": revenues,
        "FCF_Margin": margins,
        "FCF": fcfs
    })
    df["DiscountFactor"] = [(1 + wacc) ** (i+1) for i in range(years)]
    df["PV_FCF"] = df["FCF"] / df["DiscountFactor"]

    last_proj_fcf = df["FCF"].iloc[-1]
    if wacc <= terminal_growth:
        tv = np.nan
    else:
        tv = last_proj_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    df.loc["Terminal"] = ["Terminal", np.nan, np.nan, tv, (1 + wacc) ** years, tv / ((1 + wacc) ** years)]
    return df

def dcf_summary(pv_fcfs_df, market_cap=None, net_debt=None, shares_outstanding=None):
    """
    pv_fcfs_df: DataFrame returned by compute_dcf_* with PV_FCF column and Terminal row containing PV
    Returns NPV (sum of PVs), equity value per share if shares provided, and enterprise value if net_debt is used.
    """
    # sum PV_FCF excluding terminal placeholder rows labelled non-numeric
    try:
        pv_sum = pv_fcfs_df["PV_FCF"].dropna().astype(float).sum()
    except Exception:
        pv_sum = np.nan
    # The pv for Terminal may already be included as last row's PV_FCF - ensure included
    npv = pv_sum
    # If net_debt is provided and we somehow had enterprise value handling, we could adjust.
    per_share = None
    if shares_outstanding and shares_outstanding > 0:
        per_share = npv / shares_outstanding
    return {"NPV": npv, "Per_Share": per_share}

def build_sensitivity_table(last_fcf, fcf_growth_rates, base_wacc, terminal_growth, wacc_range, term_g_range):
    """
    Build a grid of valuations by varying WACC and terminal growth.
    wacc_range: list of WACC values to evaluate
    term_g_range: list of terminal growth values
    Returns DataFrame indexed by terminal growth, columns by WACC.
    """
    rows = {}
    for tg in term_g_range:
        vals = []
        for w in wacc_range:
            df = compute_dcf_from_fcf(last_fcf, fcf_growth_rates, w, tg)
            try:
                pv = df["PV_FCF"].dropna().astype(float).sum()
            except Exception:
                pv = np.nan
            vals.append(pv)
        rows[tg] = vals
    sens = pd.DataFrame(rows, index=[f"{w:.2%}" for w in wacc_range]).T
    sens.index.name = "Terminal Growth"
    sens.columns = [f"WACC={w:.2%}" for w in wacc_range]
    return sens

# ===== Utilities for export =====
def to_excel_bytes_multi(dfs, names):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for df, name in zip(dfs, names):
            if df is None or df.empty:
                pd.DataFrame({"info": ["empty"]}).to_excel(writer, sheet_name=name, index=False)
            else:
                # avoid using object rows for Terminal which may contain mixed types—convert to string-safe table
                try:
                    df.to_excel(writer, sheet_name=name)
                except Exception:
                    # try converting all to string
                    pd.DataFrame(df.astype(str)).to_excel(writer, sheet_name=name)
    return output.getvalue()

# ===== Sidebar & data load (same as earlier app) =====
st.sidebar.header("Data input")
data_source = st.sidebar.radio("Select data source", ("Upload Excel/CSV", "Ticker (yfinance)", "Manual input"))

st.sidebar.markdown("### Missing value handling")
missing_strategy = st.sidebar.selectbox("Choose strategy", options=["none", "ffill", "bfill", "interpolate"], index=0)
min_years_for_growth = st.sidebar.number_input("Min non-NaN points for CAGR/growth", min_value=2, max_value=10, value=2, step=1)

income_df = pd.DataFrame()
bal_df = pd.DataFrame()
cf_df = pd.DataFrame()
market = {}
financials_df = pd.DataFrame()
ratios_df = pd.DataFrame()

if data_source == "Upload Excel/CSV":
    st.sidebar.markdown("Upload Excel with sheets: Income, Balance, CashFlow OR three CSV files.")
    uploaded = st.sidebar.file_uploader("Upload .xlsx / .xls / .csv", type=["xlsx", "xls", "csv"], accept_multiple_files=False)
    if uploaded:
        try:
            if uploaded.name.endswith(('.xls', '.xlsx')):
                xls = pd.ExcelFile(uploaded)
                st.sidebar.write("Detected sheets:", xls.sheet_names)
                income_sheet = st.sidebar.selectbox("Choose Income sheet", options=xls.sheet_names, index=0 if len(xls.sheet_names)>0 else 0)
                bal_sheet = st.sidebar.selectbox("Choose Balance sheet", options=xls.sheet_names, index=1 if len(xls.sheet_names)>1 else 0)
                cf_sheet = st.sidebar.selectbox("Choose CashFlow sheet", options=xls.sheet_names, index=2 if len(xls.sheet_names)>2 else 0)
                income_df = pd.read_excel(xls, sheet_name=income_sheet, index_col=0)
                bal_df = pd.read_excel(xls, sheet_name=bal_sheet, index_col=0)
                cf_df = pd.read_excel(xls, sheet_name=cf_sheet, index_col=0)
            else:
                df = pd.read_csv(uploaded, index_col=0)
                st.sidebar.write("Single CSV uploaded — using for Income. For best results, upload Excel with three sheets.")
                income_df = df
        except Exception as e:
            st.sidebar.error(f"Failed to read file: {e}")

elif data_source == "Ticker (yfinance)":
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="")
    if ticker:
        with st.spinner("Fetching data from yfinance..."):
            md = fetch_market_data(ticker)
            if md.get("error"):
                st.sidebar.error("Failed to fetch ticker: " + md["error"])
            else:
                market = md
                st.sidebar.write("Price:", market.get('price'))
                st.sidebar.write("Market cap:", market.get('market_cap'))
                if md.get('financials_df') is not None and md.get('balance_df') is not None and md.get('cashflow_df') is not None:
                    try:
                        income_df = md['financials_df']
                        bal_df = md['balance_df']
                        cf_df = md['cashflow_df']
                    except Exception:
                        income_df, bal_df, cf_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                else:
                    st.sidebar.warning("yfinance did not return full financial tables. Consider uploading Excel or inputting manually.")

elif data_source == "Manual input":
    st.sidebar.markdown("Enter aggregated values for a single period (one column).")
    manual = {}
    manual['revenue'] = st.sidebar.number_input("Revenue", value=0.0, step=1.0, format="%.2f")
    manual['net_income'] = st.sidebar.number_input("Net Income", value=0.0, step=1.0, format="%.2f")
    manual['ebitda'] = st.sidebar.number_input("EBITDA", value=0.0, step=1.0, format="%.2f")
    manual['operating_cf'] = st.sidebar.number_input("Operating Cash Flow", value=0.0, step=1.0, format="%.2f")
    manual['capex'] = st.sidebar.number_input("CAPEX", value=0.0, step=1.0, format="%.2f")
    manual['total_assets'] = st.sidebar.number_input("Total Assets", value=0.0, step=1.0, format="%.2f")
    manual['shareholders_equity'] = st.sidebar.number_input("Shareholders' equity", value=0.0, step=1.0, format="%.2f")
    manual['total_debt'] = st.sidebar.number_input("Total Debt", value=0.0, step=1.0, format="%.2f")
    manual['cash_and_equivalents'] = st.sidebar.number_input("Cash & equivalents", value=0.0, step=1.0, format="%.2f")
    manual['shares_outstanding'] = st.sidebar.number_input("Shares outstanding", value=0.0, step=1.0, format="%.0f")

    periods = ["Manual"]
    income_df = pd.DataFrame({
        "Manual": [manual['revenue'], manual['ebitda'], manual['net_income']]
    }, index=["Revenue", "EBITDA", "Net Income"])
    bal_df = pd.DataFrame({
        "Manual": [manual['total_assets'], manual['cash_and_equivalents'], manual['total_debt'], manual['shareholders_outstanding'] if 'shareholders_outstanding' in manual else manual['shares_outstanding'], manual['shares_outstanding']]
    }, index=["Total Assets", "Cash And Cash Equivalents", "Total Debt", "Total Stockholder Equity", "Shares Outstanding"])
    cf_df = pd.DataFrame({
        "Manual": [manual['operating_cf'], manual['capex']]
    }, index=["Total Cash From Operating Activities", "Capital Expenditures"])

# ===== Mapping UI & Build =====
st.header("Step 1 — Data mapping & preview")
cols_present = []
for df, name in ((income_df, "Income"), (bal_df, "Balance"), (cf_df, "CashFlow")):
    if isinstance(df, pd.DataFrame) and not df.empty:
        cols_present.append((name, list(df.columns)))
        st.subheader(f"{name} sheet — preview")
        st.dataframe(df.head(8))

if not cols_present:
    st.info("No financial sheets loaded yet. Use Upload, Ticker (if available), or Manual input to proceed.")
else:
    st.markdown("### Map rows to logical metrics (map once; mapping applies across all periods)")
    inc_index = list(income_df.index) if isinstance(income_df, pd.DataFrame) and not income_df.empty else []
    bal_index = list(bal_df.index) if isinstance(bal_df, pd.DataFrame) and not bal_df.empty else []
    cf_index = list(cf_df.index) if isinstance(cf_df, pd.DataFrame) and not cf_df.empty else []

    mapping = {}
    mapping['revenue'] = st.selectbox("Revenue (Income sheet)", options=[""] + inc_index, index=0)
    mapping['net_income'] = st.selectbox("Net Income (Income sheet)", options=[""] + inc_index, index=0)
    mapping['ebitda'] = st.selectbox("EBITDA (Income sheet) — optional", options=[""] + inc_index, index=0)
    mapping['ebit'] = st.selectbox("EBIT / Operating Income (Income sheet) — optional", options=[""] + inc_index, index=0)
    mapping['interest_expense'] = st.selectbox("Interest Expense (Income sheet) — optional", options=[""] + inc_index, index=0)
    mapping['tax_expense'] = st.selectbox("Tax Expense (Income sheet) — optional", options=[""] + inc_index, index=0)

    mapping['total_assets'] = st.selectbox("Total Assets (Balance sheet)", options=[""] + bal_index, index=0)
    mapping['current_assets'] = st.selectbox("Current Assets (Balance sheet) — optional", options=[""] + bal_index, index=0)
    mapping['inventories'] = st.selectbox("Inventories (Balance sheet) — optional", options=[""] + bal_index, index=0)
    mapping['current_liabilities'] = st.selectbox("Current Liabilities (Balance sheet) — optional", options=[""] + bal_index, index=0)
    mapping['total_debt'] = st.selectbox("Total Debt (Balance sheet) — optional", options=[""] + bal_index, index=0)
    mapping['cash_and_equivalents'] = st.selectbox("Cash & Equivalents (Balance sheet) — optional", options=[""] + bal_index, index=0)
    mapping['shareholders_equity'] = st.selectbox("Shareholders' Equity (Balance sheet)", options=[""] + bal_index, index=0)
    mapping['shares_outstanding'] = st.selectbox("Shares Outstanding (Balance sheet) — optional", options=[""] + bal_index, index=0)
    mapping['ppe_gross'] = st.selectbox("PPE Gross (Balance sheet) — optional", options=[""] + bal_index, index=0)
    mapping['accum_depr'] = st.selectbox("Accumulated Depreciation (Balance sheet) — optional", options=[""] + bal_index, index=0)

    mapping['operating_cf'] = st.selectbox("Operating Cash Flow (CashFlow sheet)", options=[""] + cf_index, index=0)
    mapping['capex'] = st.selectbox("CAPEX (CashFlow sheet)", options=[""] + cf_index, index=0)
    mapping['free_cash_flow'] = st.selectbox("Free Cash Flow (CashFlow sheet) — optional", options=[""] + cf_index, index=0)

    build_btn = st.button("Build time-series financials & compute ratios")
    if build_btn:
        with st.spinner("Building financials..."):
            financials_df = build_financials_df(income_df, bal_df, cf_df, mapping)
            financials_df = apply_missing_value_strategy(financials_df, missing_strategy)
            ratios_df = compute_time_series_ratios(financials_df)
            st.success("Built financials and computed ratios.")

            st.subheader("Financials — metrics x periods")
            st.dataframe(financials_df)

            st.subheader("Computed ratios — ratios x periods")
            st.dataframe(ratios_df)

            st.subheader("Plot a ratio across periods")
            if not ratios_df.empty:
                choice = st.selectbox("Choose ratio to plot", options=ratios_df.index.tolist())
                plot_df = ratios_df.loc[choice].reset_index().rename(columns={'index': 'period', choice: 'value'})
                fig = px.line(plot_df, x='period', y='value', title=f"{choice} trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)

            # Quick growth compute (revenue)
            st.subheader("Growth & quick stats")
            try:
                rev_series = financials_df.loc['revenue'].dropna()
                if len(rev_series) >= 2:
                    start_val, end_val = float(rev_series.iloc[0]), float(rev_series.iloc[-1])
                    years = len(rev_series.index) - 1
                    cagr = (end_val / start_val) ** (1.0 / years) - 1 if (start_val > 0 and years > 0) else np.nan
                    st.write(f"Revenue CAGR ({rev_series.index[0]} → {rev_series.index[-1]}) = {cagr:.2%}" if not pd.isna(cagr) else "CAGR cannot be computed.")
                else:
                    st.write("Not enough revenue points to compute CAGR.")
            except Exception:
                st.write("Revenue CAGR: insufficient data / mapping.")

            excel_bytes = to_excel_bytes_multi([financials_df, ratios_df], ["financials", "ratios"])
            st.download_button("Download financials & ratios (Excel)", data=excel_bytes, file_name="pride_time_series.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ===== DCF module UI & logic =====
            st.markdown("---")
            st.header("Step 2 — DCF Valuation")
            st.write("Project Free Cash Flow (FCF) and discount using WACC. If shares outstanding are present the app will compute per-share intrinsic value.")

            # Determine last available values
            # Prefer explicit 'free_cash_flow' row; else compute as operating_cf - capex using latest period
            if not financials_df.empty:
                periods = financials_df.columns.tolist()
                last_period = periods[-1]
                last_fcf = None
                if 'free_cash_flow' in financials_df.index and not pd.isna(financials_df.loc['free_cash_flow', last_period]):
                    last_fcf = financials_df.loc['free_cash_flow', last_period]
                else:
                    opcf = financials_df.loc['operating_cf', last_period] if 'operating_cf' in financials_df.index else np.nan
                    capex = financials_df.loc['capex', last_period] if 'capex' in financials_df.index else np.nan
                    if not pd.isna(opcf) and not pd.isna(capex):
                        # Note capex sign conventions: if capex stored as negative, do opcf + capex
                        # We'll attempt both conventions: if capex < 0 assume it's already negative
                        if capex < 0:
                            last_fcf = opcf + capex
                        else:
                            last_fcf = opcf - capex
                last_revenue = financials_df.loc['revenue', last_period] if 'revenue' in financials_df.index else np.nan
                last_shares = None
                if 'shares_outstanding' in financials_df.index and not pd.isna(financials_df.loc['shares_outstanding', last_period]):
                    last_shares = financials_df.loc['shares_outstanding', last_period]

                st.write(f"Last period detected: **{last_period}**")
                st.write(f"Last FCF (computed): **{last_fcf:.2f}**" if not pd.isna(last_fcf) else "Last FCF: **not available**")
                st.write(f"Last Revenue: **{last_revenue:.2f}**" if not pd.isna(last_revenue) else "Last Revenue: **not available**")
                st.write(f"Shares outstanding (last period): **{int(last_shares):,}**" if last_shares and not pd.isna(last_shares) else "Shares outstanding: **not available**")
            else:
                st.warning("Financials unavailable. Build financials first to use DCF.")
                last_fcf = np.nan
                last_revenue = np.nan
                last_shares = None

            # DCF input controls
            if not pd.isna(last_fcf) or not pd.isna(last_revenue):
                projection_method = st.selectbox("Projection method", options=["Project FCF directly (simpler)", "Project Revenue then FCF margin (more detailed)"])
                forecast_years = st.slider("Forecast years", min_value=1, max_value=10, value=5)
                # WACC inputs: allow manual WACC or quick CAPM estimate
                wacc_method = st.radio("WACC input method", ("Manual WACC", "Estimate via CAPM (quick)"))
                if wacc_method == "Manual WACC":
                    wacc = st.number_input("WACC (as decimal, e.g., 0.12 for 12%)", value=0.12, step=0.005, format="%.4f")
                else:
                    rf = st.number_input("Risk-free rate (decimal)", value=0.06, step=0.005, format="%.4f")
                    beta = st.number_input("Equity beta (estimate)", value=1.0, step=0.05, format="%.2f")
                    mkt_prem = st.number_input("Equity market premium (decimal)", value=0.05, step=0.005, format="%.4f")
                    cost_of_equity = rf + beta * mkt_prem
                    st.write(f"Estimated Cost of Equity (CAPM) = {cost_of_equity:.2%}")
                    # simple WACC assumption: assume target debt/equity mix from last period; cost of debt = rf + 3%; tax rate assume 30% (configurable)
                    assumed_debt_cost = st.number_input("Assumed pre-tax cost of debt (decimal)", value=0.09, step=0.005, format="%.4f")
                    tax_rate = st.number_input("Tax rate for WACC (decimal)", value=0.30, step=0.01, format="%.2f")
                    # compute weights from balance sheet if available
                    if not financials_df.empty and 'total_debt' in financials_df.index and 'shareholders_equity' in financials_df.index:
                        last_debt = financials_df.loc['total_debt', last_period]
                        last_equity = financials_df.loc['shareholders_equity', last_period]
                        if not pd.isna(last_debt) and not pd.isna(last_equity) and (last_debt + last_equity) > 0:
                            w_d = last_debt / (last_debt + last_equity)
                            w_e = last_equity / (last_debt + last_equity)
                        else:
                            w_d, w_e = 0.25, 0.75
                    else:
                        w_d, w_e = 0.25, 0.75
                    after_tax_cost_debt = assumed_debt_cost * (1 - tax_rate)
                    wacc = w_e * cost_of_equity + w_d * after_tax_cost_debt
                    st.write(f"Estimated WACC = {wacc:.2%} (weights: Debt {w_d:.1%}, Equity {w_e:.1%})")

                terminal_growth = st.number_input("Terminal growth rate (decimal, e.g., 0.03 for 3%)", value=0.03, step=0.005, format="%.4f")
                st.write("Projection inputs (you can enter either uniform rates or per-year lists):")

                if projection_method.startswith("Project FCF"):
                    # allow uniform growth or per-year
                    uniform = st.checkbox("Use uniform FCF growth rate for all forecast years", value=True)
                    if uniform:
                        fcf_growth = st.number_input("FCF growth rate (decimal)", value=0.05, step=0.005, format="%.4f")
                        fcf_growth_rates = [fcf_growth] * forecast_years
                    else:
                        st.write("Enter growth rates for each forecast year (as decimals).")
                        cols = st.beta_columns(forecast_years)
                        fcf_growth_rates = []
                        for i in range(forecast_years):
                            with cols[i]:
                                g = st.number_input(f"Year {i+1} g", value=0.05, step=0.005, format="%.4f")
                                fcf_growth_rates.append(g)
                    base_last_fcf = float(last_fcf)
                    run_dcf = st.button("Run DCF (FCF growth method)")
                    if run_dcf:
                        # compute projections
                        proj_df = compute_dcf_from_fcf(base_last_fcf, fcf_growth_rates, wacc, terminal_growth)
                        # sum PVs
                        try:
                            pv_sum = proj_df["PV_FCF"].dropna().astype(float).sum()
                        except Exception:
                            pv_sum = np.nan
                        per_share = None
                        if last_shares and not pd.isna(last_shares) and last_shares > 0:
                            per_share = pv_sum / last_shares
                        st.subheader("DCF Results")
                        st.write(f"NPV of projected FCFs + Terminal (Enterprise-style) = **{pv_sum:,.2f}**")
                        if per_share:
                            st.write(f"Implied value per share = **{per_share:,.2f}** (using last period shares outstanding)")
                        st.dataframe(proj_df)

                        # Sensitivity table
                        st.subheader("Sensitivity: vary WACC ± range and Terminal growth ± range")
                        wacc_steps = st.slider("WACC range: center +/- (decimal)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)
                        termg_steps = st.slider("Terminal growth +/- (decimal)", min_value=0.0, max_value=0.03, value=0.01, step=0.005)
                        # build lists
                        wacc_vals = [max(0.0001, wacc - wacc_steps), wacc, wacc + wacc_steps]
                        term_vals = [max(-0.05, terminal_growth - termg_steps), terminal_growth, terminal_growth + termg_steps]
                        sens = build_sensitivity_table(base_last_fcf, fcf_growth_rates, wacc, terminal_growth, wacc_vals, term_vals)
                        st.dataframe(sens.style.format("{:,.2f}"))
                        # Plot projected FCFs and PVs
                        st.subheader("Projected FCFs vs PVs")
                        df_plot = proj_df.reset_index(drop=True).loc[:, ["Period", "FCF", "PV_FCF"]].dropna()
                        df_plot = df_plot.astype(float)
                        fig = px.bar(df_plot.melt(id_vars=["Period"], value_vars=["FCF", "PV_FCF"]), x="Period", y="value", color="variable", barmode="group", title="FCF and PV_FCF by Period")
                        st.plotly_chart(fig, use_container_width=True)
                        # Download DCF details
                        excel_bytes = to_excel_bytes_multi([proj_df, sens], ["dcf_projection", "sensitivity"])
                        st.download_button("Download DCF results (Excel)", data=excel_bytes, file_name="pride_dcf_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                else:
                    # revenue -> fcf margin projection
                    uniform_rev = st.checkbox("Use uniform revenue growth rate for all forecast years", value=True)
                    if uniform_rev:
                        rev_growth = st.number_input("Revenue growth rate (decimal)", value=0.05, step=0.005, format="%.4f")
                        revenue_growth_rates = [rev_growth] * forecast_years
                    else:
                        cols = st.beta_columns(forecast_years)
                        revenue_growth_rates = []
                        for i in range(forecast_years):
                            with cols[i]:
                                g = st.number_input(f"Year {i+1} rev g", value=0.05, step=0.005, format="%.4f")
                                revenue_growth_rates.append(g)
                    # FCF margin inputs
                    last_fcf_margin = None
                    if not pd.isna(last_revenue) and not pd.isna(last_fcf):
                        last_fcf_margin = last_fcf / last_revenue if last_revenue != 0 else np.nan
                    else:
                        last_fcf_margin = st.number_input("Base FCF margin (decimal) (used if FCF or Revenue missing)", value=0.08, step=0.01, format="%.4f")

                    st.write(f"Detected last FCF margin = {last_fcf_margin:.2%}" if not pd.isna(last_fcf_margin) else "No last margin detected; please input base margin.")

                    uniform_margin_change = st.checkbox("Keep FCF margin constant (no change)", value=True)
                    if uniform_margin_change:
                        fcf_margin_change_rates = [0.0] * forecast_years
                    else:
                        cols2 = st.beta_columns(forecast_years)
                        fcf_margin_change_rates = []
                        for i in range(forecast_years):
                            with cols2[i]:
                                mg = st.number_input(f"Year {i+1} margin change (decimal)", value=0.0, step=0.005, format="%.4f")
                                fcf_margin_change_rates.append(mg)

                    run_dcf2 = st.button("Run DCF (Revenue → FCF method)")
                    if run_dcf2:
                        base_last_rev = float(last_revenue) if not pd.isna(last_revenue) else 0.0
                        base_last_margin = float(last_fcf_margin)
                        proj_df = compute_dcf_revenue_based(base_last_rev, revenue_growth_rates, base_last_margin, fcf_margin_change_rates, wacc, terminal_growth)
                        # sum PVs (column may contain mixed types; coerce)
                        try:
                            pv_sum = pd.to_numeric(proj_df["PV_FCF"], errors='coerce').dropna().astype(float).sum()
                        except Exception:
                            pv_sum = np.nan
                        per_share = None
                        if last_shares and not pd.isna(last_shares) and last_shares > 0:
                            per_share = pv_sum / last_shares
                        st.subheader("DCF Results")
                        st.write(f"NPV of projected FCFs + Terminal (Enterprise-style) = **{pv_sum:,.2f}**")
                        if per_share:
                            st.write(f"Implied value per share = **{per_share:,.2f}**")
                        st.dataframe(proj_df)

                        # Sensitivity (same style)
                        st.subheader("Sensitivity: vary WACC ± range and Terminal growth ± range")
                        wacc_steps = st.slider("WACC range: center +/- (decimal)", min_value=0.0, max_value=0.10, value=0.02, step=0.005, key="rev_wacc_range")
                        termg_steps = st.slider("Terminal growth +/- (decimal)", min_value=0.0, max_value=0.03, value=0.01, step=0.005, key="rev_termg_range")
                        wacc_vals = [max(0.0001, wacc - wacc_steps), wacc, wacc + wacc_steps]
                        term_vals = [max(-0.05, terminal_growth - termg_steps), terminal_growth, terminal_growth + termg_steps]
                        # For sensitivity, produce FCF growth rates by deriving fcf growth implied by revenue+margins
                        # Simpler approach: use proj_df['FCF'] growth rates relative to last_fcf
                        try:
                            last_fcf_for_sens = proj_df.loc[proj_df.index == 0, 'FCF'].iloc[0] if 0 in proj_df.index else proj_df['FCF'].iloc[0]
                        except Exception:
                            last_fcf_for_sens = proj_df['FCF'].dropna().iloc[0] if len(proj_df['FCF'].dropna())>0 else 0.0
                        # build simple fcf growth vector from proj_df FCFs
                        fcf_vals = proj_df['FCF'].dropna().astype(float).tolist()
                        if len(fcf_vals) < forecast_years:
                            # pad with last growth
                            while len(fcf_vals) < forecast_years:
                                fcf_vals.append(fcf_vals[-1] if fcf_vals else 0.0)
                        # compute simple growth rates from last_fcf to each projected fcf (not ideal but practical)
                        base_last_fcf_for_sens = float(fcf_vals[0]) if len(fcf_vals) > 0 else float(last_fcf if not pd.isna(last_fcf) else 0.0)
                        # use simplistic uniform growth derived from average growth among projected fcfs
                        fcf_growth_rates_for_sens = []
                        prev = base_last_fcf_for_sens
                        for fv in fcf_vals:
                            if prev == 0:
                                g = 0.0
                            else:
                                g = (fv / prev) - 1.0
                            fcf_growth_rates_for_sens.append(g)
                            prev = fv
                        sens = build_sensitivity_table(base_last_fcf_for_sens, fcf_growth_rates_for_sens, wacc, terminal_growth, wacc_vals, term_vals)
                        st.dataframe(sens.style.format("{:,.2f}"))
                        # Plot projected FCFs
                        st.subheader("Projected Revenue, FCF, PVs")
                        plot_df = proj_df.reset_index(drop=True).loc[:, ["Period", "Revenue", "FCF", "PV_FCF"]].dropna()
                        plot_df = plot_df.astype(float)
                        fig2 = px.line(plot_df.melt(id_vars=["Period"], value_vars=["Revenue", "FCF"]), x="Period", y="value", color="variable", title="Revenue & FCF projection")
                        st.plotly_chart(fig2, use_container_width=True)
                        excel_bytes = to_excel_bytes_multi([proj_df, sens], ["dcf_projection_rev", "sensitivity_rev"])
                        st.download_button("Download DCF results (Excel)", data=excel_bytes, file_name="pride_dcf_results_rev.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            else:
                st.info("Insufficient data (FCF or Revenue) detected to run DCF. Provide at least Operating CF and CAPEX or Free Cash Flow in uploaded sheets.")
