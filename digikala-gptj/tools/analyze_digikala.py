import argparse, os
import pandas as pd
import numpy as np

HF_MODEL = "EleutherAI/gpt-j-6b"
HF_API = "https://api-inference.huggingface.co/models/" + HF_MODEL
HF_TOKEN = os.getenv("HF_API_TOKEN")

def gptj_generate(prompt: str, max_new_tokens: int = 160) -> str:
    if not HF_TOKEN:
        return "HF_API_TOKEN تنظیم نشده؛ یادداشت GPT‑J تولید نشد."
    import requests, json
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.7}}
    r = requests.post(HF_API, headers=headers, data=json.dumps(payload), timeout=60)
    if r.status_code == 200:
        try:
            return r.json()[0]["generated_text"]
        except Exception:
            return "خطا در پارس پاسخ GPT‑J."
    return f"خطا در فراخوانی GPT‑J: {r.status_code}"

def to_num(s):
    if pd.isna(s): return 0.0
    s = str(s).replace(",", "").replace("٬", "").strip()
    out = ''.join(ch for ch in s if ch.isdigit() or ch in ".-")
    return float(out) if out else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpt-scope", default="all", choices=["all","flagged"])
    ap.add_argument("--default-cogs-rate", type=float, default=None)  # درصد بهای تمام‌شده پیش‌فرض
    args = ap.parse_args()

    df = pd.read_excel(args.input)

    # تشخیص ستون‌ها
    col_type = "نوع" if "نوع" in df.columns else "عنوان"
    col_title = "عنوان تنوع" if "عنوان تنوع" in df.columns else ("عنوان" if "عنوان" in df.columns else "")
    col_dkpc = "کد تنوع" if "کد تنوع" in df.columns else None
    col_qty = "تعداد" if "تعداد" in df.columns else None
    col_credit = "بستانکار (﷼)" if "بستانکار (﷼)" in df.columns else ("بستانکار (ریال)" if "بستانکار (ریال)" in df.columns else None)
    col_debit = "بدهکار (﷼)" if "بدهکار (﷼)" in df.columns else ("بدهکار (ریال)" if "بدهکار (ریال)" in df.columns else None)
    col_final_credit = "مبلغ نهایی بستانکار (﷼)" if "مبلغ نهایی بستانکار (﷼)" in df.columns else ("مبلغ نهایی" if "مبلغ نهایی" in df.columns else col_credit)
    col_final_debit = "مبلغ نهایی بدهکار (﷼)" if "مبلغ نهایی بدهکار (﷼)" in df.columns else col_debit

    # برچسب دسته‌ها
    t = df[col_type].fillna("")
    is_sale = t.str.contains(r"^فروش$", regex=True)
    is_sale_credit = t.str.contains(r"^فروش اعتباری$", regex=True)
    is_return_sale = t.str.contains(r"^برگشت از فروش$", regex=True)
    is_return_credit = t.str.contains(r"^برگشت از فروش اعتباری$", regex=True)
    is_commission = t.str.contains(r"^کمیسیون", regex=True)
    is_fee = t.str.contains(r"^هزینه", regex=True)
    is_penalty = t.str.contains(r"^جریمه", regex=True)

    # نرمال‌سازی اعداد
    for c in [col_credit, col_debit, col_final_credit, col_final_debit, col_qty]:
        if c and c in df.columns:
            df[c] = df[c].apply(to_num)

    # محاسبات مالی
    df["Revenue"] = 0.0
    df.loc[is_sale | is_sale_credit, "Revenue"] = df[col_final_credit] if col_final_credit in df.columns else df[col_credit]

    df["Commission"] = 0.0
    df.loc[is_commission, "Commission"] = df[col_debit]

    df["Fees"] = 0.0
    df.loc[is_fee, "Fees"] = df[col_debit]

    df["Penalties"] = 0.0
    df.loc[is_penalty, "Penalties"] = df[col_debit]

    df["Returns"] = 0.0
    df.loc[is_return_sale | is_return_credit, "Returns"] = df[col_debit]

    # تعداد فروش نهایی
    total_qty_final = 0
    if col_qty and col_qty in df.columns:
        total_qty_sales = df.loc[is_sale, col_qty].sum()
        total_qty_sales_credit = df.loc[is_sale_credit, col_qty].sum()
        total_qty_returns = df.loc[is_return_sale, col_qty].sum()
        total_qty_returns_credit = df.loc[is_return_credit, col_qty].sum()
        total_qty_final = int((total_qty_sales + total_qty_sales_credit) - (total_qty_returns + total_qty_returns_credit))
# قیمت خرید (COGS)
    if "قیمت خرید" not in df.columns:
        df["قیمت خرید"] = np.nan
        if args.default_cogs_rate:
            df["قیمت خرید"] = (df["Revenue"] * (args.default_cogs_rate / 100.0)).round(2)

    # سود ردیف و حاشیه سود
    df["Row_Profit"] = (df["Revenue"] - df["Returns"]) - df["Commission"] - df["Fees"] - df["Penalties"] - df["قیمت خرید"].fillna(0.0)
    df["Profit_Margin"] = np.where(df["Revenue"] > 0, df["Row_Profit"] / df["Revenue"], np.nan)

    # مرتب‌سازی بر اساس DKPC
    if col_dkpc and col_dkpc in df.columns:
        df = df.sort_values(by=col_dkpc, ascending=True)

    # یادداشت‌های GPT‑J
    notes = []
    for _, r in df.iterrows():
        if args.gpt_scope == "flagged":
            cond = (r["Row_Profit"] < 0) or (r["Penalties"] > 0) or (r["Commission"] / (r["Revenue"]+1e-9) > 0.15)
            if not cond:
                notes.append("")
                continue
        summary = f"""تو یک تحلیل‌گر مالی هستی. داده‌های یک ردیف از صورت‌حساب دیجی‌کالا:
- عنوان: {r.get(col_title, '')}
- DKPC: {r.get(col_dkpc, '')}
- فروش خالص (Revenue): {r['Revenue']:.2f}
- برگشت‌ها (Returns): {r['Returns']:.2f}
- کمیسیون: {r['Commission']:.2f}
- هزینه‌ها (Fees): {r['Fees']:.2f}
- جریمه‌ها (Penalties): {r['Penalties']:.2f}
- قیمت خرید (COGS): {0 if pd.isna(r['قیمت خرید']) else r['قیمت خرید']:.2f}
- سود ردیف (Row_Profit): {r['Row_Profit']:.2f}
- حاشیه سود (Profit_Margin): {'' if pd.isna(r['Profit_Margin']) else f'{r['Profit_Margin']*100:.2f}%'}
لطفاً در 3-5 جمله:
1) تفسیر کوتاه سود/زیان این ردیف،
2) اشاره به هر ریسک یا ناهنجاری،
3) یک پیشنهاد عملی برای بهبود حاشیه سود."""
        notes.append(gptj_generate(summary, max_new_tokens=160))
    df_notes = pd.DataFrame({
        "DKPC": df[col_dkpc] if col_dkpc in df.columns else "",
        "GPTJ_Analysis": notes
    })

    # Summary
    df_summary = pd.DataFrame([{
        "Total_Revenue": float(df["Revenue"].sum()),
        "Total_Returns": float(df["Returns"].sum()),
        "Total_Commission": float(df["Commission"].sum()),
        "Total_Fees": float(df["Fees"].sum()),
        "Total_Penalties": float(df["Penalties"].sum()),
        "Total_COGS": float(df["قیمت خرید"].fillna(0.0).sum()),
        "Total_Profit": float(df["Row_Profit"].sum()),
        "Avg_Margin_%": float((df["Row_Profit"].sum() / df["Revenue"].sum() * 100.0) if df["Revenue"].sum() > 0 else np.nan),
        "Total_Sales_Qty_Final": int(total_qty_final),
    }])

    # خروجی اکسل
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as