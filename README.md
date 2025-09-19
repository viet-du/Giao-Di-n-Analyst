# app_improved.py
import logging
import os
import re
import sqlite3
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, filedialog, messagebox
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import unicodedata
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

# Lựa chọn thư viện mchien learning
try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional PDF export
try:
    from fpdf import FPDF

    FPDF_OK = True
except Exception:
    FPDF_OK = False

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger("TikiApp")

# App globals
APP_TITLE = "Phần mềm trực quan hóa dữ liệu + Dự đoán"
df_global = pd.DataFrame()
processed_df = pd.DataFrame()
DB_FILES = []  # Thay đổi từ DB_FILE sang DB_FILES (danh sách)
fig_global = None
current_fig = None

# Thiết lập font tiếng Việt cho matplotlib
try:
    # Thử sử dụng font Arial Unicode MS nếu có
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


# ---------------- Helpers ----------------
def sanitize_table_name(name):
    base = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r'[^0-9a-zA-Z_]', '_', base)
    if re.match(r'^\d', s):
        s = 't_' + s
    return s.lower()


def show_message(msg, title="Thông báo"):
    messagebox.showinfo(title, msg)


def show_error(msg, title="Lỗi"):
    messagebox.showerror(title, msg)


def ensure_db_files():
    global DB_FILES
    if not DB_FILES:
        base_dir = os.getcwd()
        default_db = os.path.join(base_dir, "tiki_data.db")
        if os.path.exists(default_db):
            DB_FILES = [default_db]
            try:
                listbox_dbs.delete(0, tk.END)
                for db_file in DB_FILES:
                    listbox_dbs.insert(tk.END, db_file)
            except Exception:
                pass


def extract_id(link):
    """Try to extract product id from tiki url query 'spid' or last numeric token."""
    if not isinstance(link, str):
        return None
    m = re.search(r'spid=(\d+)', link)
    if m:
        return m.group(1)
    # fallback: last numeric sequence
    m2 = re.findall(r'(\d+)', link)
    return m2[-1] if m2 else None


# ---------------- TIKI CRAWL ----------------
def build_tiki_link(p):
    product_id = p.get("id") or p.get('product_id')
    url_path = p.get("url_path") or p.get('url_key')
    if product_id and url_path:
        return f"https://tiki.vn/{url_path}?spid={product_id}"
    return p.get("url") or p.get('link') or 'N/A'


def crawl_tiki(category_id=1686, limit=50):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*"
    }
    url = f"https://tiki.vn/api/personalish/v1/blocks/listings?limit={limit}&page=1&category={category_id}"
    log.info("Requesting %s", url)
    resp = requests.get(url, headers=headers, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"API lỗi: {resp.status_code}")

    data = resp.json()
    products = data.get("data", [])
    rows = []
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for p in products:
        name = (p.get("name") or "N/A").strip()
        price = p.get("price") or 0
        original_price = p.get("original_price") or p.get("list_price") or price
        discount = p.get("discount_rate") or 0
        qty = p.get("quantity_sold", {}).get("value", 0) if isinstance(p.get("quantity_sold"), dict) else (
                p.get("qty") or 0)
        link = build_tiki_link(p)
        rows.append([name, price, original_price, discount, qty, link, today])

    df = pd.DataFrame(rows, columns=["Tên sản phẩm", "Giá", "Giá gốc", "Giảm giá (%)", "Đã bán", "Link", "Ngày crawl"])
    fname = f"tiki_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(fname, index=False, encoding='utf-8-sig')
    log.info("Saved CSV %s", fname)

    # chỉ nạp vào df_global
    df_global_update(df)
    return df

def action_crawl_dialog():
    dlg = tk.Toplevel(root)
    dlg.title("Crawl Tiki - Tham số")
    dlg.geometry("400x200")
    dlg.resizable(False, False)
    dlg.transient(root)
    dlg.grab_set()

    # Center the dialog
    dlg.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - dlg.winfo_width()) // 2
    y = root.winfo_y() + (root.winfo_height() - dlg.winfo_height()) // 2
    dlg.geometry(f"+{x}+{y}")

    tk.Label(dlg, text="Category ID:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
    ent_cat = tk.Entry(dlg, width=30)
    ent_cat.grid(row=0, column=1, padx=6, pady=6)
    ent_cat.insert(0, "8322")

    tk.Label(dlg, text="Limit (max items):").grid(row=1, column=0, padx=6, pady=6, sticky="w")
    ent_lim = tk.Entry(dlg, width=30)
    ent_lim.grid(row=1, column=1, padx=6, pady=6)
    ent_lim.insert(0, "50")

    var_append = tk.IntVar(value=1)
    chk = tk.Checkbutton(dlg, text="Append to DB (don't overwrite)", variable=var_append)
    chk.grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="w")

    button_frame = tk.Frame(dlg)
    button_frame.grid(row=3, column=0, columnspan=2, pady=10)

    def go():
        cat = ent_cat.get().strip()
        try:
            lim = int(ent_lim.get().strip() or 50)
        except Exception:
            lim = 50
        append_db = bool(var_append.get())
        dlg.destroy()

        # Hiển thị thông báo đang xử lý
        progress = tk.Toplevel(root)
        progress.title("Đang xử lý")
        progress.geometry("300x100")
        progress.transient(root)
        progress.grab_set()

        # Center the progress dialog
        progress.update_idletasks()
        x = root.winfo_x() + (root.winfo_width() - progress.winfo_width()) // 2
        y = root.winfo_y() + (root.winfo_height() - progress.winfo_height()) // 2
        progress.geometry(f"+{x}+{y}")

        tk.Label(progress, text="Đang crawl dữ liệu từ Tiki...").pack(pady=20)
        progress.update()

        def worker():
            try:
                # REMOVED: append_db parameter from crawl_tiki call
                df = crawl_tiki(category_id=int(cat), limit=lim)
                df_global_update(df)

                # ADDED: Handle database append if requested
                if append_db and DB_FILES:
                    for db_file in DB_FILES:
                        try:
                            conn = sqlite3.connect(db_file)
                            df.to_sql("tiki_products", conn, if_exists="append", index=False)
                            conn.commit()
                            conn.close()
                        except Exception as e:
                            log.error(f"Lỗi khi ghi vào CSDL {db_file}: {e}")

                progress.destroy()
                show_message(
                    f"Crawl xong: {len(df)} sản phẩm. CSV & DB đã cập nhật." if append_db else f"Crawl xong: {len(df)} sản phẩm. CSV đã lưu.")
                cap_nhat_bang()
                cap_nhat_cot()
                hien_thi_preview(df_global)
            except Exception as e:
                progress.destroy()
                log.exception("Crawl lỗi")
                show_error(str(e))

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(button_frame, text="🕷️ Bắt đầu crawl", command=go, style="Accent.TButton").pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="❌ Hủy", command=dlg.destroy).pack(side=tk.LEFT, padx=10)

# ---------------- Data management ----------------
def df_global_update(df):
    global df_global
    if df is None:
        return
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df_global is None or df_global.empty:
        df_global = df.copy()
    else:
        df_global = pd.concat([df_global, df], ignore_index=True)
        # try to coerce common columns to string for dedup
        for c in ["Link", "Ngày crawl", "Tên sản phẩm"]:
            if c in df_global.columns:
                df_global[c] = df_global[c].astype(str)
        df_global.drop_duplicates(subset=[c for c in ["Link", "Ngày crawl", "Tên sản phẩm"] if c in df_global.columns],
                                  inplace=True, ignore_index=True)

# Replace the action_import_files function with this optimized version
def action_import_files():
    files = filedialog.askopenfilenames(
        title="Chọn file dữ liệu",
        filetypes=[
            ("Tất cả hỗ trợ", "*.csv *.xlsx *.xls *.json"),
            ("CSV", "*.csv"),
            ("Excel", "*.xlsx;*.xls"),
            ("JSON", "*.json"),
            ("All files", "*.*")
        ]
    )
    if not files:
        return

    progress = tk.Toplevel(root)
    progress.title("Đang xử lý")
    progress.geometry("300x120")
    progress.transient(root)
    progress.grab_set()

    tk.Label(progress, text="Đang import dữ liệu...").pack(pady=5)
    progress_bar = ttk.Progressbar(progress, orient='horizontal', length=250, mode='determinate')
    progress_bar.pack(pady=5)
    progress_bar['maximum'] = len(files)
    progress.update()

    def import_thread():
        dfs = []
        file_names = []
        success_count = 0
        fail_count = 0

        for i, f in enumerate(files):
            try:
                df = None
                if f.lower().endswith(".csv"):
                    # Try multiple encodings with UTF-8 BOM first
                    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1258', 'iso-8859-1']

                    for enc in encodings:
                        try:
                            # Try to detect delimiter first
                            with open(f, 'r', encoding=enc) as test_file:
                                sample = test_file.read(4096)
                                dialect = csv.Sniffer().sniff(sample)

                            df = pd.read_csv(
                                f,
                                encoding=enc,
                                delimiter=dialect.delimiter,
                                dtype=str,
                                on_bad_lines='skip',
                                engine='python',
                                quoting=csv.QUOTE_MINIMAL,
                                escapechar='\\'
                            )
                            break
                        except (UnicodeDecodeError, csv.Error):
                            continue
                        except Exception as e:
                            log.error(f"Lỗi đọc CSV {f} với encoding {enc}: {e}")
                            continue

                    if df is None:
                        # Fallback: try without delimiter detection
                        for enc in encodings:
                            try:
                                df = pd.read_csv(
                                    f,
                                    encoding=enc,
                                    dtype=str,
                                    on_bad_lines='skip',
                                    engine='python',
                                    quoting=csv.QUOTE_MINIMAL,
                                    escapechar='\\'
                                )
                                break
                            except (UnicodeDecodeError, csv.Error):
                                continue

                elif f.lower().endswith((".xlsx", ".xls")):
                    try:
                        df = pd.read_excel(f, dtype=str, engine='openpyxl')
                    except Exception as e:
                        log.error(f"Lỗi đọc Excel {f}: {e}")
                        # Try with different engine
                        try:
                            df = pd.read_excel(f, dtype=str, engine='xlrd')
                        except:
                            raise

                elif f.lower().endswith(".json"):
                    try:
                        df = pd.read_json(f, dtype=str, orient="records")
                    except:
                        try:
                            df = pd.read_json(f, dtype=str, lines=True)
                        except:
                            with open(f, "r", encoding="utf-8") as jf:
                                raw = json.load(jf)
                            if isinstance(raw, dict):
                                raw = [raw]
                            df = pd.json_normalize(raw)
                else:
                    raise ValueError(f"Định dạng file {f} không được hỗ trợ.")

                if df is not None and not df.empty:
                    # Clean column names (remove BOM and extra spaces)
                    df.columns = df.columns.str.replace('\ufeff', '').str.strip()
                    df = rename_standard_columns(df)
                    dfs.append(df)
                    file_names.append(os.path.basename(f))
                    success_count += 1
                else:
                    fail_count += 1
                    log.warning(f"File {f} rỗng hoặc không thể đọc")

            except Exception as e:
                fail_count += 1
                error_msg = f"Lỗi đọc file {f}:\n{str(e)}"
                log.error(error_msg)
                root.after(0, lambda e=error_msg: show_error(e))
                continue
            finally:
                # Update progress bar
                root.after(0, lambda i=i: progress_bar.config(value=i + 1))

        # Batch_update the file listbox after all processing is done
        if file_names:
            root.after(0, lambda: [listbox_files.insert(tk.END, name) for name in file_names])

        if dfs:
            # Use concat with ignore_index for better performance
            merged = pd.concat(dfs, ignore_index=True, sort=False)

            # Optimize numeric conversion
            numeric_columns = ['Đã bán', 'Giá', 'Giá gốc', 'Giảm giá (%)']
            for col in numeric_columns:
                if col in merged.columns:
                    # Use to_numeric with errors='coerce' for better performance
                    merged[col] = pd.to_numeric(
                        merged[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True),
                        errors='coerce'
                    ).fillna(0)

            df_global_update(merged)
            root.after(0, lambda: [
                cap_nhat_cot(),
                hien_thi_preview(df_global),
                progress.destroy(),
                show_message(
                    f"Đã import {success_count} file(s) thành công, {fail_count} file(s) thất bại. Dữ liệu sẵn sàng, chưa ghi vào SQLite.")
            ])
        else:
            root.after(0, lambda: [
                progress.destroy(),
                show_error("Không thể đọc bất kỳ file nào. Vui lòng kiểm tra định dạng file.")
            ])

    # Use a separate thread for import to prevent GUI freezing
    threading.Thread(target=import_thread, daemon=True).start()

def rename_standard_columns(df):
    mmap = {}
    # various possible names
    colmap = {
        'Ngay_crawl': 'Ngày crawl', 'Ngay crawl': 'Ngày crawl', 'NgayCrawl': 'Ngày crawl',
        'Ten_san_pham': 'Tên sản phẩm', 'Ten san pham': 'Tên sản phẩm', 'Ten': 'Tên sản phẩm',
        'Gia': 'Giá', 'gia': 'Giá', 'price': 'Giá',
        'Gia_goc': 'Giá gốc', 'Gia goc': 'Giá gốc', 'original_price': 'Giá gốc',
        'Giam_gia': 'Giảm giá (%)', 'Giam gia': 'Giảm giá (%)', 'discount': 'Giảm giá (%)',
        'Da_ban': 'Đã bán', 'Da ban': 'Đã bán', 'DaBan': 'Đã bán', 'sold': 'Đã bán',
        'Link': 'Link', 'URL': 'Link', 'url': 'Link'
    }
    for k, v in colmap.items():
        if k in df.columns and v not in df.columns:
            mmap[k] = v
    if mmap:
        df = df.rename(columns=mmap)
    return df


# ---------------- Preview & Filter ----------------
def hien_thi_preview(df):
    """
    Hiển thị DataFrame vào vùng frame_preview.
    An toàn: dọn sạch widget cũ, xử lý df rỗng, tạo Treeview + thanh cuộn bằng pack.
    """
    try:
        # xóa nội dung cũ
        for w in frame_preview.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        if df is None or df.empty:
            lbl = tk.Label(frame_preview, text="Chưa có dữ liệu để hiển thị", anchor="center", bg='white')
            lbl.pack(fill=tk.BOTH, expand=True)
            return

        # Frame chứa treeview + scrollbar (dùng pack để tránh mix grid/pack trên cùng master)
        tree_frame = tk.Frame(frame_preview, bg='white')
        tree_frame.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        cols = list(df.columns)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        # đặt vị trí bằng pack
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # header & col width
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=140, minwidth=80, anchor="w")

        # show up to 200 rows
        maxr = min(200, len(df))
        for i in range(maxr):
            row = list(df.iloc[i].fillna("").astype(str))
            tree.insert("", tk.END, values=row)

        # auto resize columns (tương đối)
        def auto_resize_columns():
            for col in cols:
                try:
                    max_len = max(df[col].astype(str).str.len().max() if not df.empty else 0, len(col))
                    tree.column(col, width=min(400, max(80, int(max_len * 7))))
                except Exception:
                    tree.column(col, width=140)

        auto_resize_columns()

        # right click menu để xuất
        def on_rclick(event):
            sel = tree.selection()
            if not sel:
                return

            menu = tk.Menu(root, tearoff=0)

            def export_selected_csv():
                rows = [tree.item(i)["values"] for i in sel]
                cols_local = cols
                out_df = pd.DataFrame(rows, columns=cols_local)
                path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                                    title="Xuất dữ liệu đã chọn")
                if path:
                    out_df.to_csv(path, index=False, encoding='utf-8-sig')
                    show_message(f"Đã xuất {len(out_df)} hàng ra {path}")

            menu.add_command(label="Xuất các dòng đã chọn ra CSV", command=export_selected_csv)
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                try:
                    menu.grab_release()
                except Exception:
                    pass

        tree.bind("<Button-3>", on_rclick)

    except Exception as e:
        log.exception("hien_thi_preview error")
        # nếu lỗi UI, hiển thị label thay vì crash
        for w in frame_preview.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        lbl = tk.Label(frame_preview, text=f"Lỗi khi hiển thị preview:\n{e}", anchor="center", bg='white', fg='red',
                       justify='center')
        lbl.pack(fill=tk.BOTH, expand=True)


def filter_preview():
    global df_global
    if df_global is None or df_global.empty:
        return

    q = entry_filter.get().strip().lower()
    if not q:
        hien_thi_preview(df_global)
        return

    df = df_global.copy()
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        try:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        except Exception:
            continue

    res = df[mask]
    hien_thi_preview(res)


# ---------------- Export ----------------
def export_current_csv():
    global df_global
    if df_global is None or df_global.empty:
        show_error("Không có dữ liệu để xuất.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
        title="Xuất dữ liệu CSV"
    )
    if path:
        try:
            df_global.to_csv(path, index=False, encoding='utf-8-sig')
            show_message(f"Đã xuất CSV: {path}")
        except Exception as e:
            show_error(f"Lỗi khi xuất CSV: {str(e)}")


def export_current_excel():
    global df_global
    if df_global is None or df_global.empty:
        show_error("Không có dữ liệu để xuất.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel", "*.xlsx")],
        title="Xuất dữ liệu Excel"
    )
    if path:
        try:
            df_global.to_excel(path, index=False)
            show_message(f"Đã xuất Excel: {path}")
        except Exception as e:
            show_error(f"Lỗi khi xuất Excel: {str(e)}")

def export_report_pdf():
    if not FPDF_OK:
        show_error("Thư viện fpdf chưa được cài. Cài đặt: pip install fpdf")
        return

    global df_global, fig_global
    if df_global is None or df_global.empty:
        show_error("Không có dữ liệu để xuất báo cáo.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF", "*.pdf")],
        title="Xuất báo cáo PDF"
    )
    if not path:
        return

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, APP_TITLE, ln=1, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
        pdf.cell(0, 8, f"Số dòng dữ liệu hiện tại: {len(df_global)}", ln=1)
        pdf.ln(4)
        pdf.cell(0, 8, "Top 10 sản phẩm theo 'Đã bán' (nếu có):", ln=1)

        try:
            if 'Đã bán' in df_global.columns:
                top = df_global.sort_values("Đã bán", ascending=False).head(10)
                pdf.set_font("Arial", size=9)
                for idx, row in top.iterrows():
                    name = str(row.get('Tên sản phẩm', ''))[:40]
                    sold = str(row.get('Đã bán', ''))
                    price = str(row.get('Giá', ''))
                    line = f"{name:40} | Đã bán: {sold} | Giá: {price}"
                    pdf.multi_cell(0, 6, line)
            else:
                pdf.multi_cell(0, 6, "Không có dữ liệu 'Đã bán'.")
        except Exception:
            pdf.multi_cell(0, 6, "Lỗi khi xử lý dữ liệu 'Đã bán'.")

        pdf.output(path)
        show_message(f"Đã xuất báo cáo PDF: {path}")
    except Exception as e:
        show_error(f"Lỗi khi xuất PDF: {str(e)}")

# ---------------- SQL & Chart ----------------
def cap_nhat_bang():
    global DB_FILES
    all_tables = []

    for db_file in DB_FILES:
        if os.path.exists(db_file):
            try:
                conn = sqlite3.connect(db_file)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [f"{os.path.basename(db_file)}::{t[0]}" for t in cur.fetchall()]
                all_tables.extend(tables)
                conn.close()
            except Exception as e:
                log.exception(f"Error reading tables from {db_file}")
                continue

    combo_table['values'] = all_tables
    if all_tables:
        combo_table.set(all_tables[0])


def cap_nhat_cot_from_table(tab=None):
    cols = []
    if tab and "::" in tab:
        db_name, table_name = tab.split("::", 1)
        db_path = None
        for db_file in DB_FILES:
            if os.path.basename(db_file) == db_name:
                db_path = db_file
                break

        if db_path:
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 1', conn)
                cols = list(df.columns)
                conn.close()
            except Exception:
                cols = []
    else:
        if df_global is not None and not df_global.empty:
            cols = list(df_global.columns)

    combo_x['values'] = cols
    combo_y['values'] = cols
    if cols:
        combo_x.set(cols[0])
        if len(cols) > 1:
            combo_y.set(cols[1])

def cap_nhat_cot():
    if df_global is not None and not df_global.empty:
        cols = list(df_global.columns)
        combo_x['values'] = cols
        combo_y['values'] = cols
        if cols:
            combo_x.set(cols[0])
            if len(cols) > 1:
                combo_y.set(cols[1])

def chay_sql_va_ve():
    global df_global, DB_FILES, fig_global
    sql = text_sql.get("1.0", tk.END).strip()
    chart_type = combo_chart.get()
    x_col = combo_x.get()
    y_col = combo_y.get()
    table_name_raw = combo_table.get().strip() if combo_table.get() else ""

    # đảm bảo có DB file (tạo mặc định nếu chưa)
    ensure_db_files()

    if not DB_FILES:
        show_error("Chưa chọn CSDL nào.")
        return

    all_dfs = []  # Lưu trữ kết quả từ tất cả các CSDL

    # Xử lý tên bảng nếu có format dbname::tablename
    table_name = ""
    if table_name_raw and "::" in table_name_raw:
        db_name, table_name = table_name_raw.split("::", 1)
    elif table_name_raw:
        # sanitize: chỉ còn a-z0-9_
        table_name = re.sub(r'[^0-9a-zA-Z_]', '_', table_name_raw)
        if re.match(r'^\d', table_name):
            table_name = 't_' + table_name
        table_name = table_name.lower()

    # Nếu user nhập tên bảng mới: tạo từ df_global trong tất cả CSDL
    if table_name and not table_name_raw.startswith("::"):
        if df_global is None or df_global.empty:
            show_error("Chưa có dữ liệu trong bộ nhớ để tạo bảng mới.")
            return

        for db_file in DB_FILES:
            try:
                conn = sqlite3.connect(db_file)
                df_global.to_sql(table_name, conn, if_exists="replace", index=False)
                conn.commit()
                conn.close()
            except Exception as e:
                log.exception(f"Lỗi khi tạo bảng mới trong {db_file}")
                show_error(f"Lỗi khi tạo bảng '{table_name}' trong {db_file}: {e}")
                return

    for db_file in DB_FILES:
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()

            # Nếu có SQL thì chạy trực tiếp
            if sql:
                try:
                    df = pd.read_sql_query(sql, conn)
                except Exception as e:
                    log.exception("Lỗi khi chạy SQL")
                    show_error(f"Lỗi khi chạy SQL trên {db_file}: {e}")
                    continue
            else:
                if not (table_name and x_col and y_col):
                    show_error("Chưa chọn bảng và/hoặc cột X/Y.")
                    return

                # kiểm tra bảng tồn tại
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if not cur.fetchone():
                    show_error(f"Bảng '{table_name}' không tồn tại trong CSDL {db_file}.")
                    continue

                try:
                    # Dùng alias X, Y để đồng nhất
                    df = pd.read_sql_query(
                        f'SELECT "{x_col}" as X, "{y_col}" as Y FROM "{table_name}"', conn
                    )
                except Exception as e:
                    log.exception("Lỗi khi đọc bảng")
                    show_error(f"Lỗi khi đọc bảng '{table_name}' từ {db_file}: {e}")
                    continue

            if df is None or df.empty:
                show_error(f"Kết quả rỗng từ {db_file}. Không có dữ liệu để vẽ.")
                continue

            df['source_db'] = os.path.basename(db_file)
            all_dfs.append(df)

        except Exception as e:
            log.exception(f"Lỗi khi xử lý CSDL {db_file}")
            show_error(f"Lỗi khi xử lý CSDL {db_file}: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    if not all_dfs:
        show_error("Không có dữ liệu để vẽ biểu đồ.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Xóa biểu đồ cũ
    for widget in frame_chart.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        # Vẽ dựa trên alias X, Y
        if chart_type == 'Bar':
            ax.bar(combined_df["X"], combined_df["Y"])
        elif chart_type == 'Line':
            ax.plot(combined_df["X"], combined_df["Y"])
        elif chart_type == 'Pie':
            ax.pie(combined_df["Y"], labels=combined_df["X"], autopct='%1.1f%%')
        elif chart_type == 'Scatter':
            ax.scatter(combined_df["X"], combined_df["Y"])

        ax.set_title(f'Biểu đồ {chart_type}')
        ax.set_xlabel(x_col)  # Hiển thị label gốc
        ax.set_ylabel(y_col)
        plt.xticks(rotation=45)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_chart)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        fig_global = fig
        frame_chart.canvas_fig = fig

    except Exception as e:
        show_error(f"Lỗi khi vẽ biểu đồ: {str(e)}")


def add_db_files():
    files = filedialog.askopenfilenames(
        title="Chọn file SQLite",
        filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
    )
    if files:
        for file_path in files:
            # Chuẩn hóa đường dẫn
            file_path = os.path.normpath(file_path)
            if file_path not in DB_FILES:
                DB_FILES.append(file_path)
                listbox_dbs.insert(tk.END, file_path)


def remove_db_file():
    selected = listbox_dbs.curselection()
    if selected:
        index = selected[0]
        DB_FILES.pop(index)
        listbox_dbs.delete(index)
        cap_nhat_bang()  # Cập nhật danh sách bảng sau khi xóa CSDL


def phong_to():
    if hasattr(frame_chart, "canvas_fig"):
        fig = frame_chart.canvas_fig
        win = tk.Toplevel(root)
        win.title("Phóng to biểu đồ")
        win.geometry("1000x700")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        # Nút đóng
        ttk.Button(win, text="❌ Đóng", command=win.destroy, style="Accent.TButton").pack(pady=5)
    else:
        show_error("Chưa có biểu đồ để phóng to.")

def save_chart_image():
    global fig_global
    if not fig_global:
        show_error("Chưa có biểu đồ để lưu.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")],
        title="Lưu biểu đồ"
    )
    if path:
        try:
            fig_global.savefig(path, bbox_inches='tight', dpi=300)
            show_message(f"Đã lưu biểu đồ: {path}")
        except Exception as e:
            show_error(f"Lỗi khi lưu biểu đồ: {str(e)}")


# Thêm hàm tìm kiếm sản phẩm trong tab SQL
def search_product_sql():
    global df_global
    search_term = entry_search_sql.get().strip().lower()
    if not search_term:
        # Nếu không có từ khóa tìm kiếm, hiển thị lại dữ liệu gốc
        hien_thi_preview(df_global)
        return

    # Kiểm tra xem có bảng nào được chọn từ combobox không
    selected_table = combo_table.get().strip()
    if selected_table and "::" in selected_table:
        # Nếu có bảng được chọn, tìm kiếm trong database
        db_name, table_name = selected_table.split("::", 1)
        db_path = None
        for db_file in DB_FILES:
            if os.path.basename(db_file) == db_name:
                db_path = db_file
                break

        if db_path:
            try:
                conn = sqlite3.connect(db_path)
                # Đọc toàn bộ dữ liệu từ bảng
                df_db = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                conn.close()

                # Tìm cột chứa tên sản phẩm
                product_columns = ['Tên sản phẩm', 'ten_san_pham', 'product_name', 'name', 'title']
                product_col = None
                for col in product_columns:
                    if col in df_db.columns:
                        product_col = col
                        break

                if product_col:
                    # Lọc dữ liệu từ database
                    filtered_df = df_db[df_db[product_col].astype(str).str.lower().str.contains(search_term, na=False)]
                    hien_thi_preview(filtered_df)
                else:
                    show_error("Không tìm thấy cột tên sản phẩm trong bảng database.")
            except Exception as e:
                show_error(f"Lỗi khi đọc từ database: {str(e)}")
        else:
            show_error("Không tìm thấy database phù hợp.")
    else:
        # Nếu không có bảng được chọn, tìm trong df_global (dữ liệu từ file)
        if df_global is None or df_global.empty:
            show_error("Không có dữ liệu để tìm kiếm.")
            return

        # Tìm cột chứa tên sản phẩm
        product_columns = ['Tên sản phẩm', 'ten_san_pham', 'product_name', 'name', 'title']
        product_col = None
        for col in product_columns:
            if col in df_global.columns:
                product_col = col
                break

        if not product_col:
            show_error("Không tìm thấy cột tên sản phẩm trong dữ liệu.")
            return

        # Lọc dữ liệu
        filtered_df = df_global[df_global[product_col].astype(str).str.lower().str.contains(search_term, na=False)]
        hien_thi_preview(filtered_df)

# ---------------- ML features ----------------
def prepare_processed_from_df_global():
    global processed_df, df_global

    # Kiểm tra xem có bảng nào được chọn không
    selected_table = combo_table.get().strip()
    if not selected_table or "::" not in selected_table:
        show_error("Vui lòng chọn một bảng dữ liệu từ tab SQL & Trực quan trước khi chuẩn hóa.")
        return

    # Lấy thông tin database và bảng từ combobox
    db_name, table_name = selected_table.split("::", 1)

    # Tìm đường dẫn database
    db_path = None
    for db_file in DB_FILES:
        if os.path.basename(db_file) == db_name:
            db_path = db_file
            break

    if not db_path:
        show_error(f"Không tìm thấy database: {db_name}")
        return

    try:
        # Đọc dữ liệu từ bảng đã chọn
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        conn.close()
    except Exception as e:
        show_error(f"Lỗi khi đọc bảng {table_name}: {str(e)}")
        return

    # Tiếp tục xử lý dữ liệu như trước
    df = rename_standard_columns(df)

    # ensure date col
    if "Ngày crawl" in df.columns:
        df['Ngày crawl'] = pd.to_datetime(df['Ngày crawl'], errors='coerce')
    elif 'Ngay_crawl' in df.columns:
        df['Ngày crawl'] = pd.to_datetime(df['Ngay_crawl'], errors='coerce')
    else:
        df['Ngày crawl'] = datetime.now()

    for c in ['Đã bán', 'Giá']:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(r'[^0-9.-]', '', regex=True),
                errors='coerce'
            ).fillna(0)

    df['Tháng'] = df['Ngày crawl'].dt.strftime('%Y-%m')
    df['SanPhamID'] = df['Link'].apply(lambda x: extract_id(x) if pd.notna(x) else None)

    agg = df.groupby(['SanPhamID', 'Tên sản phẩm', 'Tháng'], dropna=False).agg({
        'Đã bán': 'max',
        'Giá': 'mean'
    }).reset_index()

    agg['DoanhSoTháng'] = agg.groupby('SanPhamID')['Đã bán'].diff().fillna(agg['Đã bán'])

    # Normalize column names
    agg.rename(columns={
        'Tên sản phẩm': 'ten_san_pham',
        'Tháng': 'thang',
        'DoanhSoTháng': 'doanhso_thang',
        'SanPhamID': 'sanpham_id'
    }, inplace=True)

    processed_df = agg

    # update listbox products
    listbox_products.delete(0, tk.END)
    for p in processed_df['ten_san_pham'].unique():
        if pd.notna(p):
            listbox_products.insert(tk.END, p)
    show_message('Đã chuẩn hóa dữ liệu từ bảng đã chọn cho tab Học máy.')


def load_ml_data_from_db():
    global processed_df, DB_FILES

    if not DB_FILES:
        show_error("Chưa chọn file database nào.")
        return

    all_data = []
    for db_file in DB_FILES:
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            for table in tables:
                try:
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    # Chuẩn hóa tên cột
                    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
                    all_data.append(df)
                except Exception as e:
                    print(f"Lỗi khi đọc bảng {table}: {e}")
            conn.close()
        except Exception as e:
            print(f"Lỗi kết nối database {db_file}: {e}")
            continue

    if not all_data:
        show_error("Không có dữ liệu trong database.")
        return

    # Gộp dữ liệu và xử lý
    df = pd.concat(all_data, ignore_index=True)
    # Chỉ giữ các cột cần thiết, đổi tên cho đồng nhất
    col_map = {
        "ten_san_pham": "Tên sản phẩm",
        "da_ban": "Đã bán",
        "ngay_crawl": "Ngày crawl",
        "gia": "Giá",
        "link": "Link"
    }
    for k, v in col_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # Loại bỏ các dòng thiếu dữ liệu quan trọng
    df = df.dropna(subset=["Tên sản phẩm", "Đã bán", "Ngày crawl"])
    # Chuẩn hóa kiểu dữ liệu
    df["Đã bán"] = pd.to_numeric(df["Đã bán"], errors="coerce").fillna(0)
    df["Ngày crawl"] = pd.to_datetime(df["Ngày crawl"], errors="coerce")
    df = df[df["Ngày crawl"].notna()]
    # Tính tháng
    df["Tháng"] = df["Ngày crawl"].dt.strftime("%Y-%m")
    # Gộp theo sản phẩm và tháng
    agg = df.groupby(["Tên sản phẩm", "Tháng"]).agg({"Đã bán": "max"}).reset_index()
    agg["Doanh số tháng"] = agg.groupby("Tên sản phẩm")["Đã bán"].diff().fillna(agg["Đã bán"])

    # Normalize column names
    agg.rename(columns={
        'Tên sản phẩm': 'ten_san_pham',
        'Tháng': 'thang',
        'Doanh số tháng': 'doanhso_thang'
    }, inplace=True)

    processed_df = agg

    # Cập nhật listbox sản phẩm
    listbox_products.delete(0, tk.END)
    for sp in processed_df["ten_san_pham"].unique():
        listbox_products.insert(tk.END, sp)
    show_message(f"Đã tải {len(processed_df)} bản ghi từ database.")


def process_data():
    try:
        df = df_global.copy()
        df = rename_standard_columns(df)

        # ensure date col
        if "Ngày crawl" in df.columns:
            df['Ngày crawl'] = pd.to_datetime(df['Ngày crawl'], errors='coerce')
        elif 'Ngay_crawl' in df.columns:
            df['Ngày crawl'] = pd.to_datetime(df['Ngay_crawl'], errors='coerce')
        else:
            # Nếu không có cột ngày, thêm cột với ngày hiện tại
            df['Ngày crawl'] = datetime.now()

        # coerce numeric
        for c in ['Đã bán', 'Giá']:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(r'[^0-9.-]', '', regex=True),
                    errors='coerce'
                ).fillna(0)

        df['Tháng'] = df['Ngày crawl'].dt.strftime('%Y-%m')
        df['SanPhamID'] = df['Link'].apply(lambda x: extract_id(x) if pd.notna(x) else None)

        agg = df.groupby(['SanPhamID', 'Tên sản phẩm', 'Tháng'], dropna=False).agg({
            'Đã bán': 'max',
            'Giá': 'mean'
        }).reset_index()

        agg['DoanhSoTháng'] = agg.groupby('SanPhamID')['Đã bán'].diff().fillna(agg['Đã bán'])

        # Normalize column names
        agg.rename(columns={
            'Tên sản phẩm': 'ten_san_pham',
            'Tháng': 'thang',
            'DoanhSoTháng': 'doanhso_thang',
            'SanPhamID': 'sanpham_id'
        }, inplace=True)

        processed_df = agg

        # update listbox products
        root.after(0, lambda: [
            listbox_products.delete(0, tk.END),
            [listbox_products.insert(tk.END, p) for p in processed_df['ten_san_pham'].unique() if pd.notna(p)],
            show_message('Đã chuẩn hóa dữ liệu cho tab Học máy.')
        ])

    except Exception as e:
        root.after(0, lambda: [
            show_error(f"Lỗi khi chuẩn hóa dữ liệu: {str(e)}")
        ])

    threading.Thread(target=process_data, daemon=True).start()


def predict_model():
    global processed_df
    if processed_df is None or processed_df.empty:
        show_error("Chưa có dữ liệu xử lý. Nhấn 'Chuẩn hóa dữ liệu' trước.")
        return

    sel = listbox_products.curselection()
    if not sel:
        show_error("Chọn 1 sản phẩm để dự đoán.")
        return

    product = listbox_products.get(sel[0])
    df = processed_df[processed_df['ten_san_pham'] == product].sort_values('thang')

    if df.empty or len(df) < 2:
        show_error("Không đủ dữ liệu để dự đoán (ít nhất 2 tháng).")
        return

    method = combo_ml_method.get()

    if method == 'Linear Regression':
        if not SKLEARN_OK:
            show_error('scikit-learn chưa cài. Cài đặt: pip install scikit-learn')
            return

        X = np.arange(len(df)).reshape(-1, 1)
        y = df['doanhso_thang'].values
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict([[len(df)]])[0]
        score = model.score(X, y)

    elif method == 'Moving Average':
        try:
            window = int(entry_ma_window.get() or 3)
        except Exception:
            window = 3

        y = df['doanhso_thang'].values
        if len(y) >= window:
            pred = float(pd.Series(y).rolling(window=window).mean().iloc[-1])
        else:
            pred = float(y.mean() if len(y) > 0 else 0)
        score = float('nan')

    else:
        show_error('Chọn phương pháp dự đoán.')
        return

    last = df['doanhso_thang'].iloc[-1]
    change = (pred - last) / last * 100 if last != 0 else float('inf')

    text_result.delete('1.0', tk.END)
    result_text = f"Sản phẩm: {product}\nPhương pháp: {method}\nDự đoán tháng kế: {pred:.2f}\nTháng trước: {last}\n"

    if pd.notna(change) and change != float('inf'):
        result_text += f"Thay đổi: {change:.2f}%\n"
    else:
        result_text += "Thay đổi: không xác định\n"

    result_text += f"Mô hình R²: {score:.4f}" if not pd.isna(score) else "Mô hình R²: n/a"

    text_result.insert(tk.END, result_text)

    # Xóa biểu đồ cũ
    for w in frame_chart_ml.winfo_children():
        w.destroy()

    # Tạo biểu đồ mới
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['thang'], df['doanhso_thang'], marker='o', label='Thực tế')
    ax.plot([df['thang'].iloc[-1], f"{df['thang'].iloc[-1]}+1"], [last, pred], 'r--o', label='Dự đoán')
    ax.set_title(product)
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Doanh số tháng')
    ax.legend()
    plt.xticks(rotation=45)

    # Hiển thị biểu đồ trong GUI
    canvas = FigureCanvasTkAgg(fig, master=frame_chart_ml)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    # Lưu trữ tham chiếu
    frame_chart_ml.canvas_fig = fig
    frame_chart_ml.canvas_obj = canvas


# ---------------- Extra analysis ----------------
def _show_figure_in_toplevel(fig, title='Figure', size=(800, 600)):
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry(f"{size[0]}x{size[1]}")

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    # Nút đóng
    ttk.Button(win, text="❌ Đóng", command=win.destroy, style="Accent.TButton").pack(pady=5)


def analyze_top_products():
    global df_global
    if df_global is None or df_global.empty:
        show_error('Không có dữ liệu để phân tích.')
        return

    df = df_global.copy()
    if 'Đã bán' not in df.columns:
        show_error("Không có cột 'Đã bán' trong dữ liệu.")
        return

    df['Đã bán'] = pd.to_numeric(
        df['Đã bán'].astype(str).str.replace(r'[^0-9.-]', '', regex=True),
        errors='coerce'
    ).fillna(0)

    top = df.groupby('Tên sản phẩm', dropna=False)['Đã bán'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index.astype(str), top.values)
    ax.invert_yaxis()
    ax.set_xlabel('Tổng đã bán')
    ax.set_title('Top 10 sản phẩm bán chạy nhất')

    # Thêm giá trị trên mỗi cột
    for i, v in enumerate(top.values):
        ax.text(v + max(top.values) * 0.01, i, str(int(v)), ha='left', va='center')

    plt.tight_layout()
    _show_figure_in_toplevel(fig, title='Top 10 sản phẩm', size=(900, 600))


def compare_by_date():
    global df_global
    if df_global is None or df_global.empty:
        show_error('Không có dữ liệu để so sánh.')
        return

    if 'Ngày crawl' not in df_global.columns:
        show_error('Không có cột "Ngày crawl" trong dữ liệu.')
        return

    unique_dates = sorted(df_global['Ngày crawl'].astype(str).unique())
    if len(unique_dates) < 2:
        show_error('Cần ít nhất 2 ngày crawl để so sánh.')
        return

    # Tạo dialog để chọn ngày
    dlg = tk.Toplevel(root)
    dlg.title("Chọn ngày để so sánh")
    dlg.geometry("400x300")
    dlg.resizable(False, False)
    dlg.transient(root)
    dlg.grab_set()

    # Center the dialog
    dlg.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - dlg.winfo_width()) // 2
    y = root.winfo_y() + (root.winfo_height() - dlg.winfo_height()) // 2
    dlg.geometry(f"+{x}+{y}")

    tk.Label(dlg, text="Chọn ngày thứ nhất:").pack(pady=5)
    date1_var = tk.StringVar(value=unique_dates[-1] if unique_dates else "")
    date1_combo = ttk.Combobox(dlg, textvariable=date1_var, values=unique_dates, state="readonly")
    date1_combo.pack(pady=5)

    tk.Label(dlg, text="Chọn ngày thứ hai:").pack(pady=5)
    date2_var = tk.StringVar(value=unique_dates[-2] if len(unique_dates) > 1 else "")
    date2_combo = ttk.Combobox(dlg, textvariable=date2_var, values=unique_dates, state="readonly")
    date2_combo.pack(pady=5)

    def do_compare():
        d1 = date1_var.get()
        d2 = date2_var.get()
        dlg.destroy()

        if not d1 or not d2:
            show_error('Vui lòng chọn cả hai ngày.')
            return

        # Sửa lỗi chính tả: astize -> astype
        df1 = df_global[df_global['Ngày crawl'].astype(str).str.contains(d1)]
        df2 = df_global[df_global['Ngày crawl'].astype(str).str.contains(d2)]

        if df1.empty or df2.empty:
            show_error('Không tìm thấy dữ liệu cho một trong hai ngày.')
            return

        agg1 = df1.groupby('Tên sản phẩm')['Đã bán'].sum().sort_values(ascending=False).head(10)
        agg2 = df2.groupby('Tên sản phẩm')['Đã bán'].sum().sort_values(ascending=False).head(10)

        names = list(set(agg1.index).union(set(agg2.index)))
        y1 = [agg1.get(n, 0) for n in names]
        y2 = [agg2.get(n, 0) for n in names]

        x = range(len(names))
        fig, ax = plt.subplots(figsize=(10, 6))
        w = 0.35
        ax.bar([i - w / 2 for i in x], y1, width=w, label=f"{d1}")
        ax.bar([i + w / 2 for i in x], y2, width=w, label=f"{d2}")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Đã bán')
        ax.set_title(f'So sánh {d1} vs {d2}')
        ax.legend()

        plt.tight_layout()
        _show_figure_in_toplevel(fig, title=f'So sánh {d1} vs {d2}', size=(1000, 600))

    ttk.Button(dlg, text="📊 So sánh", command=do_compare, style="Accent.TButton").pack(pady=20)
    ttk.Button(dlg, text="❌ Hủy", command=dlg.destroy).pack(pady=5)


# ---------------- New ML functions ----------------
def search_product_ml(*args):
    global processed_df
    if processed_df is None:
        return
    query = entry_search_ml.get().lower()
    listbox_products.delete(0, tk.END)

    # Use the correct column name based on how data was processed
    product_col = "ten_san_pham" if "ten_san_pham" in processed_df.columns else "Tên sản phẩm"

    for sp in processed_df[product_col].unique():
        if query in str(sp).lower():
            listbox_products.insert(tk.END, sp)

def predict_next_month():
    global processed_df
    if processed_df is None or processed_df.empty:
        show_error("Chưa có dữ liệu xử lý. Nhấn 'Chuẩn hóa dữ liệu' trước.")
        return

    sel = listbox_products.curselection()
    if not sel:
        show_error("Chọn 1 sản phẩm để dự đoán.")
        return

    # Validate month input
    month_input = entry_predict_month.get().strip()
    if not month_input or not re.match(r'^\d{2}/\d{4}$', month_input):
        show_error("Vui lòng nhập tháng dự đoán theo định dạng MM/YYYY (ví dụ: 12/2023)")
        return

    try:
        input_month, input_year = map(int, month_input.split('/'))
        if input_month < 1 or input_month > 12:
            show_error("Tháng phải nằm trong khoảng từ 01 đến 12")
            return
    except ValueError:
        show_error("Định dạng tháng không hợp lệ. Vui lòng nhập theo định dạng MM/YYYY")
        return

    # Disable predict button during processing
    btn_predict.config(state=tk.DISABLED)

    # Show progress window
    progress = tk.Toplevel(root)
    progress.title("Đang xử lý")
    progress.geometry("300x100")
    progress.transient(root)
    progress.grab_set()

    # Center the progress dialog
    progress.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - progress.winfo_width()) // 2
    y = root.winfo_y() + (root.winfo_height() - progress.winfo_height()) // 2
    progress.geometry(f"+{x}+{y}")

    tk.Label(progress, text="Đang xử lý dự đoán...").pack(pady=20)
    progress.update()

    def prediction_worker():
        try:
            product = listbox_products.get(sel[0])
            df = processed_df[processed_df['ten_san_pham'] == product].sort_values('thang')

            if df.empty or len(df) < 2:
                root.after(0, lambda: show_error("Không đủ dữ liệu để dự đoán (ít nhất 2 tháng)."))
                return

            method = combo_ml_method.get()

            if method == 'Linear Regression':
                if not SKLEARN_OK:
                    root.after(0, lambda: show_error('scikit-learn chưa cài. Cài đặt: pip install scikit-learn'))
                    return

                # Convert thang to datetime and calculate months since start
                df['thang_dt'] = pd.to_datetime(df['thang'])
                start_date = df['thang_dt'].min()

                # Calculate months since start for each data point
                df['months_since_start'] = ((df['thang_dt'].dt.year - start_date.year) * 12 +
                                            (df['thang_dt'].dt.month - start_date.month))

                # Calculate months since start for prediction point
                target_date = datetime(input_year, input_month, 1)
                months_to_predict = ((target_date.year - start_date.year) * 12 +
                                     (target_date.month - start_date.month))

                # Prepare features for prediction
                X = df['months_since_start'].values.reshape(-1, 1)
                y = df['doanhso_thang'].values

                model = LinearRegression()
                model.fit(X, y)

                # Predict for the specified month and year
                pred = model.predict([[months_to_predict]])[0]
                score = model.score(X, y)

            elif method == 'Moving Average':
                try:
                    window = int(entry_ma_window.get() or 3)
                except Exception:
                    window = 3

                # Use all available data for moving average
                y = df['doanhso_thang'].values
                if len(y) >= window:
                    pred = float(pd.Series(y).rolling(window=window).mean().iloc[-1])
                else:
                    pred = float(y.mean() if len(y) > 0 else 0)
                score = float('nan')

            else:
                root.after(0, lambda: show_error('Chọn phương pháp dự đoán.'))
                return

            # Get the last actual value for comparison
            last = df['doanhso_thang'].iloc[-1]
            change = (pred - last) / last * 100 if last != 0 else float('inf')

            result_text = f"Sản phẩm: {product}\n"
            result_text += f"Phương pháp: {method}\n"
            result_text += f"Tháng dự đoán: {month_input}\n"
            result_text += f"Dự đoán: {pred:.2f}\n"
            result_text += f"Tháng trước: {last}\n"

            if pd.notna(change) and change != float('inf'):
                result_text += f"Thay đổi: {change:+.2f}%\n"
            else:
                result_text += "Thay đổi: không xác định\n"

            result_text += f"Mô hình R²: {score:.4f}" if not pd.isna(score) else "Mô hình R²: n/a"

            # Update UI in main thread
            root.after(0, lambda: [
                text_result.delete('1.0', tk.END),
                text_result.insert(tk.END, result_text),
                update_prediction_chart(df, product, last, pred, month_input, months_to_predict),
                progress.destroy(),
                btn_predict.config(state=tk.NORMAL)
            ])

        except Exception as e:
            root.after(0, lambda: [
                progress.destroy(),
                btn_predict.config(state=tk.NORMAL),
                show_error(f"Lỗi khi dự đoán: {str(e)}")
            ])

    # Run prediction in separate thread
    threading.Thread(target=prediction_worker, daemon=True).start()

def update_prediction_chart(df, product, last, pred, target_month, months_to_predict):
    """Update the prediction chart with new data"""
    # Clear old chart
    for w in frame_chart_ml.winfo_children():
        w.destroy()

    # Create new chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert thang to datetime for plotting
    df['thang_dt'] = pd.to_datetime(df['thang'])

    # Plot historical data
    ax.plot(df['thang_dt'], df['doanhso_thang'], marker='o', label='Thực tế', linewidth=2)

    # Calculate position for prediction point
    start_date = df['thang_dt'].min()
    prediction_date = start_date + pd.DateOffset(months=months_to_predict)

    # Add prediction point
    ax.plot(prediction_date, pred, 'ro', markersize=8, label='Dự đoán')

    # Add a line connecting the last data point to the prediction
    last_date = df['thang_dt'].iloc[-1]
    ax.plot([last_date, prediction_date], [last, pred], 'r--', alpha=0.7, linewidth=2)

    # Format x-axis to handle potentially distant dates
    ax.set_xlim(df['thang_dt'].min() - pd.DateOffset(months=1),
                prediction_date + pd.DateOffset(months=1))

    # Format dates on x-axis for better readability
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=max(1, len(df) // 6)))

    ax.set_title(f'{product} - Dự đoán tháng {target_month}', fontsize=14)
    ax.set_xlabel('Tháng', fontsize=12)
    ax.set_ylabel('Doanh số tháng', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display chart in GUI
    canvas = FigureCanvasTkAgg(fig, master=frame_chart_ml)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    # Store references
    frame_chart_ml.canvas_fig = fig
    frame_chart_ml.canvas_obj = canvas
    current_fig = fig

def load_csvs():
    global processed_df
    files = filedialog.askopenfilenames(title="Chọn các file CSV", filetypes=[("CSV Files", "*.csv")])
    if not files:
        return
    # Preprocess files function needs to be implemented
    # For now, we'll use the existing import function
    action_import_files()

def load_from_db():
    global processed_df, DB_FILES
    conn = None

    try:
        if not DB_FILES or not all(os.path.exists(db_file) for db_file in DB_FILES):
            show_error("Chưa chọn file CSDL")
            return

        all_data = []

        for db_file in DB_FILES:
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()

                # Lấy danh sách tất cả các bảng
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [table[0] for table in cursor.fetchall()]

                for table in tables:
                    try:
                        # Kiểm tra xem bảng có các cột cần thiết không
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [col[1] for col in cursor.fetchall()]

                        # Use normalized column names for comparison
                        normalized_columns = [_normalize_colname(col) for col in columns]
                        required_columns = ["ten_san_pham", "da_ban", "ngay_crawl", "link"]
                        has_required = all(col in normalized_columns for col in required_columns)

                        if has_required:
                            # Lấy dữ liệu từ bảng
                            df_table = pd.read_sql_query(f"SELECT * FROM {table}", conn)

                            # Normalize column names in the dataframe
                            col_map = {col: _normalize_colname(col) for col in df_table.columns}
                            df_table.rename(columns=col_map, inplace=True)

                            all_data.append(df_table)

                    except Exception as e:
                        print(f"Lỗi khi đọc bảng {table}: {e}")
                        continue

            except Exception as e:
                show_error(f"Lỗi khi kết nối CSDL {db_file}: {str(e)}")
                continue
            finally:
                if conn:
                    conn.close()

        if not all_data:
            show_error("Không tìm thấy bảng nào có dữ liệu phù hợp")
            return

        # Gộp tất cả dữ liệu từ các bảng
        df_combined = pd.concat(all_data, ignore_index=True)

        # Xử lý dữ liệu - thay đổi từ tuần sang tháng
        df_combined["ngay_crawl"] = pd.to_datetime(df_combined["ngay_crawl"], errors="coerce")
        df_combined = df_combined.dropna(subset=["ngay_crawl"])
        df_combined["thang"] = df_combined["ngay_crawl"].dt.strftime("%Y-%m")  # Định dạng năm-tháng
        df_combined["sanpham_id"] = df_combined["link"].apply(extract_id)

        # Nhóm dữ liệu theo tháng
        agg = (
            df_combined.groupby(["sanpham_id", "ten_san_pham", "thang"])
            .agg({
                "da_ban": "max",
                "gia": "mean"
            })
            .reset_index()
        )

        agg["doanhso_thang"] = agg.groupby("sanpham_id")["da_ban"].diff().fillna(agg["da_ban"])

        processed_df = agg

        # Cập nhật list sản phẩm
        listbox_products.delete(0, tk.END)
        for sp in processed_df["ten_san_pham"].unique():
            listbox_products.insert(tk.END, sp)

        show_message(f"Đã tải dữ liệu từ {len(DB_FILES)} CSDL thành công")

    except Exception as e:
        show_error(f"Lỗi khi tải dữ liệu từ CSDL: {str(e)}")

def load_data_ml():
    # Hỏi người dùng muốn tải từ CSV hay từ CSDL
    choice = messagebox.askquestion("Chọn nguồn dữ liệu",
                                    "Bạn muốn tải dữ liệu từ file CSV hay từ CSDL hiện tại?",
                                    icon='question',
                                    type='yesno',
                                    default='yes',
                                    detail='Chọn Yes để tải từ CSV, No để tải từ CSDL')

    if choice == 'yes':
        load_csvs()
    else:
        load_from_db()

# Add these helper functions for date processing
def _normalize_colname(col):
    """Normalize column names to lowercase with underscores"""
    if not isinstance(col, str):
        col = str(col)
    col = col.lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    return col


def _normalize_vietnamese_text(text):
    """Normalize Vietnamese text by removing diacritics"""
    if not isinstance(text, str):
        return text
    # Normalize unicode characters and remove diacritics
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.lower()


def _parse_date_flexible(date_str):
    """Try to parse date from multiple formats"""
    if not isinstance(date_str, str):
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def extract_date_from_url(url):
    """Try to extract date from URL if present"""
    if not isinstance(url, str):
        return None

    # Look for date patterns in URL
    patterns = [
        r'/(\d{4})/(\d{2})/(\d{2})/',
        r'(\d{4})-(\d{2})-(\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return datetime(year, month, day)
            except ValueError:
                continue
    return None

# Thêm nút làm mới dữ liệu
def refresh_data():
    cap_nhat_bang()
    cap_nhat_cot()
    if df_global is not None and not df_global.empty:
        hien_thi_preview(df_global)
    show_message("Đã làm mới dữ liệu")


# ---------------- New functions for file management ----------------
def clear_imported_files():
    global df_global
    if not listbox_files.size():
        show_message("Không có file nào để xóa")
        return

    if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa tất cả file đã import?"):
        listbox_files.delete(0, tk.END)
        df_global = pd.DataFrame()
        show_message("Đã xóa tất cả file import và làm trống bộ nhớ")


def delete_selected_file():
    selected = listbox_files.curselection()
    if not selected:
        show_error("Vui lòng chọn một file để xóa")
        return

    if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa file đã chọn?"):
        listbox_files.delete(selected)
        show_message("Đã xóa file đã chọn khỏi danh sách")


def clear_sql():
    text_sql.delete('1.0', tk.END)


def update_product_list(*args):
    search_term = entry_search_ml.get().lower()
    listbox_products.delete(0, tk.END)
    if processed_df is not None and not processed_df.empty and 'ten_san_pham' in processed_df.columns:
        products = processed_df['ten_san_pham'].unique()
        for p in products:
            if pd.notna(p) and search_term in str(p).lower():
                listbox_products.insert(tk.END, p)


# ---------------- New function to handle table selection ----------------
def on_table_selected(event):
    """Xử lý sự kiện khi chọn bảng từ combobox"""
    selected_table = combo_table.get()
    if not selected_table or "::" not in selected_table:
        return

    # Lấy tên database và tên bảng
    db_name, table_name = selected_table.split("::", 1)

    # Tìm đường dẫn database
    db_path = None
    for db_file in DB_FILES:
        if os.path.basename(db_file) == db_name:
            db_path = db_file
            break

    if not db_path:
        show_error(f"Không tìm thấy database: {db_name}")
        return

    try:
        # Kết nối database và đọc dữ liệu
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        conn.close()

        # Hiển thị dữ liệu trong phần xem trước
        hien_thi_preview(df)

        # Cập nhật danh sách sản phẩm trong tab Học máy
        listbox_products.delete(0, tk.END)

        # Tìm cột chứa tên sản phẩm (có thể có các tên khác nhau)
        product_col = None
        possible_names = ['Tên sản phẩm', 'ten_san_pham', 'product_name', 'name', 'title']

        for col in possible_names:
            if col in df.columns:
                product_col = col
                break

        if product_col:
            products = df[product_col].dropna().unique()
            for p in products:
                listbox_products.insert(tk.END, p)
        else:
            show_message("Không tìm thấy cột tên sản phẩm trong bảng này")

    except Exception as e:
        show_error(f"Lỗi khi đọc bảng: {str(e)}")


# --Giao diện--
root = tk.Tk()
root.title(APP_TITLE)
root.geometry('1400x900')
root.configure(bg='#f5f5f5')

# Tạo style cho giao diện
style = ttk.Style()
style.theme_use('clam')

# Cấu hình màu sắc chung
BG_COLOR = '#ffffff'  # Màu nền trắng
FG_COLOR = '#2d2d2d'  # Màu chữ tối
ACCENT_COLOR = '#2b579a'  # Màu xanh
HOVER_COLOR = '#f3f3f3'  # Màu nền khi hover
RED_COLOR = '#dc3545'  # Màu đỏ cho nút xóa
RED_HOVER = '#c82333'  # Màu đỏ khi hover

# Cấu hình style
style.configure('TFrame', background=BG_COLOR)
style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR, font=('Segoe UI', 9))
style.configure('TNotebook', background=BG_COLOR)
style.configure('TNotebook.Tab', font=('Segoe UI', 9, 'bold'), padding=[10, 5])

# Style cho các nút
style.configure('TButton',
                font=('Segoe UI', 9, 'bold'),
                borderwidth=1,
                focusthickness=0,
                padding=(14, 8),
                anchor="center")
style.map('TButton',
          background=[('active', HOVER_COLOR), ('pressed', '#e1e1e1')],
          relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

# Style cho nút chính (nổi bật hơn)
style.configure('Accent.TButton',
                background=ACCENT_COLOR,
                foreground='white')
style.map('Accent.TButton',
          background=[('active', '#3a6bb0'), ('pressed', '#1e4a7e')])

# Style cho nút xóa
style.configure('Danger.TButton',
                background=RED_COLOR,
                foreground='white')
style.map('Danger.TButton',
          background=[('active', RED_HOVER), ('pressed', '#a71e2d')])

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# --- Tab SQL & Chart ---
tab_sql = ttk.Frame(notebook)
notebook.add(tab_sql, text='📊 SQL & Trực quan')

# top controls
top_frame = tk.Frame(tab_sql, bg=BG_COLOR)
top_frame.pack(fill=tk.X, padx=10, pady=10)

tk.Label(top_frame, text='CSDL:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)

# Thay thế entry_db bằng listbox_dbs và các nút
frame_dbs = tk.Frame(top_frame, bg=BG_COLOR)
frame_dbs.pack(side=tk.LEFT, padx=6)

listbox_dbs = tk.Listbox(frame_dbs, width=60, height=3, bg='white', fg=FG_COLOR, font=('Segoe UI', 9), relief='solid',
                         bd=1)
listbox_dbs.pack(side=tk.LEFT, fill=tk.X, expand=True)

scrollbar_dbs = ttk.Scrollbar(frame_dbs, orient=tk.VERTICAL, command=listbox_dbs.yview)
scrollbar_dbs.pack(side=tk.RIGHT, fill=tk.Y)
listbox_dbs.config(yscrollcommand=scrollbar_dbs.set)

btn_add_db = ttk.Button(top_frame, text='➕ Thêm CSDL', command=add_db_files)
btn_add_db.pack(side=tk.LEFT, padx=2)

btn_remove_db = ttk.Button(top_frame, text='➖ Xóa CSDL', command=remove_db_file, style="Danger.TButton")
btn_remove_db.pack(side=tk.LEFT, padx=2)

ttk.Button(top_frame, text='🕷️ Crawl Tiki', command=action_crawl_dialog, style="Accent.TButton").pack(side=tk.LEFT,
                                                                                                      padx=2)
ttk.Button(top_frame, text='📁 Import files', command=action_import_files).pack(side=tk.LEFT, padx=2)
ttk.Button(top_frame, text='📊 Export CSV', command=export_current_csv).pack(side=tk.LEFT, padx=2)
ttk.Button(top_frame, text='📊 Export Excel', command=export_current_excel).pack(side=tk.LEFT, padx=2)
ttk.Button(top_frame, text='📄 Export PDF', command=export_report_pdf).pack(side=tk.LEFT, padx=2)
ttk.Button(top_frame, text='🔄 Làm mới', command=refresh_data).pack(side=tk.RIGHT, padx=2)

# SQL area
frame_sql = tk.Frame(tab_sql, bg=BG_COLOR)
frame_sql.pack(fill=tk.X, padx=10, pady=10)

tk.Label(frame_sql, text='SQL Query:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)
text_sql = tk.Text(frame_sql, height=4, width=70, bg='white', fg=FG_COLOR,
                   insertbackground=FG_COLOR, wrap=tk.WORD, font=('Consolas', 10),
                   relief='solid', bd=1)
text_sql.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

ttk.Button(frame_sql, text='❌ Xóa', command=clear_sql, style="Danger.TButton").pack(side=tk.LEFT, padx=6)

frame_ctrl = tk.Frame(tab_sql, bg=BG_COLOR)
frame_ctrl.pack(fill=tk.X, padx=10, pady=10)

tk.Label(frame_ctrl, text='Bảng:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)
combo_table = ttk.Combobox(frame_ctrl, width=25, postcommand=cap_nhat_bang, font=('Segoe UI', 9))
combo_table.pack(side=tk.LEFT, padx=6)
# Thêm sự kiện khi chọn bảng
combo_table.bind('<<ComboboxSelected>>', on_table_selected)

tk.Label(frame_ctrl, text='Cột X:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)
combo_x = ttk.Combobox(frame_ctrl, width=20, font=('Segoe UI', 9))
combo_x.pack(side=tk.LEFT, padx=6)

tk.Label(frame_ctrl, text='Cột Y:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)
combo_y = ttk.Combobox(frame_ctrl, width=20, font=('Segoe UI', 9))
combo_y.pack(side=tk.LEFT, padx=6)

tk.Label(frame_ctrl, text='Loại biểu đồ:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)
combo_chart = ttk.Combobox(frame_ctrl, values=['Bar', 'Line', 'Pie', 'Scatter'], width=12, font=('Segoe UI', 9))
combo_chart.set('Bar')
combo_chart.pack(side=tk.LEFT, padx=6)

ttk.Button(frame_ctrl, text='📊 Chạy & Vẽ', command=chay_sql_va_ve, style="Accent.TButton").pack(side=tk.LEFT, padx=6)
ttk.Button(frame_ctrl, text='💾 Lưu ảnh', command=save_chart_image).pack(side=tk.LEFT, padx=6)
ttk.Button(frame_ctrl, text='📈 Top 10 SP', command=analyze_top_products, style="Accent.TButton").pack(side=tk.LEFT, padx=6)
ttk.Button(frame_ctrl, text='📅 So sánh ngày', command=compare_by_date).pack(side=tk.LEFT, padx=6)
ttk.Button(frame_ctrl, text='🔍 Phóng to', command=phong_to).pack(side=tk.LEFT, padx=6)

# file list + preview
listbox_frame = tk.Frame(tab_sql, bg=BG_COLOR)
listbox_frame.pack(fill=tk.X, padx=10, pady=5)

# Tạo frame chứa tiêu đề và nút xóa files nằm ngang hàng
file_header_frame = tk.Frame(listbox_frame, bg=BG_COLOR)
file_header_frame.pack(fill=tk.X)

# Label tiêu đề bám trái
tk.Label(
    file_header_frame, text='📂 Các file đã import:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 10, 'bold')
).pack(side=tk.LEFT, anchor="w", padx=(2, 0))

# Frame chứa listbox và các nút
file_list_frame = tk.Frame(listbox_frame, bg=BG_COLOR)
file_list_frame.pack(fill=tk.X, pady=5)

# Listbox với selectmode EXTENDED để chọn nhiều file

listbox_files = tk.Listbox(

    file_list_frame, height=4, bg='white', fg=FG_COLOR, font=('Segoe UI', 9), relief='solid', bd=1,
    selectmode=tk.EXTENDED
)
listbox_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

# Frame chứa các nút bên phải
button_frame = tk.Frame(file_list_frame, bg=BG_COLOR)
button_frame.pack(side=tk.RIGHT, fill=tk.Y)

# Nút xóa file đã chọn
btn_delete_selected = ttk.Button(
    button_frame, text='🗑️ Xóa file đã chọn', command=delete_selected_file, style="Danger.TButton"
)
btn_delete_selected.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

# Nút xóa tất cả files
btn_clear = ttk.Button(
    button_frame, text='❌ Xóa tất cả', command=clear_imported_files, style="Danger.TButton"
)
btn_clear.pack(side=tk.TOP, fill=tk.X)

# Frame preview với label
preview_frame = tk.Frame(tab_sql, bg=BG_COLOR)
preview_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

# Tạo frame chứa cho tiêu đề và thanh tìm kiếm
header_frame = tk.Frame(preview_frame, bg=BG_COLOR)
header_frame.pack(fill=tk.X, pady=(0, 5))

# Nhãn "Xem trước dữ liệu" bên trái
tk.Label(header_frame, text='Xem trước dữ liệu:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)

# Frame tìm kiếm bên phải
search_frame = tk.Frame(header_frame, bg=BG_COLOR)
search_frame.pack(side=tk.LEFT)

# Các thành phần tìm kiếm
tk.Label(search_frame, text='Tìm sản phẩm:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)
entry_search_sql = ttk.Entry(search_frame, width=20, font=('Segoe UI', 9))
entry_search_sql.pack(side=tk.LEFT, padx=6)
ttk.Button(search_frame, text='🔍 Tìm', command=search_product_sql).pack(side=tk.LEFT, padx=2)

# Khung xem trước dữ liệu
frame_preview = tk.Frame(preview_frame, relief='solid', borderwidth=1, height=220, bg='white')
frame_preview.pack(fill=tk.BOTH, pady=5, expand=True)

# Frame chart với label
chart_frame = tk.Frame(tab_sql, bg=BG_COLOR)
chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(chart_frame, text='Biểu đồ:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(anchor='w')
frame_chart = tk.Frame(chart_frame, bg='white', height=360, relief='solid', bd=1)
frame_chart.pack(fill=tk.BOTH, pady=5, expand=True)

# bottom filter in SQL tab
bottom_filter = tk.Frame(tab_sql, bg=BG_COLOR)
bottom_filter.pack(fill=tk.X, padx=10, pady=10)

tk.Label(bottom_filter, text='Lọc dữ liệu:', bg=BG_COLOR, fg=FG_COLOR, font=('Segoe UI', 9)).pack(side=tk.LEFT)
entry_filter = ttk.Entry(bottom_filter, width=30, font=('Segoe UI', 9))
entry_filter.pack(side=tk.LEFT, padx=6)
ttk.Button(bottom_filter, text='🔍 Áp dụng bộ lọc', command=filter_preview).pack(side=tk.LEFT, padx=6)
ttk.Button(bottom_filter, text='❌ Xóa bộ lọc',
           command=lambda: [entry_filter.delete(0, tk.END), hien_thi_preview(df_global)],
           style="Danger.TButton").pack(side=tk.LEFT, padx=6)

# =========== TAB HỌC MÁY ===========
tab_ml = ttk.Frame(notebook)
notebook.add(tab_ml, text="🤖 Học máy")

# Frame bên trái
frame_left = ttk.Frame(tab_ml, width=300)
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

# Tiêu đề
ttk.Label(frame_left, text="DỰ ĐOÁN DOANH SỐ", font=("Arial", 12, "bold")).pack(pady=10)

# Ô tìm kiếm
ttk.Label(frame_left, text="Tìm kiếm sản phẩm:").pack(anchor=tk.W, pady=(10, 5))
entry_search_ml = ttk.Entry(frame_left)
entry_search_ml.pack(fill=tk.X, padx=5, pady=5)
entry_search_ml.bind("<KeyRelease>", search_product_ml)

# Danh sách sản phẩm
ttk.Label(frame_left, text="Danh sách sản phẩm:").pack(anchor=tk.W, pady=(10, 5))
listbox_products = tk.Listbox(frame_left, height=15)
listbox_products.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Frame cho tháng dự đoán
frame_predict_month = ttk.Frame(frame_left)
frame_predict_month.pack(fill=tk.X, pady=5)

ttk.Label(frame_predict_month, text="Tháng dự đoán (MM/YYYY):").pack(anchor=tk.W)
entry_predict_month = ttk.Entry(frame_predict_month)
entry_predict_month.pack(fill=tk.X, pady=2)
entry_predict_month.insert(0, datetime.now().strftime("%m/%Y"))

# Thêm các thành phần cho phương pháp dự đoán
frame_methods = ttk.Frame(frame_left)
frame_methods.pack(fill=tk.X, pady=5)

ttk.Label(frame_methods, text="Phương pháp:").pack(anchor=tk.W)
combo_ml_method = ttk.Combobox(frame_methods, values=["Linear Regression", "Moving Average"], state="readonly")
combo_ml_method.set("Linear Regression")
combo_ml_method.pack(fill=tk.X, pady=2)

ttk.Label(frame_methods, text="Cửa sổ MA:").pack(anchor=tk.W)
entry_ma_window = ttk.Entry(frame_methods)
entry_ma_window.insert(0, "3")
entry_ma_window.pack(fill=tk.X, pady=2)

# Nút chuẩn hóa dữ liệu
btn_preprocess = ttk.Button(frame_left, text="🔄 Chuẩn hóa dữ liệu", command=prepare_processed_from_df_global)
btn_preprocess.pack(fill=tk.X, pady=5)

# Nút dự đoán
btn_predict = ttk.Button(frame_left, text="🔮 Dự đoán", command=predict_next_month)
btn_predict.pack(fill=tk.X, pady=5)

# Frame bên phải
frame_right = ttk.Frame(tab_ml)
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Kết quả dự đoán
ttk.Label(frame_right, text="KẾT QUẢ DỰ ĐOÁN", font=("Arial", 12, "bold")).pack(pady=10)
text_result = tk.Text(frame_right, height=8, wrap=tk.WORD)
text_result.pack(fill=tk.X, padx=5, pady=5)

# Biểu đồ
ttk.Label(frame_right, text="BIỂU ĐỒ DOANH SỐ", font=("Arial", 12, "bold")).pack(pady=(20, 5))
frame_chart_ml = ttk.Frame(frame_right, height=300)
frame_chart_ml.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Thêm hướng dẫn sử dụng
instruction_text = """
HƯỚNG DẪN SỬ DỤNG:
1. Chọn bảng dữ liệu từ tab SQL & Trực quan
2. Nhấn 'Chuẩn hóa dữ liệu' để xử lý dữ liệu
3. Tìm kiếm và chọn sản phẩm cần dự đoán
4. Chọn phương pháp dự đoán và nhấn 'Dự đoán'
"""
instruction_label = ttk.Label(frame_left, text=instruction_text, justify=tk.LEFT)
instruction_label.pack(fill=tk.X, padx=5, pady=10)

# initialize
ensure_db_files()
cap_nhat_bang()
cap_nhat_cot_from_table()

# Chạy ứng dụng
root.mainloop()
