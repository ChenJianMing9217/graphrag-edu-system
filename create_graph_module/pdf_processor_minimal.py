#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_processor.py 的精簡版本
只包含 process_pdf_to_graph.py 會用到的部分：
- PDFProcessor 類別（extract_tables_structured 方法）
- merge_table_rows_by_id 函數
- build_grouped_table_units 函數
"""

import hashlib
import re
from collections import defaultdict, Counter
from typing import List, Dict, Optional

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("[警告] pdfplumber 未安裝，PDF處理功能將無法使用。請執行: pip install pdfplumber")


class PDFProcessor:
    """PDF 處理器（精簡版，只包含 extract_tables_structured 方法）"""
    
    def __init__(self):
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber 未安裝，無法處理PDF檔案")

    def sha1_16(self, s: str) -> str:
        """產生 16 位元 SHA1 雜湊值"""
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

    def extract_tables_structured(self, page, doc_id: str, page_no: int) -> List[Dict]:
        """
        抽取當頁所有表格列，並拆成列資料：
        {
            "doc_id": ...,
            "page_no": ...,
            "table_id": ...,        # 依表頭雜湊
            "row_index": 0-based,   # 在該表中的列序
            "header": [...],        # 欄名列表（原始表頭）
            "row_dict": {欄名: 值}
        }
        """
        TABLE_SETTINGS = dict(
            vertical_strategy="lines",
            horizontal_strategy="lines",
            intersection_tolerance=5,
            edge_min_length=12,
        )
        out: List[Dict] = []
        try:
            tables = page.extract_tables(TABLE_SETTINGS)
        except Exception:
            tables = None
        if not tables:
            return out
        for t in tables:
            if not t or len(t) < 2:
                continue
            header = t[0]
            # 標準化表頭作為 table_id 來源
            header_clean = [(h or "").strip() for h in header]

            # 排除「分數小表」（例如：原始分數／標準分數／百分等級／發展年齡等）
            score_header_keywords = ["原始分數", "標準分數", "百分等級", "發展年齡", "FSIQ", "T分數", "IQ"]
            non_empty_headers = [h for h in header_clean if h]
            if non_empty_headers:
                # 若所有非空表頭都只是在描述分數欄位，視為分數小表，整張略過
                if all(any(kw in h for kw in score_header_keywords) for h in non_empty_headers):
                    continue

            header_norm = " | ".join(header_clean)
            table_id = self.sha1_16(header_norm) if header_norm else "NO_HEADER"

            for row_index, r in enumerate(t[1:]):
                rd: Dict[str, str] = {}
                for h, v in zip(header, r):
                    h = (h or "").strip()
                    v = (v or "").strip()
                    if not h and not v:
                        continue
                    if h:
                        rd[h] = v
                if rd:
                    out.append({
                        "doc_id": doc_id,
                        "page_no": page_no,
                        "table_id": table_id,
                        "row_index": row_index,
                        "header": header_clean,
                        "row_dict": rd,
                    })
        return out


def merge_table_rows_by_id(table_rows: List[Dict]) -> List[Dict]:
    """
    依 doc_id + table_id 將表格列合併成「一張表」的結構，方便跨頁表格重建。
    回傳格式：
    {
        "doc_id": ...,
        "table_id": ...,
        "start_page": ...,
        "end_page": ...,
        "header": [...],
        "rows": [
            {"page_no": ..., "row_index": ..., "row_dict": {...}},
            ...
        ]
    }
    """
    tables = defaultdict(list)
    for row in table_rows:
        key = (row.get("doc_id"), row.get("table_id", "NO_ID"))
        tables[key].append(row)

    def normalize_category(row_dict: Dict[str, str]) -> Optional[str]:
        """
        主要根據「類別」欄位，粗略對應到幾個標準類別：
        - 知覺動作功能
        - 吞嚥/口腔功能
        - 口語溝通功能
        - 認知功能
        - 社會情緒功能
        """
        if not row_dict:
            return None
        raw = row_dict.get("類別")
        if not raw:
            return None
        # 只針對「類別」欄位做正規化，避免被內文中的「語言／溝通／口腔」干擾
        s = re.sub(r"\s+", "", str(raw))  # 去除換行與多餘空白
        if not s:
            return None
        # 1) 吞嚥 / 口腔 相關（優先於一般「動作」關鍵字）
        if any(k in s for k in ["吞嚥", "吞嚥功能", "吞嚥反射", "吞嚥/口腔", "吞嚥/口", "口腔", "口腔功能", "口腔動作", "腔功能"]):
            return "吞嚥/口腔功能"
        # 2) 口語 / 溝通 相關
        if any(k in s for k in ["口語", "語言", "溝通"]):
            return "口語溝通功能"
        # 3) 認知相關
        if "認知" in s:
            return "認知功能"
        # 4) 社會情緒 / 情緒行為 與社會適應
        if any(k in s for k in ["社會情緒", "情緒行為", "社會適應"]):
            return "社會情緒功能"
        # 5) 知覺動作 / 粗大 / 精細 等都歸為「知覺動作功能」
        if any(k in s for k in ["知覺動作", "粗大動作", "精細動作", "知覺", "動作"]):
            return "知覺動作功能"
        return None

    merged: List[Dict] = []
    for (doc_id, table_id), row_list in tables.items():
        if not row_list:
            continue
        # 依頁碼與列序排序，確保跨頁時順序正確
        row_list.sort(key=lambda r: (r.get("page_no", 0), r.get("row_index", 0)))
        header = row_list[0].get("header", [])
        # 向下帶入同欄位上一筆非空內容，但「整列都空」的排版列會被略過，不輸出
        if header:
            last_vals: Dict[str, Optional[str]] = {col: None for col in header}
            cleaned_rows: List[Dict] = []
            for r in row_list:
                rd = r.get("row_dict", {})
                # 先判斷這一列是否完全沒有實際內容（所有欄位皆為空白/空字串）
                has_non_empty = False
                for col in header:
                    val = rd.get(col, "")
                    if isinstance(val, str):
                        if val.strip():
                            has_non_empty = True
                            break
                    else:
                        if val not in (None, ""):
                            has_non_empty = True
                            break
                # 若整列皆空：當作純排版列，直接略過，不做 forward-fill，也不輸出
                if not has_non_empty:
                    continue

                # 對有內容的列做逐欄 forward-fill
                for col in header:
                    val = rd.get(col, "")
                    if isinstance(val, str):
                        if val.strip():
                            last_vals[col] = val
                        else:
                            if last_vals.get(col) is not None:
                                rd[col] = last_vals[col]
                    else:
                        if val not in (None, ""):
                            last_vals[col] = val
                cleaned_rows.append(r)
            # 只保留處理過且非全空的列
            row_list = cleaned_rows

        # 1-b) 在單一表內，嘗試將被直書與跨列切開的「類別」片段合併成標準類別
        STANDARD_CATEGORIES = [
            "知覺動作功能",
            "吞嚥/口腔功能",
            "口語溝通功能",
            "認知功能",
            "社會情緒功能",
        ]
        n_rows = len(row_list)
        # 預先計算每列清洗後的類別字串
        cat_clean_list: List[Optional[str]] = []
        for r in row_list:
            rd = r.get("row_dict", {})
            raw_cat = rd.get("類別", "")
            if raw_cat:
                s = re.sub(r"\s+", "", str(raw_cat))
                cat_clean_list.append(s if s else None)
            else:
                cat_clean_list.append(None)
        # 逐列嘗試合併相鄰兩列的類別片段
        idx = 0
        while idx < n_rows:
            cur = cat_clean_list[idx]
            # 若本列已經是完整標準類別，直接套用標準字串
            if cur in STANDARD_CATEGORIES:
                row_list[idx]["row_dict"]["類別"] = cur
                idx += 1
                continue
            # 嘗試與下一列組合成完整標準類別（處理像「口語溝」+「通功能」→「口語溝通功能」的情況）
            if cur and idx + 1 < n_rows:
                nxt = cat_clean_list[idx + 1]
                if nxt:
                    combined = cur + nxt
                    if combined in STANDARD_CATEGORIES:
                        row_list[idx]["row_dict"]["類別"] = combined
                        row_list[idx + 1]["row_dict"]["類別"] = combined
                        cat_clean_list[idx] = combined
                        cat_clean_list[idx + 1] = combined
                        idx += 2
                        continue
            idx += 1

        pages = [r.get("page_no", 0) for r in row_list if r.get("page_no") is not None]
        start_page = min(pages) if pages else None
        end_page = max(pages) if pages else None

        # 2) 為每一列建立標準化類別與 group（同一個「類別 + 評估／訓練項目」視為一組）
        table_rows_with_norm: List[Dict] = []
        table_level_cats: List[str] = []
        groups_by_cat_item: Dict[tuple, List[Dict]] = defaultdict(list)

        for r in row_list:
            rd = r.get("row_dict", {})
            # 2-1) 類別標準化（僅根據「類別」欄位）
            norm_cat = normalize_category(rd)

            # 2-2) 建立列物件
            row_obj = {
                "page_no": r.get("page_no"),
                "row_index": r.get("row_index"),
                "row_dict": rd,
            }

            if norm_cat:
                row_obj["normalized_category"] = norm_cat
                # 也在 row_dict 中增加一個欄位方便直接使用
                rd["normalized_category"] = norm_cat
                table_level_cats.append(norm_cat)

            # 2-3) 依「類別 + 評估／訓練項目」建立群組 key（去除空白與換行以避免直書切斷問題）
            raw_cat = rd.get("類別", "") or ""
            raw_item = rd.get("評估／訓練項目", "") or ""
            key_cat = re.sub(r"\s+", "", str(raw_cat))
            key_item = re.sub(r"\s+", "", str(raw_item))
            group_key = (key_cat, key_item)
            groups_by_cat_item[group_key].append(row_obj)

            table_rows_with_norm.append(row_obj)

        # 2-4) 為每一組 (類別, 評估／訓練項目) 指派 group_id，方便後續「同一功能區塊」整合
        for idx, ((key_cat, key_item), rows_in_group) in enumerate(groups_by_cat_item.items(), start=1):
            # 僅在兩者皆非空時建立 group_id；否則視為未分組的雜項列
            if not key_cat and not key_item:
                continue
            group_id = f"{table_id}-g{idx:03d}"
            for row_obj in rows_in_group:
                row_obj["group_id"] = group_id
                row_obj["row_dict"].setdefault("group_id", group_id)

        # 3) 若整張表內只有一種 normalized_category，才寫到表層；
        #    若有多種或皆無，則表層 normalized_category 保持 None，避免誤判。
        table_cat: Optional[str] = None
        if table_level_cats:
            cnt = Counter(table_level_cats)
            if len(cnt) == 1:
                table_cat = next(iter(cnt.keys()))

        merged.append({
            "doc_id": doc_id,
            "table_id": table_id,
            "start_page": start_page,
            "end_page": end_page,
            "header": header,
            "normalized_category": table_cat,
            "rows": table_rows_with_norm,
        })
    return merged


def build_grouped_table_units(merged_tables: List[Dict]) -> List[Dict]:
    """
    以「類別 + 評估／訓練項目」為單位，將同一張表中屬於同一功能區塊的多列合併為一筆。

    回傳結構範例：
    {
        "doc_id": ...,
        "table_id": ...,
        "group_id": ...,
        "category": ...,
        "item": ...,
        "normalized_category": ...,
        "page_start": ...,
        "page_end": ...,
        "text": "...合併後的『評估工具、結果與訓練方向』全文...",
        "rows": [ {...原始列...}, ... ]  # 保留原始列以便除錯或進階使用
    }
    """
    units: List[Dict] = []
    for tbl in merged_tables:
        doc_id = tbl.get("doc_id")
        table_id = tbl.get("table_id")
        rows = tbl.get("rows", [])
        if not rows:
            continue
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in rows:
            gid = r.get("group_id")
            if not gid:
                continue
            groups[gid].append(r)
        for gid, rs in groups.items():
            if not rs:
                continue
            # 依頁碼與列序排序，保持原始閱讀順序
            rs_sorted = sorted(rs, key=lambda x: (x.get("page_no", 0), x.get("row_index", 0)))
            first_rd = (rs_sorted[0].get("row_dict") or {})
            category = first_rd.get("類別")
            item = first_rd.get("評估／訓練項目")
            norm_cat = first_rd.get("normalized_category")
            pages = [r.get("page_no", 0) for r in rs_sorted if r.get("page_no") is not None]
            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None
            # 合併「評估工具、結果與訓練方向」欄位文字，中間以空行分隔
            texts: List[str] = []
            for r in rs_sorted:
                rd = r.get("row_dict") or {}
                txt = rd.get("評估工具、結果與訓練方向", "")
                if txt:
                    texts.append(txt)
            merged_text = "\n\n".join(texts)
            units.append({
                "doc_id": doc_id,
                "table_id": table_id,
                "group_id": gid,
                "category": category,
                "item": item,
                "normalized_category": norm_cat,
                "page_start": page_start,
                "page_end": page_end,
                "text": merged_text,
                "rows": rs_sorted,
            })
    return units
