#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_grouped_tables_to_neo4j.py 的精簡版本
只包含 process_pdf_to_graph.py 會用到的部分：
- ensure_constraints 函數
- upsert_unit_with_subdomain 函數
以及其依賴的輔助函數
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
import re

from neo4j import GraphDatabase


def infer_domain(
    norm_cat: Optional[str],
    raw_category: Optional[str],
    item: Optional[str] = None,
) -> Optional[str]:
    """
    根據 normalized_category（若有）、category（表格中的「類別」欄位）
    以及 item（評估／訓練項目）推論 Domain。

    - 若 normalized_category 已存在，直接使用它。
    - 否則，針對一些特殊表頭（主訴與就診問題、團隊評估總結、疾病診斷、評估結果符號說明、
      病因分類、相關疾病）歸為『綜合評估』這個新的 Domain。
    - 若上述皆無，則嘗試從 item 中推論（例如「粗動作訓練」「表達訓練」）。
    """
    if norm_cat:
        return norm_cat
    # 先從 category 判斷特殊的「綜合評估」區塊（主訴、團隊總結、疾病診斷、符號說明、病因分類、相關疾病）
    if raw_category:
        s = "".join(str(raw_category).split())  # 去除換行與空白
        if any(
            k in s
            for k in [
                "主訴與就診問題",
                "團隊評估總結",
                "疾病診斷",
                "評估結 果符號說明",
                "評估結果符號說明",
                "病因分類",
                "相關疾病",
            ]
        ):
            return "綜合評估"

    # 再從 item（評估／訓練項目）推論 Domain，例如「粗動作訓練」「表達訓練」
    if item:
        si = "".join(str(item).split())
        if "粗動作訓練" in si:
            return "知覺動作功能"
        # 口語相關：表達訓練 / 口語表達 發展遲緩 等，都歸到口語溝通功能
        if "表達訓練" in si or "口語表達" in si:
            return "口語溝通功能"

    return None


def split_unit_text_sections(text: str) -> Dict[str, str]:
    """
    將 FunctionUnit 的 text 粗略切成幾個區塊：
    - assessment_date:   評估日期：...
    - assessment_result: 評估結果：...(含百分位等)
    - assessment_tools:  評估工具： 或「評估工具、結果與訓練方向」以下
    - observation:       行為觀察及綜合結果： / 行為觀察：
    - training:          訓練方向：
    - suggestion:        具體建議： / 居家練習與環境建議： 之後

    規則為行為式狀態機，遇到標題切換 section，否則延續當前 section。
    """
    sections = {
        "assessment_date": [],
        "assessment_result": [],
        "assessment_tools": [],
        "observation": [],
        "training": [],
        "suggestion": [],
    }
    current: Optional[str] = None
    if not text:
        return {k: "" for k in sections.keys()}

    for line in text.splitlines():
        stripped = line.strip()
        # 新 section 判斷
        if stripped.startswith("評估日期："):
            current = "assessment_date"
            sections[current].append(line)
            continue
        if stripped.startswith("評估結果："):
            current = "assessment_result"
            sections[current].append(line)
            continue
        if stripped.startswith("評估工具：") or stripped.startswith("評估工具、結果與訓練方向"):
            current = "assessment_tools"
            sections[current].append(line)
            continue
        if "行為觀察及綜合結果：" in stripped or stripped.startswith("行為觀察："):
            current = "observation"
            sections[current].append(line)
            continue
        if stripped.startswith("訓練方向：") or stripped == "訓練方向":
            current = "training"
            sections[current].append(line)
            continue
        if stripped.startswith("具體建議：") or stripped.startswith("居家練習與環境建議："):
            current = "suggestion"
            sections[current].append(line)
            continue

        # 非標題行：若已有當前 section，就附加進去
        if current:
            sections[current].append(line)

    # 合併回字串
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def split_section_items(text: str) -> List[Dict[str, Any]]:
    """
    將單一 section 的文字依「編號」拆成多個 item：
    - 支援樣式：
      1. xxx
      1) xxx / 1）xxx / （1）xxx
    - 回傳每個 item: {index, level, text}
      其中 level 粗略代表層級（1 = 主項，2 = 子項）。
    """
    items: List[Dict[str, Any]] = []
    if not text:
        return items

    current: Optional[Dict[str, Any]] = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                current["lines"].append(line)
            continue

        # 若是純標題行（例如「具體建議：」「訓練方向：」「行為觀察及綜合結果：」），
        # 不應該被拆成獨立的 item，直接略過，避免出現只有四個字的無意義節點。
        if re.match(
            r"^(評估結果|行為觀察及綜合結果|行為觀察|訓練方向|具體建議|居家練習與環境建議)[：:]\s*$",
            stripped,
        ):
            continue

        # 主項：1. 1、 1。 等（不包含 1)）
        m_main = re.match(r"^(\d+)[\.\u3002、]\s*", stripped)
        # 子項：1) 1） (1) （1） 等
        m_sub = re.match(r"^[\(（]?(\d+)[\)）]\s*", stripped)

        new_index = None
        level = 1
        cut_pos = 0

        if m_main:
            new_index = int(m_main.group(1))
            level = 1
            cut_pos = m_main.end()
        elif m_sub:
            new_index = int(m_sub.group(1))
            level = 2
            cut_pos = m_sub.end()

        if new_index is not None:
            # 關閉前一個 item
            if current:
                items.append(
                    {
                        "index": current["index"],
                        "level": current["level"],
                        "text": "\n".join(current["lines"]).strip(),
                    }
                )
            # 開啟新 item
            content = stripped[cut_pos:].lstrip()
            # 保留原行（包含編號）以避免遺失資訊
            current = {
                "index": new_index,
                "level": level,
                "lines": [line] if content else [line],
            }
        else:
            if current is None:
                # 沒有編號就出現的內容，視為 index=0 的 item
                current = {"index": 0, "level": 1, "lines": [line]}
            else:
                current["lines"].append(line)

    if current:
        items.append(
            {
                "index": current["index"],
                "level": current["level"],
                "text": "\n".join(current["lines"]).strip(),
            }
        )

    # 過濾掉空白 item，並補上一個「序號 seq」避免僅用 index 導致 1. / 1) 衝突
    cleaned: List[Dict[str, Any]] = [it for it in items if it["text"]]
    for seq, it in enumerate(cleaned, start=1):
        it["seq"] = seq

    # 依出現順序設定 parent_seq：
    # - level=1 的主項 parent_seq=None，並成為往後子項的 parent 候選
    # - level=2 的子項 parent_seq=最近一個上方主項的 seq（若沒有主項則為 None）
    last_parent_seq: Optional[int] = None
    for it in cleaned:
        if it["level"] == 1:
            it["parent_seq"] = None
            last_parent_seq = it["seq"]
        else:
            it["parent_seq"] = last_parent_seq

    return cleaned


def generate_domain_id(doc_id: str, domain_name: str) -> str:
    """
    生成 domain_id：使用 sha1(doc_id + "|Domain|" + domain_name) 的十六進位字串。
    確保不同報告的同名 Domain 不會共用節點。
    """
    content = f"{doc_id}|Domain|{domain_name}"
    return hashlib.sha1(content.encode('utf-8')).hexdigest()


def generate_subdomain_id(doc_id: str, subdomain_name: str) -> str:
    """
    生成 subdomain_id：使用 sha1(doc_id + "|Subdomain|" + subdomain_name) 的十六進位字串。
    確保不同報告的同名 Subdomain 不會共用節點。
    """
    content = f"{doc_id}|Subdomain|{subdomain_name}"
    return hashlib.sha1(content.encode('utf-8')).hexdigest()


def infer_subdomain(domain: Optional[str], item: Optional[str], raw_category: Optional[str] = None) -> Optional[str]:
    """
    根據大 Domain + item（評估／訓練項目）推論小類 Subdomain。

    Domain 與小類對應關係（由使用者提供）：
    - 知覺動作功能: 粗大動作、精細動作、感覺統合
    - 吞嚥/口腔功能: 口腔動作、吞嚥反射
    - 口語溝通功能: 口語理解、口語表達
    - 認知功能: 認知功能
    - 社會情緒功能: 情緒行為與社會適應功能
    """
    if not domain:
        return None
    # 綜合評估：目前先將小類視為「類別」本身（主訴與就診問題、團隊評估總結…）
    if domain == "綜合評估":
        if raw_category:
            return "".join(str(raw_category).split())
        return None
    if not item:
        return None
    # 去除所有空白與換行，避免 "粗大動作\n發展遲緩" 被空白干擾
    s = "".join(str(item).split())

    if domain == "知覺動作功能":
        if "粗大動作" in s or "粗動作" in s:
            return "粗大動作"
        if "精細動作" in s:
            return "精細動作"
        if "感覺統合" in s:
            return "感覺統合"

    if domain == "吞嚥/口腔功能":
        if "口腔動作" in s or "口腔功能" in s:
            return "口腔動作"
        # item 中出現任何「吞嚥」相關就歸到吞嚥反射/功能
        if "吞嚥" in s:
            return "吞嚥反射"

    if domain == "口語溝通功能":
        # 口語理解 / 理解訓練 → 口語理解
        if "口語理解" in s or "理解訓練" in s:
            return "口語理解"
        # 口語表達 / 表達訓練 / 語用訓練 / 說話 → 口語表達
        if any(k in s for k in ["口語表達", "表達訓練", "語用訓練", "說話"]):
            return "口語表達"

    if domain == "認知功能":
        return "認知功能"

    if domain == "社會情緒功能":
        # 評估 / 訓練項目都會帶到「情緒行為與社會適應功能」
        return "情緒行為與社會適應功能"

    return None


def ensure_constraints(driver, database: Optional[str]) -> None:
    """
    建立必要的唯一約束，重複執行也不會出錯。
    
    注意：Domain 和 Subdomain 現在使用 domain_id/subdomain_id（包含 doc_id scope）
    而非 name，以確保每份報告的圖完全獨立。
    """
    stmts = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Report) REQUIRE r.doc_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.domain_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sd:Subdomain) REQUIRE sd.subdomain_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Assessment) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Training) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Suggestion) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ai:AssessmentItem) REQUIRE ai.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (oi:ObservationItem) REQUIRE oi.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ti:TrainingItem) REQUIRE ti.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (si:SuggestionItem) REQUIRE si.id IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for cy in stmts:
            session.run(cy)


def upsert_unit_with_subdomain(tx, unit: Dict[str, Any]) -> None:
    """
    與 upsert_unit 類似，但多了 Subdomain：
    - Domain/name 來自 normalized_category
    - Subdomain/name 由 infer_subdomain 推論
    - 使用 domain_id 和 subdomain_id 確保每份報告獨立
    """
    doc_id = unit.get("doc_id")
    table_id = unit.get("table_id")
    group_id = unit.get("group_id")
    raw_category = unit.get("category")
    norm_cat = unit.get("normalized_category")
    item = unit.get("item")
    page_start = unit.get("page_start")
    page_end = unit.get("page_end")
    text = unit.get("text") or ""

    # 推論實際 Domain 名稱與 Subdomain
    domain_name = infer_domain(norm_cat, raw_category, item)
    subdomain = infer_subdomain(domain_name, item, raw_category=raw_category)
    
    # 計算 domain_id 和 subdomain_id（如果有對應的 name）
    domain_id = generate_domain_id(doc_id, domain_name) if domain_name else None
    subdomain_id = generate_subdomain_id(doc_id, subdomain) if subdomain else None
    
    # 切出各區段文字
    sec = split_unit_text_sections(text)
    # 進一步將 section 文字拆成 item 列表
    # - *_items_all: 所有層級（level=1 主項 + level=2 子項）都保留
    # - *_items: 只取 level=1，給 Section -[:HAS_ITEM]-> 大分項用
    assessment_items_all = split_section_items(sec["assessment_result"])
    observation_items_all = split_section_items(sec["observation"])
    training_items_all = split_section_items(sec["training"])
    suggestion_items_all = split_section_items(sec["suggestion"])
    assessment_items = [it for it in assessment_items_all if it["level"] == 1]
    observation_items = [it for it in observation_items_all if it["level"] == 1]
    training_items = [it for it in training_items_all if it["level"] == 1]
    suggestion_items = [it for it in suggestion_items_all if it["level"] == 1]

    cypher = """
    // 報告節點
    MERGE (r:Report {doc_id: $doc_id})

    // Domain（若有，使用 domain_id 確保每份報告獨立）
    FOREACH (_ IN CASE WHEN $domain_id IS NULL THEN [] ELSE [1] END |
      MERGE (d:Domain {domain_id: $domain_id})
        SET d.doc_id = $doc_id,
            d.name = $domain_name
      MERGE (r)-[:COVERS_DOMAIN]->(d)
    )

    // Subdomain（若有，使用 subdomain_id 確保每份報告獨立）
    FOREACH (_ IN CASE WHEN $subdomain_id IS NULL OR $domain_id IS NULL THEN [] ELSE [1] END |
      MERGE (d:Domain {domain_id: $domain_id})
      MERGE (sd:Subdomain {subdomain_id: $subdomain_id})
        SET sd.doc_id = $doc_id,
            sd.name = $subdomain
      MERGE (d)-[:HAS_SUBDOMAIN]->(sd)
    )

    // Assessment 節點（id 包含 doc_id）
    FOREACH (_ IN CASE WHEN $assessment_all = '' THEN [] ELSE [1] END |
      MERGE (a:Assessment {id: $doc_id + ':' + $group_id + ':assessment'})
        SET a.doc_id = $doc_id,
            a.group_id = $group_id,
            a.table_id = $table_id,
            a.raw_category = $raw_category,
            a.normalized_category = $domain_name,
            a.subdomain = $subdomain,
            a.item = $item,
            a.text = $assessment_all,
            a.date = $assessment_date,
            a.result = $assessment_result,
            a.tools = $assessment_tools,
            a.page_start = $page_start,
            a.page_end = $page_end
      // Subdomain 直接連到 Assessment（使用 subdomain_id）
      FOREACH (_ IN CASE WHEN $subdomain_id IS NULL THEN [] ELSE [1] END |
        MERGE (sd:Subdomain {subdomain_id: $subdomain_id})
        MERGE (sd)-[:HAS_ASSESSMENT]->(a)
      )
      // 先建立所有 AssessmentItem（含主項與子項，id 包含 doc_id）
      FOREACH (ai IN $assessment_items_all |
        MERGE (aiNode:AssessmentItem {id: $doc_id + ':' + $group_id + ':assessment#' + toString(ai.seq)})
          SET aiNode.doc_id = $doc_id,
              aiNode.group_id = $group_id,
              aiNode.index = ai.index,
              aiNode.level = ai.level,
              aiNode.seq = ai.seq,
              aiNode.parent_seq = ai.parent_seq,
              aiNode.text = ai.text
      )
      // Section 只連到 level=1 的大分項
      FOREACH (ai IN $assessment_items |
        MERGE (rootAI:AssessmentItem {id: $doc_id + ':' + $group_id + ':assessment#' + toString(ai.seq)})
        MERGE (a)-[:HAS_ITEM]->(rootAI)
      )
      // AssessmentItem 父子層級（1. → 1)）
      FOREACH (ai IN $assessment_items_all |
        FOREACH (_ IN CASE WHEN ai.parent_seq IS NULL THEN [] ELSE [1] END |
          MERGE (parentAI:AssessmentItem {id: $doc_id + ':' + $group_id + ':assessment#' + toString(ai.parent_seq)})
          MERGE (childAI:AssessmentItem {id: $doc_id + ':' + $group_id + ':assessment#' + toString(ai.seq)})
          MERGE (parentAI)-[:HAS_SUBITEM]->(childAI)
        )
      )
    )

    // Observation 節點（id 包含 doc_id）
    FOREACH (_ IN CASE WHEN $observation_text = '' THEN [] ELSE [1] END |
      MERGE (o:Observation {id: $doc_id + ':' + $group_id + ':observation'})
        SET o.doc_id = $doc_id,
            o.group_id = $group_id,
            o.table_id = $table_id,
            o.raw_category = $raw_category,
            o.normalized_category = $domain_name,
            o.subdomain = $subdomain,
            o.item = $item,
            o.text = $observation_text,
            o.page_start = $page_start,
            o.page_end = $page_end
      // Subdomain 直接連到 Observation（使用 subdomain_id）
      FOREACH (_ IN CASE WHEN $subdomain_id IS NULL THEN [] ELSE [1] END |
        MERGE (sd:Subdomain {subdomain_id: $subdomain_id})
        MERGE (sd)-[:HAS_OBSERVATION]->(o)
      )
      // 先建立所有 ObservationItem（含主項與子項，id 包含 doc_id）
      FOREACH (oi IN $observation_items_all |
        MERGE (oiNode:ObservationItem {id: $doc_id + ':' + $group_id + ':observation#' + toString(oi.seq)})
          SET oiNode.doc_id = $doc_id,
              oiNode.group_id = $group_id,
              oiNode.index = oi.index,
              oiNode.level = oi.level,
              oiNode.seq = oi.seq,
              oiNode.parent_seq = oi.parent_seq,
              oiNode.text = oi.text
      )
      // Section 只連到 level=1 的大分項
      FOREACH (oi IN $observation_items |
        MERGE (rootOI:ObservationItem {id: $doc_id + ':' + $group_id + ':observation#' + toString(oi.seq)})
        MERGE (o)-[:HAS_ITEM]->(rootOI)
      )
      // ObservationItem 父子層級
      FOREACH (oi IN $observation_items_all |
        FOREACH (_ IN CASE WHEN oi.parent_seq IS NULL THEN [] ELSE [1] END |
          MERGE (parentOI:ObservationItem {id: $doc_id + ':' + $group_id + ':observation#' + toString(oi.parent_seq)})
          MERGE (childOI:ObservationItem {id: $doc_id + ':' + $group_id + ':observation#' + toString(oi.seq)})
          MERGE (parentOI)-[:HAS_SUBITEM]->(childOI)
        )
      )
    )

    // Training 節點（id 包含 doc_id）
    FOREACH (_ IN CASE WHEN $training_text = '' THEN [] ELSE [1] END |
      MERGE (t:Training {id: $doc_id + ':' + $group_id + ':training'})
        SET t.doc_id = $doc_id,
            t.group_id = $group_id,
            t.table_id = $table_id,
            t.raw_category = $raw_category,
            t.normalized_category = $domain_name,
            t.subdomain = $subdomain,
            t.item = $item,
            t.text = $training_text,
            t.page_start = $page_start,
            t.page_end = $page_end
      // Subdomain 直接連到 Training（使用 subdomain_id）
      FOREACH (_ IN CASE WHEN $subdomain_id IS NULL THEN [] ELSE [1] END |
        MERGE (sd:Subdomain {subdomain_id: $subdomain_id})
        MERGE (sd)-[:HAS_TRAINING]->(t)
      )
      // 先建立所有 TrainingItem（含主項與子項，id 包含 doc_id）
      FOREACH (ti IN $training_items_all |
        MERGE (tiNode:TrainingItem {id: $doc_id + ':' + $group_id + ':training#' + toString(ti.seq)})
          SET tiNode.doc_id = $doc_id,
              tiNode.group_id = $group_id,
              tiNode.index = ti.index,
              tiNode.level = ti.level,
              tiNode.seq = ti.seq,
              tiNode.parent_seq = ti.parent_seq,
              tiNode.text = ti.text
      )
      // Section 只連到 level=1 的大分項
      FOREACH (ti IN $training_items |
        MERGE (rootTI:TrainingItem {id: $doc_id + ':' + $group_id + ':training#' + toString(ti.seq)})
        MERGE (t)-[:HAS_ITEM]->(rootTI)
      )
      // TrainingItem 父子層級
      FOREACH (ti IN $training_items_all |
        FOREACH (_ IN CASE WHEN ti.parent_seq IS NULL THEN [] ELSE [1] END |
          MERGE (parentTI:TrainingItem {id: $doc_id + ':' + $group_id + ':training#' + toString(ti.parent_seq)})
          MERGE (childTI:TrainingItem {id: $doc_id + ':' + $group_id + ':training#' + toString(ti.seq)})
          MERGE (parentTI)-[:HAS_SUBITEM]->(childTI)
        )
      )
    )

    // Suggestion 節點（id 包含 doc_id）
    FOREACH (_ IN CASE WHEN $suggestion_text = '' THEN [] ELSE [1] END |
      MERGE (s:Suggestion {id: $doc_id + ':' + $group_id + ':suggestion'})
        SET s.doc_id = $doc_id,
            s.group_id = $group_id,
            s.table_id = $table_id,
            s.raw_category = $raw_category,
            s.normalized_category = $domain_name,
            s.subdomain = $subdomain,
            s.item = $item,
            s.text = $suggestion_text,
            s.page_start = $page_start,
            s.page_end = $page_end
      // Subdomain 直接連到 Suggestion（使用 subdomain_id）
      FOREACH (_ IN CASE WHEN $subdomain_id IS NULL THEN [] ELSE [1] END |
        MERGE (sd:Subdomain {subdomain_id: $subdomain_id})
        MERGE (sd)-[:HAS_SUGGESTION]->(s)
      )
      // 先建立所有 SuggestionItem（含主項與子項，id 包含 doc_id）
      FOREACH (si IN $suggestion_items_all |
        MERGE (siNode:SuggestionItem {id: $doc_id + ':' + $group_id + ':suggestion#' + toString(si.seq)})
          SET siNode.doc_id = $doc_id,
              siNode.group_id = $group_id,
              siNode.index = si.index,
              siNode.level = si.level,
              siNode.seq = si.seq,
              siNode.parent_seq = si.parent_seq,
              siNode.text = si.text
      )
      // Section 只連到 level=1 的大分項
      FOREACH (si IN $suggestion_items |
        MERGE (rootSI:SuggestionItem {id: $doc_id + ':' + $group_id + ':suggestion#' + toString(si.seq)})
        MERGE (s)-[:HAS_ITEM]->(rootSI)
      )
      // SuggestionItem 父子層級
      FOREACH (si IN $suggestion_items_all |
        FOREACH (_ IN CASE WHEN si.parent_seq IS NULL THEN [] ELSE [1] END |
          MERGE (parentSI:SuggestionItem {id: $doc_id + ':' + $group_id + ':suggestion#' + toString(si.parent_seq)})
          MERGE (childSI:SuggestionItem {id: $doc_id + ':' + $group_id + ':suggestion#' + toString(si.seq)})
          MERGE (parentSI)-[:HAS_SUBITEM]->(childSI)
        )
      )
    )
    """

    tx.run(
        cypher,
        doc_id=doc_id,
        table_id=table_id,
        group_id=group_id,
        raw_category=raw_category,
        domain_name=domain_name,
        domain_id=domain_id,
        subdomain=subdomain,
        subdomain_id=subdomain_id,
        item=item,
        page_start=page_start,
        page_end=page_end,
        text=text,
        assessment_all="\n".join(
            [p for p in [sec["assessment_date"], sec["assessment_result"], sec["assessment_tools"]] if p]
        ),
        assessment_date=sec["assessment_date"],
        assessment_result=sec["assessment_result"],
        assessment_tools=sec["assessment_tools"],
        observation_text=sec["observation"],
        training_text=sec["training"],
        suggestion_text=sec["suggestion"],
        assessment_items_all=assessment_items_all,
        assessment_items=assessment_items,
        observation_items_all=observation_items_all,
        observation_items=observation_items,
        training_items_all=training_items_all,
        training_items=training_items,
        suggestion_items_all=suggestion_items_all,
        suggestion_items=suggestion_items,
    )
