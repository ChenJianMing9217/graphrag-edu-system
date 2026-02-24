#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → Neo4j 完整流程：呼叫現有的正確模組

流程：
1. 呼叫 pdf_processor.py 的函數解析 PDF → 產生 grouped JSON
2. 呼叫 load_grouped_tables_to_neo4j.py 的函數建立圖譜

使用方式：
    python process_pdf_to_graph.py \
        --pdf IEP_ex.pdf \
        --doc-id report_1_example_child \
        --password your_password
"""

import os
import sys
import argparse
from datetime import datetime

from .pdf_processor_minimal import PDFProcessor, build_grouped_table_units, merge_table_rows_by_id
from .load_grouped_tables_to_neo4j_minimal import (
    ensure_constraints,
    upsert_unit_with_subdomain,
    GraphDatabase
)

def process_pdf_to_graph(
    pdf_path: str,
    doc_id: str,
    neo4j_uri: str = "bolt://10.242.84.204:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    save_json: bool = True,
    json_output_path: str = None
):
    """
    完整流程：PDF → JSON → Neo4j
    
    參數：
        pdf_path: PDF 檔案路徑
        doc_id: 文件 ID
        neo4j_uri: Neo4j URI
        neo4j_user: Neo4j 使用者
        neo4j_password: Neo4j 密碼
        neo4j_database: Neo4j 資料庫名稱
        save_json: 是否儲存 JSON（預設 True）
        json_output_path: JSON 輸出路徑（可選）
    """
    
    print("=" * 80)
    print("PDF → Neo4j 完整流程")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # ========================================================================
    # 步驟 1: 使用 pdf_processor 解析 PDF
    # ========================================================================
    print(f"\n[步驟 1/3] 解析 PDF: {pdf_path}")
    print(f"  doc_id: {doc_id}")
    
    try:
        # 導入 pdfplumber
        import pdfplumber
    except ImportError:
        print("  ✗ 錯誤: pdfplumber 未安裝")
        print("  請執行: pip install pdfplumber")
        return None
    
    # 使用 pdf_processor 的解析邏輯
    processor = PDFProcessor()
    all_table_rows = []
    
    print(f"  開始解析 PDF...")
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            if pno % 5 == 0:
                print(f"    處理第 {pno} 頁...")
            table_rows = processor.extract_tables_structured(page, doc_id, pno)
            if table_rows:
                all_table_rows.extend(table_rows)
    
    print(f"  ✓ 提取了 {len(all_table_rows)} 列表格資料")
    
    if not all_table_rows:
        print("  ✗ 警告: 沒有提取到任何表格資料")
        print("  請確認 PDF 包含表格結構")
        return None
    
    # 合併跨頁表格
    print(f"  合併跨頁表格...")
    merged_tables = merge_table_rows_by_id(all_table_rows)
    print(f"  ✓ 合併後共 {len(merged_tables)} 張表格")
    
    # 產生 grouped units
    print(f"  產生功能區塊分組...")
    grouped_units = build_grouped_table_units(merged_tables)
    print(f"  ✓ 產生了 {len(grouped_units)} 個功能區塊")
    
    # ========================================================================
    # 步驟 2: 儲存 JSON（可選）
    # ========================================================================
    if save_json:
        if json_output_path is None:
            base, _ = os.path.splitext(pdf_path)
            json_output_path = base + "_grouped.json"
        
        print(f"\n[步驟 2/3] 儲存 JSON: {json_output_path}")
        import json
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(grouped_units, f, ensure_ascii=False, indent=2)
        print(f"  ✓ JSON 已儲存")
    else:
        print(f"\n[步驟 2/3] 跳過 JSON 儲存")
    
    # ========================================================================
    # 步驟 3: 使用 load_grouped_tables_to_neo4j 建立圖譜
    # ========================================================================
    print(f"\n[步驟 3/3] 建立 Neo4j 圖譜")
    print(f"  連接到: {neo4j_uri}")
    print(f"  資料庫: {neo4j_database}")
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    try:
        # 建立約束
        print(f"  建立約束...")
        ensure_constraints(driver, neo4j_database)
        print(f"  ✓ 約束建立完成")
        
        # 寫入資料
        print(f"  開始寫入 {len(grouped_units)} 個功能區塊...")
        
        with driver.session(database=neo4j_database) as session:
            for idx, unit in enumerate(grouped_units, start=1):
                # 使用 upsert_unit_with_subdomain（有 Subdomain 結構）
                session.execute_write(upsert_unit_with_subdomain, unit)
                
                if idx % 10 == 0:
                    print(f"    已處理 {idx}/{len(grouped_units)} 個功能區塊")
        
        print(f"  ✓ 全部寫入完成")
        
        # 統計資訊
        print(f"\n  查詢統計資訊...")
        with driver.session(database=neo4j_database) as session:
            # 統計各類節點
            stats = {}
            labels = ["Report", "Domain", "Subdomain", "Assessment", "Observation", 
                     "Training", "Suggestion"]
            
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                stats[label] = result.single()["count"]
            
            print(f"\n  [統計資訊]")
            print(f"    Reports: {stats['Report']}")
            print(f"    Domains: {stats['Domain']}")
            print(f"    Subdomains: {stats['Subdomain']}")
            print(f"    Assessments: {stats['Assessment']}")
            print(f"    Observations: {stats['Observation']}")
            print(f"    Trainings: {stats['Training']}")
            print(f"    Suggestions: {stats['Suggestion']}")
        
    except Exception as e:
        print(f"  ✗ Neo4j 錯誤: {e}")
        raise
    
    finally:
        driver.close()
    
    # ========================================================================
    # 完成
    # ========================================================================
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n" + "=" * 80)
    print(f"✓ 完成！總耗時: {elapsed:.2f} 秒")
    print("=" * 80)
    
    return {
        "grouped_units_count": len(grouped_units),
        "json_path": json_output_path if save_json else None,
        "elapsed_seconds": elapsed,
    }


def main():
    """命令列介面"""
    parser = argparse.ArgumentParser(
        description="PDF → Neo4j 完整流程（使用現有正確模組）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python process_pdf_to_graph.py \\
    --pdf IEP_ex.pdf \\
    --doc-id report_1_example_child \\
    --password your_password

  python process_pdf_to_graph.py \\
    --pdf IEP_ex3.pdf \\
    --doc-id report_3_child \\
    --password password \\
    --no-save-json
        """
    )
    
    # PDF 相關參數
    parser.add_argument("--pdf", required=True, help="PDF 檔案路徑")
    parser.add_argument("--doc-id", required=True, help="文件 ID")
    
    # Neo4j 相關參數
    parser.add_argument("--neo4j-uri", default="bolt://10.242.84.204:7687", 
                       help="Neo4j URI (預設: bolt://10.242.84.204:7687)")
    parser.add_argument("--user", default="neo4j", 
                       help="Neo4j 使用者 (預設: neo4j)")
    parser.add_argument("--password", required=True, 
                       help="Neo4j 密碼")
    parser.add_argument("--database", default="neo4j", 
                       help="Neo4j 資料庫名稱 (預設: neo4j)")
    
    # JSON 輸出相關參數
    parser.add_argument("--save-json", dest="save_json", action="store_true",
                       help="儲存 JSON 檔案 (預設)")
    parser.add_argument("--no-save-json", dest="save_json", action="store_false",
                       help="不儲存 JSON 檔案")
    parser.set_defaults(save_json=True)
    
    parser.add_argument("--json-output", default=None, 
                       help="JSON 輸出路徑 (預設: PDF路徑_grouped.json)")
    
    args = parser.parse_args()
    
    # 檢查 PDF 檔案是否存在
    if not os.path.exists(args.pdf):
        print(f"✗ 錯誤: PDF 檔案不存在: {args.pdf}")
        return 1
    
    try:
        result = process_pdf_to_graph(
            pdf_path=args.pdf,
            doc_id=args.doc_id,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.user,
            neo4j_password=args.password,
            neo4j_database=args.database,
            save_json=args.save_json,
            json_output_path=args.json_output,
        )
        
        if result is None:
            return 1
        
        print("\n✓ 處理成功！")
        
        if result.get("json_path"):
            print(f"\nJSON 檔案: {result['json_path']}")
        
        print(f"\n可以在 Neo4j Browser 中執行以下查詢驗證：")
        print(f"  MATCH (r:Report {{doc_id: '{args.doc_id}'}})")
        print(f"  -[:COVERS_DOMAIN]->(d:Domain)-[:HAS_SUBDOMAIN]->(sd:Subdomain)")
        print(f"  RETURN r.doc_id, d.name, sd.name")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

