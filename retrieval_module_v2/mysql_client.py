from typing import List, Dict, Any
from sqlalchemy import text
from .types import CandidateNode

class MySQLResourceClient:
    """
    Handles retrieval of local resources from MySQL database.
    """
    def __init__(self, sql_db):
        self.sql_db = sql_db

    def fetch_resources_by_region(self, region: str, keywords: str = None) -> List[CandidateNode]:
        if not region or not self.sql_db:
            return []
        
        print(f"[MySQLClient] Fetching Resources for region: {region}, keywords: {keywords}")
        
        candidates = []
        kw_pattern = f"%{keywords}%" if keywords else "%%"
        
        # 1. 查詢 sfaa_units (社會局/社政單位)
        sfaa_query = text("""
            SELECT id, unit_name, category, address, phone, service_area 
            FROM sfaa_units 
            WHERE (address LIKE :region OR service_area LIKE :region)
            AND (unit_name LIKE :kw OR category LIKE :kw OR service_area LIKE :kw)
            LIMIT 10
        """)
        
        # 2. 查詢 community_intervention_units (療育據點/社區單位)
        community_query = text("""
            SELECT id, city, location_name, service_address, contact_phone, service_scope, service_unit
            FROM community_intervention_units
            WHERE (city LIKE :region OR service_address LIKE :region OR service_scope LIKE :region)
            AND (location_name LIKE :kw OR service_scope LIKE :kw OR service_unit LIKE :kw)
            LIMIT 10
        """)
        
        try:
            # 處理 sfaa_units 結果
            sfaa_result = self.sql_db.session.execute(sfaa_query, {"region": f"%{region}%", "kw": kw_pattern})
            sfaa_count = 0
            for row in sfaa_result:
                sfaa_count += 1
                text_content = (
                    f"【在地資源-機構】{row.unit_name} ({row.category})\n"
                    f"地址: {row.address}\n"
                    f"電話: {row.phone}\n"
                    f"服務區域: {row.service_area}"
                )
                print(f"  [MySQL] 找到機構: {row.unit_name} ({row.category})")
                candidates.append(CandidateNode(
                    node_id=f"sfaa_{row.id}",
                    label="LocalResource",
                    text=text_content,
                    properties={
                        "unit_name": row.unit_name,
                        "category": row.category,
                        "address": row.address,
                        "phone": row.phone,
                        "source": "sfaa_units"
                    }
                ))
            
            # 處理 community_intervention_units 結果
            community_result = self.sql_db.session.execute(community_query, {"region": f"%{region}%", "kw": kw_pattern})
            community_count = 0
            for row in community_result:
                community_count += 1
                text_content = (
                    f"【在地資源-據點】{row.location_name} ({row.service_unit or '社區療育'})\n"
                    f"地址: {row.service_address}\n"
                    f"電話: {row.contact_phone}\n"
                    f"服務內容: {row.service_scope}"
                )
                print(f"  [MySQL] 找到據點: {row.location_name} ({row.service_unit or '社區療育'})")
                candidates.append(CandidateNode(
                    node_id=f"community_{row.id}",
                    label="LocalResource",
                    text=text_content,
                    properties={
                        "unit_name": row.location_name,
                        "category": row.service_unit,
                        "address": row.service_address,
                        "phone": row.contact_phone,
                        "source": "community_intervention_units"
                    }
                ))
            
            print(f"[MySQLClient] 檢索完成: 機構 {sfaa_count} 筆, 據點 {community_count} 筆")
                
        except Exception as e:
            print(f"[MySQLClient][Error] query failed: {e}")
            
        return candidates
