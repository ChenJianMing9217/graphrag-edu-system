from typing import List, Dict, Any
from .types import SearchOperation, SearchOperationType, CandidateNode
from .mysql_client import MySQLResourceClient

class ExecutionEngine:
    """
    Executes SearchOperations against Neo4j.
    """
    def __init__(self, graph_client, sql_db=None):
        self.graph_client = graph_client
        self.mysql_client = MySQLResourceClient(sql_db) if sql_db else None

    def execute_strategy(self, strategy, doc_id: str) -> List[CandidateNode]:
        all_candidates = []
        
        # 分離 Neo4j 與 MySQL 的操作
        neo4j_ops = []
        mysql_ops = []
        for op in strategy.operations:
            if op.op_type == SearchOperationType.MYSQL_RESOURCE_FETCH:
                mysql_ops.append(op)
            else:
                neo4j_ops.append(op)
        
        # 執行 Neo4j 操作 (僅在有需要時開啟 session)
        if neo4j_ops:
            with self.graph_client.driver.session(database=self.graph_client.database) as session:
                for op in neo4j_ops:
                    if op.op_type == SearchOperationType.SUBDOMAIN_FETCH:
                        all_candidates.extend(self._fetch_subdomain(session, op.params, doc_id))
                    elif op.op_type == SearchOperationType.SUMMARY_FETCH:
                        all_candidates.extend(self._fetch_summary(session, op.params, doc_id))
                    elif op.op_type == SearchOperationType.META_FETCH:
                        all_candidates.extend(self._fetch_meta(session, op.params, doc_id))
                    elif op.op_type == SearchOperationType.CONTEXTUAL_EXPANSION:
                        all_candidates.extend(self._fetch_contextual(session, op.params, doc_id))
        
        # 執行 MySQL 操作
        for op in mysql_ops:
            all_candidates.extend(self._fetch_mysql_resources(op.params))
            
        return all_candidates

    def _fetch_subdomain(self, session, params: Dict[str, Any], doc_id: str) -> List[CandidateNode]:
        subdomain = params.get("subdomain", "").strip()
        print(f"[Engine] Fetching Subdomain: '{subdomain}' for doc: '{doc_id}'")
        
        # Use trim() for robustness against invisible characters
        cypher = """
        MATCH (r:Report {id: $doc_id})-[:HAS_DOMAIN]->(d:Domain)-[:HAS_SUBDOMAIN]->(sd:Subdomain)
        WHERE trim(sd.name) = trim($subdomain)
        MATCH (sd)-[:HAS_ASSESSMENT_TOOLS|HAS_OBSERVATIONS|HAS_RECOMMENDATIONS|HAS_TRAINING_PLAN|HAS_SCORES]->(h:CategoryHub)
        MATCH (h)-[:USED_TOOL|OBSERVED|RECOMMENDED|TRAINED_BY|HAS_VALUE]->(item)
        OPTIONAL MATCH (item)-[:HAS_SUB_ITEM]->(sub:SubItem)
        RETURN labels(item)[0] as label, item.text as text, item.raw_text as raw_text, item.id as id, 
               h.name as category, collect(sub.text) as sub_items
        """
        result = session.run(cypher, doc_id=doc_id, subdomain=subdomain)
        candidates = []
        for record in result:
            text = record["text"]
            if record["sub_items"]:
                text += "\n  - " + "\n  - ".join(record["sub_items"])
                
            candidates.append(CandidateNode(
                node_id=record["id"],
                label=record["label"],
                text=text,
                properties={
                    "raw_text": record["raw_text"], 
                    "category": record["category"],
                    "subdomain": subdomain
                }
            ))
            
        if not candidates:
            print(f"[Engine][Warning] No items found for subdomain '{subdomain}'. Checking available subdomains...")
            check_cypher = "MATCH (r:Report {id: $doc_id})-[:HAS_DOMAIN]->(d)-[:HAS_SUBDOMAIN]->(sd) RETURN sd.name as name"
            available = [r["name"] for r in session.run(check_cypher, doc_id=doc_id)]
            print(f"[Engine][Debug] Available subdomains in graph: {available}")
            
        return candidates

    def _fetch_summary(self, session, params: Dict[str, Any], doc_id: str) -> List[CandidateNode]:
        print(f"[Engine] Fetching Summary for doc: {doc_id}")
        # Summary hubs use HAS_CATEGORY but different item relations
        cypher = """
        MATCH (r:Report {id: $doc_id})-[:HAS_SUMMARY]->(s:Summary)
        MATCH (s)-[:HAS_CATEGORY]->(h:CategoryHub)-[:HAS_CONTENT|DIAGNOSED_AS|HAS_RESULT|HAS_SUGGESTION]->(item)
        RETURN labels(item)[0] as label, item.text as text, item.raw_text as raw_text, item.id as id, 
               h.name as category, [] as sub_items
        """
        result = session.run(cypher, doc_id=doc_id)
        candidates = []
        for record in result:
            candidates.append(CandidateNode(
                node_id=record["id"],
                label=record["label"],
                text=f"[{record['category']}] {record['text']}",
                properties={"category": record["category"]}
            ))
        return candidates

    def _fetch_meta(self, session, params: Dict[str, Any], doc_id: str) -> List[CandidateNode]:
        print(f"[Engine] Fetching Meta for doc: {doc_id}")
        cypher = """
        MATCH (r:Report {id: $doc_id})-[:HAS_META]->(m:Meta)
        RETURN m
        """
        result = session.run(cypher, doc_id=doc_id)
        candidates = []
        record = result.single()
        if record:
            m = record["m"]
            meta_text = (
                f"個案姓名: {m.get('patient_name')}\n"
                f"性別: {m.get('gender')}\n"
                f"年齡: {m.get('age')}\n"
                f"就診日期: {m.get('doctor_visit_date')}\n"
                f"報告完成日期: {m.get('report_complete_date')}"
            )
            candidates.append(CandidateNode(
                node_id=f"{doc_id}_meta",
                label="Meta",
                text=meta_text,
                properties=dict(m)
            ))
        return candidates

    def _fetch_contextual(self, session, params: Dict[str, Any], doc_id: str) -> List[CandidateNode]:
        return []

    def _fetch_mysql_resources(self, params: Dict[str, Any]) -> List[CandidateNode]:
        if not self.mysql_client:
            return []
        
        region = params.get("region")
        keywords = params.get("keywords")
        return self.mysql_client.fetch_resources_by_region(region, keywords)
