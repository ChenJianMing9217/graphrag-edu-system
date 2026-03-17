from .strategy_mapper import StrategyMapper
from .execution_engine import ExecutionEngine
from .reranker import Reranker
from .types import SearchStrategy, SearchOperation, SearchOperationType, CandidateNode
from typing import List, Dict, Any

class RetrievalModuleV2:
    def __init__(self, graph_client, sql_db=None, text_encoder=None):
        self.graph_client = graph_client
        self.sql_db = sql_db
        self.strategy_mapper = StrategyMapper()
        self.execution_engine = ExecutionEngine(graph_client, sql_db=sql_db)
        self.reranker = Reranker(text_encoder)

    def retrieve(self, turn_state: Dict[str, Any], user_query: str, doc_id: str) -> List[CandidateNode]:
        # 1. Map DST to Strategy
        strategy = self.strategy_mapper.map_dst_to_strategy(turn_state, user_query)
        
        # 2. Execute Strategy
        candidates = self.execution_engine.execute_strategy(strategy, doc_id)
        
        # 3. Rerank
        task_label = turn_state.get("task_pred")
        domain_dist = turn_state.get("domain_distribution")
        final_results = self.reranker.rerank(
            candidates, 
            user_query, 
            strategy.rerank_config,
            task_label=task_label,
            domain_distribution=domain_dist
        )
        
        return final_results
