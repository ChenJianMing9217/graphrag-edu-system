# 測試 anchor_embeddings：每個領域抽不同的句子，並與問題比較相似度

from __future__ import annotations
import numpy as np
from embedding import EncoderConfig, TextEncoder, encode_anchors, cosine_sim
from domain_anchors import DOMAINS, DOMAIN_ANCHORS


def test_anchor_embeddings(test_question: str = "孩子走路不穩怎麼辦？"):
    """
    測試 encode_anchors，每個領域抽不同的句子，並與問題比較相似度
    
    Args:
        test_question: 要比較的測試問題（預設為粗大動作相關問題）
    """
    
    print("=" * 80)
    print("測試 anchor_embeddings（每個領域抽不同句子 + 問題比較）")
    print("=" * 80)
    print(f"測試問題: {test_question}\n")
    
    # 初始化編碼器
    print("正在初始化 TextEncoder...")
    cfg = EncoderConfig(device="cpu")  # 使用 CPU 以避免 CUDA 依賴
    encoder = TextEncoder(cfg)
    print("✓ TextEncoder 初始化完成\n")
    
    # 編碼所有 anchors
    print("正在編碼 anchors...")
    anchor_embeddings = encode_anchors(encoder, DOMAIN_ANCHORS, DOMAINS)
    print(f"✓ 完成編碼 {len(anchor_embeddings)} 個 domain\n")
    
    # 從每個領域抽取不同的句子和向量（循環使用索引）
    selected_sentences = []
    selected_vectors = []
    domain_names = []
    selected_indices = []
    
    print("=" * 80)
    print("抽取結果（每個領域抽不同的句子）：")
    print("=" * 80)
    
    for idx, domain in enumerate(DOMAINS):
        anchor_sentences = DOMAIN_ANCHORS[domain]  # List[str]
        embeddings = anchor_embeddings[domain]  # List[np.ndarray]
        
        # 循環使用索引（領域0用索引0，領域1用索引1，以此類推，超出範圍則取模）
        sentence_index = idx % len(anchor_sentences)
        
        selected_sentence = anchor_sentences[sentence_index]
        selected_vector = embeddings[sentence_index]
        
        selected_sentences.append(selected_sentence)
        selected_vectors.append(selected_vector)
        domain_names.append(domain)
        selected_indices.append(sentence_index)
        
        print(f"\n【{domain}】")
        print(f"  抽取索引: {sentence_index}（第 {sentence_index + 1} 句）")
        print(f"  句子: {selected_sentence}")
        print(f"  向量維度: {selected_vector.shape}")
        print(f"  向量範數: {np.linalg.norm(selected_vector):.6f}")
    
    # 編碼測試問題
    print("\n" + "=" * 80)
    print("編碼測試問題：")
    print("=" * 80)
    question_vector = encoder.encode(test_question)
    print(f"問題: {test_question}")
    print(f"向量維度: {question_vector.shape}")
    print(f"向量範數: {np.linalg.norm(question_vector):.6f}")
    
    # 計算相似度
    print("\n" + "=" * 80)
    print("相似度比較結果：")
    print("=" * 80)
    
    similarities = []
    for domain, vector in zip(domain_names, selected_vectors):
        sim = cosine_sim(question_vector, vector)
        similarities.append((domain, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n測試問題與各領域的相似度（由高到低）：")
    print("-" * 80)
    for i, (domain, sim) in enumerate(similarities, 1):
        # 找到對應的句子和索引
        domain_idx = domain_names.index(domain)
        sentence = selected_sentences[domain_idx]
        sent_idx = selected_indices[domain_idx]
        
        print(f"{i:2d}. [{domain}] 相似度: {sim:.6f}")
        print(f"    句子（索引 {sent_idx}）: {sentence}")
    
    # 輸出句子列表
    print("\n" + "=" * 80)
    print("抽出句子列表：")
    print("=" * 80)
    for i, (domain, sentence, idx) in enumerate(zip(domain_names, selected_sentences, selected_indices), 1):
        print(f"{i}. [{domain}] (索引 {idx}) {sentence}")
    
    # 輸出向量列表（摘要）
    print("\n" + "=" * 80)
    print("向量列表（摘要）：")
    print("=" * 80)
    print(f"總共 {len(selected_vectors)} 個向量")
    print(f"向量維度: {selected_vectors[0].shape if selected_vectors else 'N/A'}")
    print(f"向量資料型別: {selected_vectors[0].dtype if selected_vectors else 'N/A'}")
    
    print("\n各領域向量範數：")
    for domain, vector in zip(domain_names, selected_vectors):
        norm = np.linalg.norm(vector)
        print(f"  {domain}: {norm:.6f}")
    
    # 輸出兩個陣列：句子陣列和向量陣列
    print("\n" + "=" * 80)
    print("輸出陣列：")
    print("=" * 80)
    
    print("\n【句子陣列】")
    print("selected_sentences = [")
    for i, sentence in enumerate(selected_sentences, 1):
        # 轉義引號以便在 Python 中正確顯示
        escaped_sentence = sentence.replace('"', '\\"').replace("'", "\\'")
        print(f'    "{escaped_sentence}",' if i < len(selected_sentences) else f'    "{escaped_sentence}"')
    print("]")
    
    print(f"\n句子陣列長度: {len(selected_sentences)}")
    
    print("\n【向量陣列】")
    print("selected_vectors = [")
    for i, vector in enumerate(selected_vectors, 1):
        # 輸出向量的前幾個元素作為示例
        print(f"    np.array([{', '.join([f'{x:.6f}' for x in vector[:5]])}...]),  # {domain_names[i-1]} (shape: {vector.shape})")
    print("]")
    
    # 將所有向量組合成一個 NumPy 陣列
    vectors_array = np.array(selected_vectors)
    print(f"\n向量陣列形狀: {vectors_array.shape}")
    print(f"向量陣列資料型別: {vectors_array.dtype}")
    
    print("\n【完整輸出】")
    print("=" * 80)
    print("句子陣列（Python list）:")
    print(selected_sentences)
    print("\n向量陣列（NumPy array）:")
    print(f"Shape: {vectors_array.shape}")
    print(f"前 3 個向量的前 5 個元素:")
    for i in range(min(3, len(vectors_array))):
        print(f"  [{domain_names[i]}] {vectors_array[i][:5]}")
    
    print("\n" + "=" * 80)
    print("測試完成！")
    print("=" * 80)
    
    return selected_sentences, selected_vectors, domain_names, question_vector, similarities


if __name__ == "__main__":
    import sys
    
    # 從命令行參數讀取測試問題（如果提供）
    test_question = "孩子走路不穩怎麼辦？"
    if len(sys.argv) > 1:
        # 如果參數是整數，視為舊版索引模式（向後兼容）
        try:
            sentence_index = int(sys.argv[1])
            print("⚠ 注意：已改為每個領域抽不同句子的模式，索引參數將被忽略")
            print("   如需指定測試問題，請直接提供問題文字作為參數\n")
        except ValueError:
            # 如果不是整數，視為測試問題
            test_question = " ".join(sys.argv[1:])
    
    try:
        test_anchor_embeddings(test_question=test_question)
    except ImportError as e:
        print(f"錯誤：缺少依賴套件 ({e})")
        print("請安裝: pip install sentence-transformers")
    except Exception as e:
        print(f"錯誤：{e}")
        import traceback
        traceback.print_exc()

