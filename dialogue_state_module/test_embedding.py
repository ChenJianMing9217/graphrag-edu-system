# 測試 embedding.py 模組

from __future__ import annotations
import numpy as np
import unittest
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

from embedding import (
    cosine_sim,
    EncoderConfig,
    TextEncoder,
    encode_anchors,
)
from domain_anchors import DOMAINS, DOMAIN_ANCHORS


class TestCosineSim(unittest.TestCase):
    """測試 cosine_sim 函數"""

    def test_identical_vectors(self):
        """測試相同向量，相似度應為 1.0"""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cosine_sim(vec, vec)
        assert abs(result - 1.0) < 1e-6, f"相同向量相似度應為 1.0，得到 {result}"

    def test_orthogonal_vectors(self):
        """測試正交向量，相似度應為 0.0"""
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0], dtype=np.float32)
        result = cosine_sim(vec1, vec2)
        assert abs(result - 0.0) < 1e-6, f"正交向量相似度應為 0.0，得到 {result}"

    def test_opposite_vectors(self):
        """測試相反向量，相似度應為 -1.0"""
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0], dtype=np.float32)
        result = cosine_sim(vec1, vec2)
        assert abs(result - (-1.0)) < 1e-6, f"相反向量相似度應為 -1.0，得到 {result}"

    def test_zero_vector(self):
        """測試零向量，應回傳 0.0"""
        vec1 = np.array([1.0, 2.0], dtype=np.float32)
        vec2 = np.array([0.0, 0.0], dtype=np.float32)
        result = cosine_sim(vec1, vec2)
        assert result == 0.0, f"零向量相似度應為 0.0，得到 {result}"

    def test_both_zero_vectors(self):
        """測試兩個零向量，應回傳 0.0"""
        vec1 = np.array([0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 0.0], dtype=np.float32)
        result = cosine_sim(vec1, vec2)
        assert result == 0.0, f"兩個零向量相似度應為 0.0，得到 {result}"

    def test_different_dtypes(self):
        """測試不同 dtype 的輸入"""
        vec1 = np.array([1.0, 2.0], dtype=np.float64)
        vec2 = np.array([2.0, 4.0], dtype=np.int32)
        result = cosine_sim(vec1, vec2)
        # 兩個向量方向相同，相似度應為 1.0
        assert abs(result - 1.0) < 1e-6, f"相同方向向量相似度應為 1.0，得到 {result}"

    def test_list_input(self):
        """測試 list 輸入（應自動轉換為 numpy array）"""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        result = cosine_sim(vec1, vec2)
        assert abs(result - 0.0) < 1e-6, f"正交向量相似度應為 0.0，得到 {result}"

    def test_high_dimensional(self):
        """測試高維向量"""
        vec1 = np.ones(100, dtype=np.float32)
        vec2 = np.ones(100, dtype=np.float32)
        result = cosine_sim(vec1, vec2)
        assert abs(result - 1.0) < 1e-6, f"相同高維向量相似度應為 1.0，得到 {result}"


class TestEncoderConfig(unittest.TestCase):
    """測試 EncoderConfig 資料類別"""

    def test_default_config(self):
        """測試預設配置"""
        cfg = EncoderConfig()
        assert cfg.model_name == "BAAI/bge-m3"
        assert cfg.device == "cuda"
        assert cfg.normalize is True

    def test_custom_config(self):
        """測試自訂配置"""
        cfg = EncoderConfig(
            model_name="test-model",
            device="cpu",
            normalize=False,
        )
        assert cfg.model_name == "test-model"
        assert cfg.device == "cpu"
        assert cfg.normalize is False


class TestTextEncoder(unittest.TestCase):
    """測試 TextEncoder 類別"""

    @patch("sentence_transformers.SentenceTransformer")
    def test_init(self, mock_sentence_transformer):
        """測試 TextEncoder 初始化"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig(model_name="test-model", device="cpu")
        encoder = TextEncoder(cfg)

        assert encoder.cfg == cfg
        mock_sentence_transformer.assert_called_once_with("test-model", device="cpu")
        assert encoder._model == mock_model

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_normal_text(self, mock_sentence_transformer):
        """測試 encode 正常文字"""
        mock_model = MagicMock()
        # 模擬 encode 回傳一個 384 維的向量
        mock_embedding = np.array([0.1] * 384, dtype=np.float32)
        mock_model.encode.return_value = [mock_embedding]
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig(normalize=True)
        encoder = TextEncoder(cfg)
        result = encoder.encode("測試文字")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 384
        mock_model.encode.assert_called_once_with(
            ["測試文字"],
            normalize_embeddings=True,
        )

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_empty_string(self, mock_sentence_transformer):
        """測試 encode 空字串"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig()
        encoder = TextEncoder(cfg)
        result = encoder.encode("")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 1
        assert result[0] == 0.0
        # 空字串不應調用 model.encode
        mock_model.encode.assert_not_called()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_whitespace_only(self, mock_sentence_transformer):
        """測試 encode 只有空白字元的字串"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig()
        encoder = TextEncoder(cfg)
        result = encoder.encode("   ")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 1
        assert result[0] == 0.0
        mock_model.encode.assert_not_called()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_none(self, mock_sentence_transformer):
        """測試 encode None（應轉換為空字串）"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig()
        encoder = TextEncoder(cfg)
        result = encoder.encode(None)  # type: ignore

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 1
        assert result[0] == 0.0
        mock_model.encode.assert_not_called()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_many(self, mock_sentence_transformer):
        """測試 encode_many 多個文字"""
        mock_model = MagicMock()
        # 模擬 encode 回傳兩個 384 維的向量
        mock_embeddings = [
            np.array([0.1] * 384, dtype=np.float32),
            np.array([0.2] * 384, dtype=np.float32),
        ]
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig(normalize=True)
        encoder = TextEncoder(cfg)
        texts = ["文字1", "文字2"]
        result = encoder.encode_many(texts)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 384)
        mock_model.encode.assert_called_once_with(
            ["文字1", "文字2"],
            normalize_embeddings=True,
        )

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_many_with_empty_strings(self, mock_sentence_transformer):
        """測試 encode_many 包含空字串"""
        mock_model = MagicMock()
        mock_embeddings = [
            np.array([0.1] * 384, dtype=np.float32),
            np.array([0.2] * 384, dtype=np.float32),
        ]
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig()
        encoder = TextEncoder(cfg)
        texts = ["文字1", "", "   ", "文字2"]
        result = encoder.encode_many(texts)

        # 空字串會被 strip 後變成空字串，但仍會傳給 model
        # 實際行為取決於 SentenceTransformer 的處理
        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_normalize_false(self, mock_sentence_transformer):
        """測試 normalize=False 的情況"""
        mock_model = MagicMock()
        mock_embedding = np.array([0.1] * 384, dtype=np.float32)
        mock_model.encode.return_value = [mock_embedding]
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig(normalize=False)
        encoder = TextEncoder(cfg)
        result = encoder.encode("測試")

        mock_model.encode.assert_called_once_with(
            ["測試"],
            normalize_embeddings=False,
        )


class TestEncodeAnchors(unittest.TestCase):
    """測試 encode_anchors 函數"""

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_anchors(self, mock_sentence_transformer):
        """測試 encode_anchors 基本功能（多句子格式）"""
        mock_model = MagicMock()
        # 模擬 encode 回傳多個向量（每個領域 3 個句子）
        mock_embeddings = [
            np.array([0.1] * 384, dtype=np.float32),  # domain1, sentence1
            np.array([0.2] * 384, dtype=np.float32),  # domain1, sentence2
            np.array([0.3] * 384, dtype=np.float32),  # domain1, sentence3
            np.array([0.4] * 384, dtype=np.float32),  # domain2, sentence1
            np.array([0.5] * 384, dtype=np.float32),  # domain2, sentence2
            np.array([0.6] * 384, dtype=np.float32),  # domain2, sentence3
        ]
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig()
        encoder = TextEncoder(cfg)

        anchors = {
            "domain1": ["這是領域1的描述1", "這是領域1的描述2", "這是領域1的描述3"],
            "domain2": ["這是領域2的描述1", "這是領域2的描述2", "這是領域2的描述3"],
        }
        domains = ["domain1", "domain2"]

        result = encode_anchors(encoder, anchors, domains)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "domain1" in result
        assert "domain2" in result
        assert isinstance(result["domain1"], list)
        assert isinstance(result["domain2"], list)
        assert len(result["domain1"]) == 3
        assert len(result["domain2"]) == 3
        assert isinstance(result["domain1"][0], np.ndarray)
        assert isinstance(result["domain2"][0], np.ndarray)
        assert result["domain1"][0].dtype == np.float32
        assert result["domain2"][0].dtype == np.float32

        # 驗證 encode_many 被正確調用
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        expected_texts = [
            "這是領域1的描述1", "這是領域1的描述2", "這是領域1的描述3",
            "這是領域2的描述1", "這是領域2的描述2", "這是領域2的描述3"
        ]
        assert call_args[0][0] == expected_texts

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_anchors_with_real_domains(self, mock_sentence_transformer):
        """測試使用真實的 DOMAINS 和 DOMAIN_ANCHORS（多句子格式）"""
        mock_model = MagicMock()
        # 只測試前 3 個 domain 以加快測試速度
        test_domains = DOMAINS[:3]
        
        # 計算總句子數
        total_sentences = sum(len(DOMAIN_ANCHORS[d]) for d in test_domains)
        
        # 模擬 encode 回傳多個向量
        mock_embeddings = [
            np.array([0.1] * 384, dtype=np.float32) for _ in range(total_sentences)
        ]
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig()
        encoder = TextEncoder(cfg)

        result = encode_anchors(encoder, DOMAIN_ANCHORS, test_domains)

        assert isinstance(result, dict)
        assert len(result) == 3
        for domain in test_domains:
            assert domain in result
            assert isinstance(result[domain], list)
            assert len(result[domain]) == len(DOMAIN_ANCHORS[domain])
            assert isinstance(result[domain][0], np.ndarray)
            assert result[domain][0].dtype == np.float32

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_anchors_order(self, mock_sentence_transformer):
        """測試 encode_anchors 的順序正確性（多句子格式）"""
        mock_model = MagicMock()
        # 回傳不同值的向量以便區分（每個領域 2 個句子）
        mock_embeddings = [
            np.array([1.0] * 384, dtype=np.float32),  # domain1, sentence1
            np.array([1.1] * 384, dtype=np.float32),  # domain1, sentence2
            np.array([2.0] * 384, dtype=np.float32),  # domain2, sentence1
            np.array([2.1] * 384, dtype=np.float32),  # domain2, sentence2
            np.array([3.0] * 384, dtype=np.float32),  # domain3, sentence1
            np.array([3.1] * 384, dtype=np.float32),  # domain3, sentence2
        ]
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        cfg = EncoderConfig()
        encoder = TextEncoder(cfg)

        anchors = {
            "domain1": ["描述1-1", "描述1-2"],
            "domain2": ["描述2-1", "描述2-2"],
            "domain3": ["描述3-1", "描述3-2"],
        }
        domains = ["domain1", "domain2", "domain3"]

        result = encode_anchors(encoder, anchors, domains)

        # 驗證順序正確
        assert result["domain1"][0][0] == 1.0
        assert result["domain1"][1][0] == 1.1
        assert result["domain2"][0][0] == 2.0
        assert result["domain2"][1][0] == 2.1
        assert result["domain3"][0][0] == 3.0
        assert result["domain3"][1][0] == 3.1


def run_integration_test():
    """
    整合測試：使用真實模型（如果可用）
    注意：此測試需要實際安裝 sentence-transformers 和模型
    """
    print("\n" + "=" * 80)
    print("整合測試：使用真實模型（如果可用）")
    print("=" * 80)

    try:
        cfg = EncoderConfig(device="cpu")  # 使用 CPU 以避免 CUDA 依賴
        encoder = TextEncoder(cfg)

        # 測試單一文字編碼
        text1 = "粗大動作發展"
        emb1 = encoder.encode(text1)
        print(f"✓ encode('{text1}') 成功")
        print(f"  向量維度: {emb1.shape}")
        print(f"  向量範數: {np.linalg.norm(emb1):.6f}")

        # 測試多個文字編碼
        texts = ["粗大動作", "精細動作", "感覺統合"]
        emb_many = encoder.encode_many(texts)
        print(f"✓ encode_many({len(texts)} 個文字) 成功")
        print(f"  矩陣形狀: {emb_many.shape}")

        # 測試相似度計算
        text2 = "大肌肉運動"
        emb2 = encoder.encode(text2)
        sim = cosine_sim(emb1, emb2)
        print(f"✓ cosine_sim('{text1}', '{text2}') = {sim:.6f}")

        # 測試 encode_anchors（多句子格式）
        test_domains = DOMAINS[:3]
        anchor_embeddings = encode_anchors(encoder, DOMAIN_ANCHORS, test_domains)
        print(f"✓ encode_anchors({len(test_domains)} 個 domain) 成功")
        for domain in test_domains:
            num_sentences = len(anchor_embeddings[domain])
            avg_norm = np.mean([np.linalg.norm(vec) for vec in anchor_embeddings[domain]])
            print(f"  - {domain}: {num_sentences} 個句子, 平均向量範數 = {avg_norm:.6f}")

        print("\n✓ 所有整合測試通過！")
        return True

    except ImportError as e:
        print(f"⚠ 跳過整合測試：缺少依賴 ({e})")
        return False
    except Exception as e:
        print(f"✗ 整合測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("執行 embedding.py 單元測試")
    print("=" * 80)

    # 執行 unittest 測試
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 如果單元測試通過，執行整合測試
    if result.wasSuccessful():
        run_integration_test()
    else:
        print("\n⚠ 單元測試失敗，跳過整合測試")

