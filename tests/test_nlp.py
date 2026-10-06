"""NLP 平台单元测试。

运行：``python -m unittest discover -s tests -v``
覆盖：分词、词性、句法、NER、情感、摘要、翻译、关键词、词向量、
分片存储（含并发锁）、流水线引擎、HMM。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import (get_segmenter, get_tagger, get_parser, get_constituency_parser,
                 get_ner, get_sentiment, get_summarizer, get_translator,
                 get_keywords, get_embeddings, get_multi_summarizer, TAGSET)
from nlp.hmm import HMM
from pipeline import PipelineEngine, PipelineError
from storage import ShardedStore, StoreRegistry


class TestSegmenter(unittest.TestCase):
    def test_basic(self):
        words = get_segmenter().cut("自然语言处理是人工智能的重要分支")
        self.assertIn("自然语言", words)
        self.assertIn("人工智能", words)
        self.assertIn("是", words)

    def test_english_number(self):
        words = get_segmenter().cut("我用Python写了100行代码")
        self.assertIn("Python", words)
        self.assertIn("100", words)


class TestPOSTagger(unittest.TestCase):
    def test_tags(self):
        pairs = get_tagger().tag("我学习自然语言处理")
        self.assertTrue(pairs)
        for word, tag in pairs:
            self.assertIn(tag, TAGSET, f"{word}:{tag}")

    def test_punct_as_other(self):
        pairs = get_tagger().tag("你好，世界。")
        tags = [t for _, t in pairs]
        for t in tags:
            if t in ("，", "。"):
                continue
        # 标点词本身应为 x
        for w, t in pairs:
            if w in ("，", "。"):
                self.assertEqual(t, "x")


class TestParser(unittest.TestCase):
    def test_dependency(self):
        dep = get_parser().parse("北京大学的研究团队开发了机器学习系统")
        self.assertEqual(len(dep["words"]), len(dep["heads"]))
        self.assertIn(-1, dep["heads"])  # 存在根
        # 每个 head 都是有效下标或 -1
        for h in dep["heads"]:
            self.assertTrue(h == -1 or 0 <= h < len(dep["words"]))

    def test_constituency_spans(self):
        c = get_constituency_parser().parse("北京大学的研究团队开发了系统")
        leaves = self._leaves(c["tree"])
        self.assertEqual("北京大学的研究团队开发了系统", leaves)

    @staticmethod
    def _leaves(tree):
        if not tree.get("children"):
            return tree.get("word", "")
        return "".join(TestParser._leaves(ch) for ch in tree["children"])


class TestNER(unittest.TestCase):
    def test_known_entities(self):
        ents = get_ner().recognize("马云在北京工作")
        types = {e["text"]: e["type"] for e in ents}
        self.assertEqual(types.get("马云"), "PERSON")
        self.assertEqual(types.get("北京"), "LOCATION")

    def test_date_money(self):
        ents = get_ner().recognize("2024年10月1日花了99.9元")
        texts = [e["text"] for e in ents]
        self.assertTrue(any("2024" in t for t in texts))
        self.assertTrue(any("99.9" in t for t in texts))


class TestSentiment(unittest.TestCase):
    def test_positive(self):
        r = get_sentiment().analyze("这个产品非常好用，我很喜欢")
        self.assertEqual(r["polarity"], "positive")

    def test_negative(self):
        r = get_sentiment().analyze("服务态度很差，令人失望")
        self.assertEqual(r["polarity"], "negative")


class TestSummarizer(unittest.TestCase):
    def test_shorter(self):
        text = ("自然语言处理是人工智能的重要分支。它研究如何让计算机理解语言。"
                "分词是基础任务。词性标注是另一个任务。")
        r = get_summarizer().summarize(text, ratio=0.5)
        self.assertTrue(len(r["summary"]) < len(text))
        self.assertTrue(r["top_indices"])


class TestMultiSummarizer(unittest.TestCase):
    def setUp(self):
        self.ms = get_multi_summarizer()
        self.docs = [
            {"doc_id": "a", "title": "甲报",
             "text": "2024年5月1日，星河科技在北京发布AI芯片昇云900。该芯片算力达每秒1000万亿次，售价3万元。公司表示将于6月量产。"},
            {"doc_id": "b", "title": "乙报",
             "text": "星河科技5月1日推出昇云900芯片，算力为每秒1000万亿次。业内认为该产品将加剧行业竞争。"},
            {"doc_id": "c", "title": "丙报",
             "text": "这款芯片5月1日亮相，售价定为2.8万元。分析称供应不足，量产可能推迟至8月。星河科技总部位于深圳。"},
        ]

    def test_dedup_and_coverage(self):
        r = self.ms.synthesize(self.docs, ratio=0.9)
        # 两篇都写的「算力1000万亿次」应聚为一个事实簇
        joined = " ".join(c["rep_text"] for c in r["clusters"])
        self.assertIn("1000万亿次", joined)
        chip_clusters = [c for c in r["clusters"] if "1000万亿次" in c["rep_text"]]
        self.assertEqual(len(chip_clusters), 1)
        self.assertGreaterEqual(chip_clusters[0]["support"], 2)
        # 独有事实（深圳总部）不应丢
        self.assertTrue(any("深圳" in c["rep_text"] for c in r["clusters"]))
        # 摘要句子全部带出处引用
        self.assertTrue(r["summary"].strip())
        for s in r["sentences"]:
            self.assertTrue(s["citations"])
            self.assertTrue(all(x["text"] for x in s["sources"]))

    def test_provenance_and_no_copy(self):
        r = self.ms.synthesize(self.docs, ratio=0.9)
        for s in r["sentences"]:
            # 每个入选句必须能定位到具体文档与句序号
            primary = [x for x in s["sources"] if x["is_primary"]]
            self.assertEqual(len(primary), 1)
            self.assertIn(primary[0]["doc_id"], ("a", "b", "c"))
        # 单一来源占比不超过 60%（不整段照搬某一篇）
        n = max(len(r["sentences"]), 1)
        for count in r["stats"]["per_source"].values():
            self.assertLessEqual(count / n, 0.7)

    def test_conflict_not_merged(self):
        r = self.ms.synthesize(self.docs, ratio=0.9)
        # 3万 vs 2.8万 是数字口径冲突，必须上报并可追溯到两篇原句
        num_conflicts = [c for c in r["conflicts"] if c["kind"] == "number"]
        self.assertTrue(num_conflicts)
        cf = num_conflicts[0]
        self.assertIn("3万元", cf["claim_a"]["text"] + cf["claim_b"]["text"])
        self.assertIn("2.8万元", cf["claim_a"]["text"] + cf["claim_b"]["text"])
        self.assertNotEqual(cf["claim_a"]["doc_id"], cf["claim_b"]["doc_id"])
        # 含 3万元 的句子与含 2.8万元 的句子不在同一事实簇
        cluster_of = lambda frag: [c for c in r["clusters"]
                                   if any(frag in m["text"] for m in c["members"])]
        ids_3w = {c["id"] for c in cluster_of("3万元")}
        ids_28w = {c["id"] for c in cluster_of("2.8万元")}
        self.assertTrue(ids_3w and ids_28w and not (ids_3w & ids_28w))
        # 甲报那句顺带提及、未被乙报印证的售价，应标为簇内独有口径
        chip = next(c for c in r["clusters"]
                    if any("3万元" in m["text"] for m in c["members"]))
        self.assertIn("3万元", chip["unique_numbers"])

        # 真正的日期冲突（5月3日 vs 5月4日）也不合并
        r2 = self.ms.synthesize([
            {"doc_id": "x", "title": "X", "text": "警方5月3日通报了这起事故，3人受伤。"},
            {"doc_id": "y", "title": "Y", "text": "警方5月4日通报了这起事故，3人受伤。"},
        ], ratio=0.9)
        self.assertEqual(r2["stats"]["facts"], 2)
        self.assertTrue(any(c["kind"] == "date" for c in r2["conflicts"]))

        # 同一事实的重复表述（日期数字都一致）应当合并
        r3 = self.ms.synthesize([
            {"doc_id": "x", "title": "X", "text": "展会将于9月10日在上海开幕，预计10万人参观。"},
            {"doc_id": "y", "title": "Y", "text": "本届展会9月10日在上海开幕，参观人数预计10万人。"},
        ], ratio=0.9)
        self.assertEqual(r3["stats"]["facts"], 1)
        self.assertEqual(r3["conflicts"], [])

    def test_timeline_order(self):
        r = self.ms.synthesize(self.docs, ratio=0.7, order="timeline")
        dated = [s for s in r["sentences"] if s["date"]]
        keys = [s["date"] for s in dated]
        self.assertEqual(keys, sorted(keys))

    def test_incremental_delete_keeps_shared_facts(self):
        first = self.ms.synthesize(self.docs, ratio=0.9)
        first_ids = {c["id"] for c in first["clusters"]}
        # 删掉甲报
        second = self.ms.update(first, self.docs[1:])
        second_ids = {c["id"] for c in second["clusters"]}
        # 共享事实（甲乙共述的算力发布会）通过 ID 继承仍然保留
        chip = next(c for c in first["clusters"]
                    if "1000万亿" in c["rep_text"])
        self.assertIn(chip["id"], second_ids)
        surviving = next(c for c in second["clusters"] if c["id"] == chip["id"])
        self.assertEqual(surviving["source_doc_ids"], ["b"])
        # 引用由 [甲][乙] 收敛为 [乙]，代表句换岗，摘要不塌
        chip_sent = next(s for s in second["sentences"]
                         if s["cluster_id"] == chip["id"])
        self.assertEqual(chip_sent["citations"], [1])
        # 甲报独有的 6 月量产事实随之消失，并在差异中标出
        removed_texts = " ".join(c["text"] for c in second["changes"]["removed"])
        self.assertIn("量产", removed_texts)
        # 来源变化被记录：多源印证 2 -> 1
        attr = next(a for a in second["changes"]["attribution"]
                    if a["cluster_id"] == chip["id"])
        self.assertEqual((attr["support_before"], attr["support_after"]), (2, 1))
        self.assertEqual(attr["removed_sources"], ["a"])

    def test_incremental_add(self):
        first = self.ms.synthesize(self.docs[:2], ratio=0.9)
        second = self.ms.update(first, self.docs)
        self.assertTrue(second["changes"]["added"])
        self.assertTrue(any("深圳" in c["text"] for c in second["changes"]["added"]))

    def test_incremental_idempotent(self):
        first = self.ms.synthesize(self.docs, ratio=0.9)
        second = self.ms.update(first, self.docs)
        self.assertEqual(second["changes"]["added"], [])
        self.assertEqual(second["changes"]["removed"], [])
        self.assertEqual(second["changes"]["attribution"], [])
        # ID 保持稳定
        self.assertEqual({c["id"] for c in first["clusters"]},
                         {c["id"] for c in second["clusters"]})

    def test_empty_and_single(self):
        self.assertEqual(self.ms.synthesize([])["summary"], "")
        r = self.ms.synthesize([{"doc_id": "a", "title": "A",
                                 "text": "仅有一个事实的报道。"}])
        self.assertIn("[1]", r["summary"])


class TestTranslator(unittest.TestCase):
    def test_zh2en(self):
        r = get_translator().translate("我喜欢机器学习", "zh2en")
        self.assertIn("machine learning", r["translation"].lower())

    def test_en2zh(self):
        r = get_translator().translate("I like China", "en2zh")
        self.assertTrue(r["translation"])


class TestKeywords(unittest.TestCase):
    def test_extract(self):
        r = get_keywords().extract("自然语言处理是人工智能的重要分支", top_k=5)
        self.assertTrue(r["keywords"])
        for k in r["keywords"]:
            self.assertIn("word", k)
            self.assertIn("score", k)


class TestEmbeddings(unittest.TestCase):
    def test_train_nearest(self):
        texts = [
            "自然语言处理是人工智能的重要分支",
            "机器学习是人工智能的核心技术",
            "深度学习推动了人工智能的发展",
            "分词是自然语言处理的基础任务",
        ] * 3
        emb = get_embeddings()
        emb.train(texts, vocab_size=60, dim=8, window=3, min_count=1)
        self.assertTrue(emb.vocab)
        self.assertTrue(emb.vectors)
        # 近邻应返回词且不包含自身
        nb = emb.nearest(emb.vocab[0], k=3)
        self.assertTrue(nb)
        self.assertNotIn(emb.vocab[0], [n["word"] for n in nb])
        # 2D 投影
        proj = emb.project_2d()
        self.assertEqual(len(proj), len(emb.vectors))


class TestHMM(unittest.TestCase):
    def test_viterbi(self):
        hmm = HMM(["A", "B"], add_k=0.1)
        hmm.train([[(1, "A"), (2, "B")], [(1, "A"), (2, "B")], [(2, "B"), (1, "A")]])
        path = hmm.viterbi([1, 2])
        self.assertEqual(len(path), 2)
        self.assertIn(path[0], ("A", "B"))


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shard_insert_query(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        store.insert_many([{"v": i} for i in range(25)])
        self.assertEqual(store.stats()["total"], 25)
        self.assertEqual(store.stats()["shard_count"], 3)
        self.assertEqual(len(store.query(where=[("v", "gt", 20)])), 4)
        self.assertEqual(len(store.query(where=[("v", "in", [1, 2, 3])])), 3)

    def test_delete_compact(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        ids = store.insert_many([{"v": i} for i in range(15)])
        store.delete(ids[0])
        stats = store.compact()
        self.assertEqual(stats["records"], 14)

    def test_concurrent_insert(self):
        store = ShardedStore(self.tmp, "t", shard_size=20)
        errors = []

        def worker(offset):
            try:
                store.insert_many([{"v": offset * 1000 + i} for i in range(30)])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertEqual(store.stats()["total"], 180)

    def test_registry_tasks(self):
        reg = StoreRegistry(self.tmp)
        reg.task("a").insert({"x": 1})
        reg.task("b").insert({"x": 2})
        # 造一个非存储目录，不应被识别为任务
        os.makedirs(os.path.join(self.tmp, "models"))
        self.assertEqual(reg.tasks(), ["a", "b"])


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = PipelineEngine().register_builtin()

    def test_run_chain(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment"}, {"name": "pos"}, {"name": "sentiment"}]}
        out = self.engine.build(cfg).run({"text": "这个产品非常好用"})
        self.assertIn("words", out)
        self.assertIn("pos", out)
        self.assertIn("sentiment", out)

    def test_batch(self):
        cfg = {"name": "p", "stages": [{"name": "segment"}, {"name": "keywords"}]}
        results = self.engine.run_batch(
            cfg, ["今天天气很好", "这个产品非常好用"], max_workers=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["ok"] for r in results))

    def test_cycle_detected(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment", "deps": ["pos"]},
            {"name": "pos", "deps": ["segment"]},
        ]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)

    def test_missing_stage(self):
        cfg = {"name": "p", "stages": [{"name": "not_exist"}]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)


if __name__ == "__main__":
    unittest.main()
