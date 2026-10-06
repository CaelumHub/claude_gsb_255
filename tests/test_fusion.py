"""多文档融合摘要单元测试。

运行：``python -m unittest tests.test_fusion -v``
覆盖：跨文档去重、独有信息保留、逐句溯源、不跨文档拼接、时间线对齐、
增删文档健壮性、轻压缩、覆盖保证、确定性。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import get_fuser
from nlp.text import split_sentences
from storage import StoreRegistry

TS = datetime(2026, 10, 5, 12).timestamp()

DOC_A = {"id": "A", "name": "新华快讯", "created_at": TS,
         "text": "10月3日，我市遭遇强降雨，城区多处积水。"
                 "市应急管理局启动防汛Ⅲ级应急响应。"
                 "交警部门封闭了5处积水严重的路段。"
                 "市长赶赴现场指挥抢险。"}
DOC_B = {"id": "B", "name": "晚报", "created_at": TS,
         "text": "10月3日，我市遭遇强降雨，城区积水严重。"
                 "市气象台发布暴雨橙色预警。"
                 "交警部门封闭了5处积水严重的路段，提醒市民绕行。"
                 "地铁2号线部分站点临时关闭。"}
DOC_C = {"id": "C", "name": "晨报", "created_at": TS,
         "text": "10月4日，城区积水基本消退，地铁2号线恢复正常运营。"
                 "市应急管理局将应急响应调整为Ⅳ级。"
                 "据统计，此次降雨未造成人员伤亡。"}


def fuse(docs, **kw):
    return get_fuser().fuse(docs, **kw)


class TestDedup(unittest.TestCase):
    """重复内容只保留一次，且能说出每句来自哪篇。"""

    def test_shared_fact_appears_once(self):
        r = fuse([DOC_A, DOC_B, DOC_C])
        hits = [s for s in r["sentences"] if "交警部门" in s["original"]]
        self.assertEqual(len(hits), 1, "重复报道的事实只应出现一次")
        self.assertEqual({src["doc_id"] for src in hits[0]["sources"]}, {"A", "B"})

    def test_duplicates_removed_stat(self):
        r = fuse([DOC_A, DOC_B, DOC_C])
        st = r["stats"]
        self.assertEqual(st["input_sentences"], 11)
        self.assertEqual(st["clusters"], 9)
        self.assertEqual(st["duplicates_removed"], 2)
        self.assertEqual(st["selected"], len(r["sentences"]))

    def test_highly_redundant_docs(self):
        # 两篇几乎一样的文档：摘要不应翻倍
        dup = {"id": "B2", "name": "转载", "created_at": TS, "text": DOC_B["text"]}
        r1 = fuse([DOC_B])
        r2 = fuse([DOC_B, dup])
        self.assertEqual(len(r2["sentences"]), len(r1["sentences"]))


class TestUniqueInfoKept(unittest.TestCase):
    """各篇独有且重要的信息必须保留。"""

    def test_unique_facts_from_each_doc(self):
        r = fuse([DOC_A, DOC_B, DOC_C])
        summary = r["summary"]
        self.assertIn("Ⅲ级应急响应", summary)   # 仅 A 有
        self.assertIn("暴雨橙色预警", summary)     # 仅 B 有
        self.assertIn("Ⅳ级", summary)              # 仅 C 有

    def test_unique_fact_attributed_to_right_doc(self):
        r = fuse([DOC_A, DOC_B, DOC_C])
        for s in r["sentences"]:
            if "暴雨橙色预警" in s["original"]:
                self.assertEqual(s["doc_id"], "B")
                self.assertEqual([x["doc_id"] for x in s["sources"]], ["B"])

    def test_coverage_with_tight_budget(self):
        # 预算小于文档数时，覆盖保证仍让每篇的关键信息入选
        r = fuse([DOC_A, DOC_B, DOC_C], max_sentences=2)
        self.assertEqual(r["stats"]["uncovered_docs"], [])
        self.assertEqual(set(r["stats"]["covered_docs"]), {"A", "B", "C"})


class TestProvenance(unittest.TestCase):
    """溯源：每句都能指回来源文档的具体句子，且不跨文档拼接。"""

    def test_sentence_points_back_to_source(self):
        docs = [DOC_A, DOC_B, DOC_C]
        by_id = {d["id"]: d for d in docs}
        r = fuse(docs)
        for s in r["sentences"]:
            src_sents = split_sentences(by_id[s["doc_id"]]["text"])
            self.assertEqual(src_sents[s["sent_index"]], s["original"],
                             "doc_id + sent_index 必须能定位原句")
            self.assertTrue(s["sources"])
            for src in s["sources"]:
                self.assertIn(src["doc_id"], by_id)

    def test_no_cross_doc_splicing(self):
        # 输出句必须逐字来自某一篇（压缩只允许删除，不允许拼接）
        r = fuse([DOC_A, DOC_B, DOC_C])
        for s in r["sentences"]:
            self.assertIn(s["text"], s["original"],
                          "压缩后的句子必须是原句的连续子串（只删不增）")

    def test_no_wholesale_paragraph_copy(self):
        docs = [DOC_A, DOC_B, DOC_C]
        r = fuse(docs)
        for d in docs:
            self.assertNotEqual(r["summary"], d["text"])
            self.assertNotIn(r["summary"], d["text"])
        # 多源交织：摘要句来自至少两篇文档
        self.assertGreaterEqual(len({s["doc_id"] for s in r["sentences"]}), 2)


class TestTimeline(unittest.TestCase):
    def test_chronological_order(self):
        r = fuse([DOC_A, DOC_B, DOC_C])
        times = [s["time"] for s in r["sentences"] if s["time"]]
        self.assertEqual(times, sorted(times))
        self.assertIn("2026-10-03", times)
        self.assertIn("2026-10-04", times)

    def test_time_extraction_rules(self):
        fuser = get_fuser()
        ref = datetime(2026, 10, 5).date()
        times = fuser._extract_times("2026年10月3日下暴雨", None)
        self.assertEqual([t.isoformat() for t in times], ["2026-10-03"])
        times = fuser._extract_times("10月4日恢复通行", ref)
        self.assertEqual([t.isoformat() for t in times], ["2026-10-04"])
        times = fuser._extract_times("昨天开始限行", ref)
        self.assertEqual([t.isoformat() for t in times], ["2026-10-04"])
        # 无参考时间时，「10月4日」与「昨天」不解析；非法日期忽略
        self.assertEqual(fuser._extract_times("10月4日恢复通行", None), [])
        self.assertEqual(fuser._extract_times("2026年13月40日", None), [])


class TestRobustness(unittest.TestCase):
    """文档增删后摘要仍连贯、不丢关键信息、不整段塌掉。"""

    def test_doc_removal_keeps_shared_facts(self):
        r = fuse([DOC_A, DOC_C])  # 删掉 B
        self.assertIn("交警部门", r["summary"])      # 共享事实由 A 继续承载
        self.assertNotIn("橙色预警", r["summary"])   # B 独有事实随之消失
        self.assertGreaterEqual(len(r["sentences"]), 2)

    def test_doc_removal_not_collapse(self):
        full = fuse([DOC_A, DOC_B, DOC_C])
        shrunk = fuse([DOC_B])
        self.assertTrue(shrunk["summary"])
        self.assertTrue(all(s["doc_id"] == "B" for s in shrunk["sentences"]))
        self.assertLessEqual(len(shrunk["sentences"]), len(full["sentences"]))
        # 预算放宽时，B 独有的关键信息必须还在（不随其它文档消失）
        roomy = fuse([DOC_B], max_sentences=4)
        self.assertIn("暴雨橙色预警", roomy["summary"])

    def test_doc_addition_merges_in(self):
        r1 = fuse([DOC_A])
        r2 = fuse([DOC_A, DOC_B])
        self.assertNotIn("橙色预警", r1["summary"])
        self.assertIn("橙色预警", r2["summary"])
        # 共享事实仍然只出现一次
        hits = [s for s in r2["sentences"] if "交警部门" in s["original"]]
        self.assertEqual(len(hits), 1)

    def test_single_doc(self):
        r = fuse([DOC_A])
        self.assertTrue(r["summary"])
        self.assertTrue(all(s["doc_id"] == "A" for s in r["sentences"]))

    def test_empty_and_blank(self):
        r = fuse([])
        self.assertEqual(r["summary"], "")
        self.assertEqual(r["sentences"], [])
        r = fuse([{"id": "x", "name": "空", "text": "   "}, DOC_A])
        self.assertTrue(r["summary"])
        self.assertEqual(r["doc_count"], 1)

    def test_deterministic(self):
        self.assertEqual(fuse([DOC_A, DOC_B, DOC_C])["summary"],
                         fuse([DOC_A, DOC_B, DOC_C])["summary"])


class TestCompression(unittest.TestCase):
    def test_attribution_and_connective_stripped(self):
        doc = {"id": "D", "name": "通讯社", "created_at": TS,
               "text": "据悉，10月5日，市政府召开防汛工作总结会。"
                       "此外，据统计，此次降雨未造成人员伤亡。（记者 张三）"}
        r = fuse([doc])
        texts = [s["text"] for s in r["sentences"]]
        self.assertTrue(any(t.startswith("10月5日") for t in texts))
        self.assertFalse(any(t.startswith("据悉") for t in texts))
        self.assertFalse(any(t.startswith("此外") for t in texts))
        self.assertFalse(any("记者" in t for t in texts))
        for s in r["sentences"]:
            self.assertTrue(s["compressed"])
            self.assertIn(s["text"], s["original"])


class TestFactConflictGuard(unittest.TestCase):
    """同模板但关键事实不同的句子不得合并（防张冠李戴）。"""

    DOCS = [
        {"id": "E1", "name": "快讯一", "created_at": TS,
         "text": "10月1日，宁波中小学停课。台风外围云系影响我省。"},
        {"id": "E2", "name": "快讯二", "created_at": TS,
         "text": "10月3日，福州中小学停课。10月3日，福州转移群众7870人。"},
        {"id": "E3", "name": "快讯三", "created_at": TS,
         "text": "10月3日，厦门转移群众2051人。沿海风浪较大。"},
    ]

    def test_different_dates_not_merged(self):
        r = fuse(self.DOCS, max_sentences=10)
        hits = [s for s in r["sentences"] if "停课" in s["original"]]
        # 宁波 10月1日 与 福州 10月3日 是两个事实，都必须保留
        self.assertEqual(len(hits), 2)
        by_text = {s["original"]: s for s in hits}
        for s in hits:
            if "宁波" in s["original"]:
                self.assertEqual(s["time"], "2026-10-01")
            if "福州" in s["original"]:
                self.assertEqual(s["time"], "2026-10-03")

    def test_different_numbers_not_merged(self):
        r = fuse(self.DOCS, max_sentences=10)
        hits = [s for s in r["sentences"] if "转移群众" in s["original"]]
        # 7870 人与 2051 人是两个事实
        self.assertEqual(len(hits), 2)
        self.assertEqual({s["doc_id"] for s in hits}, {"E2", "E3"})

    def test_time_label_matches_representative(self):
        # 输出句的时间标签必须与其来源句自身的时间一致
        r = fuse(self.DOCS, max_sentences=10)
        for s in r["sentences"]:
            if s["time"]:
                self.assertIn(s["time"].split("-")[1].lstrip("0") + "月", s["original"])


class TestShardedCorpusFusion(unittest.TestCase):
    """语料分片存放：融合直接读分片存储，跨分片对齐，结果可持久化。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # shard_size=2 强制多篇文档落到不同分片
        self.registry = StoreRegistry(self.tmp, shard_size=2)

    def test_fuse_across_shards_and_persist(self):
        store = self.registry.task("corpus")
        docs = [DOC_A, DOC_B, DOC_C,
                {"id": None, "name": "补充", "created_at": TS,
                 "text": "10月5日，市政府召开防汛工作总结会。"},
                {"id": None, "name": "转载", "created_at": TS, "text": DOC_A["text"]}]
        ids = [store.insert({"name": d["name"], "text": d["text"],
                             "created_at": d["created_at"]}) for d in docs]
        self.assertGreater(store.stats()["shard_count"], 1, "应跨多个分片")

        # 与 API 相同的取数路径：store.all() 合并各分片
        records = [r for r in store.all() if not r.get("_deleted")]
        self.assertEqual(len(records), 5)
        fused = get_fuser().fuse(
            [{"id": r["id"], "name": r["name"], "text": r["text"],
              "created_at": r["created_at"]} for r in records])
        self.assertTrue(fused["summary"])
        self.assertEqual(fused["stats"]["uncovered_docs"], [])
        # 转载篇与原文重复，不应让摘要句数翻倍
        self.assertLessEqual(fused["stats"]["selected"],
                             fused["stats"]["clusters"])

        # 融合结果写入分片存储并可按 id 读回（API 的持久化路径）
        rid = self.registry.task("fusion").insert({
            "type": "fusion", "doc_ids": ids, "result": fused})
        back = self.registry.task("fusion").get(rid)
        self.assertEqual(back["result"]["summary"], fused["summary"])
        self.assertEqual(back["doc_ids"], ids)

    def test_delete_doc_then_refuse(self):
        store = self.registry.task("corpus")
        ids = [store.insert({"name": d["name"], "text": d["text"],
                             "created_at": d["created_at"]})
               for d in (DOC_A, DOC_B, DOC_C)]
        store.delete(ids[1])  # 删除 B
        records = [r for r in store.all() if not r.get("_deleted")]
        fused = get_fuser().fuse(
            [{"id": r["id"], "name": r["name"], "text": r["text"]} for r in records])
        self.assertNotIn("橙色预警", fused["summary"])   # B 独有信息消失
        self.assertIn("交警部门", fused["summary"])      # 共享事实不塌


if __name__ == "__main__":
    unittest.main()
