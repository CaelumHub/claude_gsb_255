"""多文档融合摘要。

针对「同一件事有多篇相关报道」的场景：把若干篇内容互有重复、各有侧重的
文档融合成一篇连贯、去重、可溯源的短文。

核心流程
--------
1. **切句与溯源**：每篇文档切成句子，每句记录 ``(doc_id, sent_index)`` 来源。
2. **事实对齐（去重）**：全部句子共享一套 TF-IDF 向量，跨文档计算余弦
   相似度；相似度超过阈值、且**关键事实不冲突**的句子用并查集聚成
   「事实簇」——同一件事在多篇报道中的不同说法归为一簇，只输出一个
   代表句，重复内容自然消除。冲突检查（日期 / 数字 / 地名 / 人名双方
   均有且互不相交则判为不同事实）防止把「10月1日宁波停课」与
   「10月3日福州停课」这类同模板不同事实的句子错误合并。
3. **时间线对齐**：规则抽取绝对日期（``2026年10月5日`` / ``10月5日`` /
   ``2026-10-05``）与相对时间（今天 / 昨天 / …），相对时间以文档的
   ``created_at`` 为基准解析；簇时间取成员最早时间，输出按
   「导语 → 时间顺序 → 背景」组织。
4. **重要性打分**：TextRank（跨文档句子图）+ 位置先验 + 信息密度
   （实体 / 数字）；簇得分再乘「多篇佐证」加成——被越多独立文档报道的
   事实越应保留。
5. **选择与覆盖**：按簇得分贪心选取至预算；随后做两步「覆盖」检查——
   先补完全未被覆盖的文档，再补「仅被共享簇覆盖、独有信息未入选」的
   文档（每篇至多补一句），保证各篇独有的关键信息不丢（这也是增删
   文档后摘要不塌的关键）。
6. **轻压缩与溯源**：代表句只做「删除式」压缩（去掉句首「据悉 / 据报道」
   等引语标记与句尾记者署名），绝不跨文档拼接句子片段，因此不会把
   不同文档的事实张冠李戴；每个输出句都携带完整来源列表（哪些文档的
   哪一句报道了该事实）。

增删文档的健壮性
----------------
``fuse`` 是当前文档集合的纯函数：删除一篇文档后，仍被其它文档佐证的
事实簇自动改由存活文档的句子代表，摘要不会整段塌掉；仅被删除文档独有
的簇随之消失（该信息确实已不在语料中）。新增文档重新调用即可。
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Optional

from .ner import NERExtractor
from .segmenter import Segmenter
from .text import (compute_tfidf, filter_stopwords, pagerank, similarity,
                   split_sentences)

# ---------------------------------------------------------------------------
# 时间表达抽取
# ---------------------------------------------------------------------------

_FULL_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
_ISO_DATE_RE = re.compile(r"(\d{4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})\s*日?")
_MD_DATE_RE = re.compile(r"(?<![\d年])\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_RELATIVE_DAYS = {"前天": -2, "昨天": -1, "昨日": -1,
                  "今天": 0, "今日": 0,
                  "明天": 1, "明日": 1, "后天": 2}

# ---------------------------------------------------------------------------
# 删除式轻压缩：句首引语 / 连接词、句尾署名
# ---------------------------------------------------------------------------

_ATTRIBUTIONS = ("据悉", "据报道", "据记者了解", "记者了解到", "记者获悉",
                 "有消息称", "据了解", "据介绍", "有报道称", "报道称", "本报讯")
_CONNECTIVES = ("此外", "与此同时", "同时", "另外", "而且", "并且")
_REPORTER_RE = re.compile(r"[（(][^（）()]{0,15}(?:记者|通讯员|编辑)"
                          r"[^（）()]{0,25}[)）]\s*$")


class MultiDocFuser:
    """多文档融合摘要器。

    :param segmenter: 分词器（缺省自建）
    :param ner: 命名实体识别器（缺省自建，用于信息密度打分）
    """

    def __init__(self, segmenter: Optional[Segmenter] = None,
                 ner: Optional[NERExtractor] = None):
        self.segmenter = segmenter or Segmenter()
        self.ner = ner or NERExtractor(self.segmenter)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def fuse(self, docs: list[dict],
             max_sentences: Optional[int] = None,
             ratio: float = 0.5,
             dedup_threshold: float = 0.55,
             graph_threshold: float = 0.10,
             ensure_coverage: bool = True) -> dict:
        """把多篇相关文档融合成一篇摘要。

        :param docs: ``[{"id", "name", "text", "created_at"?}, ...]``
        :param max_sentences: 最多输出句数（缺省按 ``ratio`` 折算）
        :param ratio: 输出句数占事实簇数的比例（未指定 max_sentences 时生效）
        :param dedup_threshold: 句子判重相似度阈值（越高越保守）
        :param graph_threshold: TextRank 建边相似度阈值
        :param ensure_coverage: 覆盖保证：每篇文档至少贡献一条事实，且各篇
            独有的关键信息尽量入选（每篇至多补一句）。显式
            ``max_sentences`` 时总数至多为 ``max(max_sentences, 文档数)``；
            自动预算时每篇至多再补一句。
        """
        docs = [d for d in docs if (d.get("text") or "").strip()]
        doc_infos = [{"id": d.get("id", f"doc_{i}"),
                      "name": d.get("name", d.get("id", f"doc_{i}"))}
                     for i, d in enumerate(docs)]
        if not docs:
            return {"summary": "", "sentences": [], "docs": [], "doc_count": 0,
                    "stats": {"input_sentences": 0, "clusters": 0, "selected": 0,
                              "duplicates_removed": 0,
                              "covered_docs": [], "uncovered_docs": []}}

        sents = self._prepare(docs)
        if not sents:
            return {"summary": "", "sentences": [], "docs": doc_infos,
                    "doc_count": len(docs),
                    "stats": {"input_sentences": 0, "clusters": 0, "selected": 0,
                              "duplicates_removed": 0,
                              "covered_docs": [], "uncovered_docs": []}}

        # -- 相似度（倒排索引预筛，避免全配对） ----------------------------
        vectors = compute_tfidf([s["tokens"] for s in sents])
        sims = self._pairwise_sims(sents, vectors, graph_threshold)

        # -- TextRank 句子重要度 -----------------------------------------
        graph: dict[int, dict[int, float]] = {i: {} for i in range(len(sents))}
        for (i, j), sim in sims.items():
            graph[i][j] = sim
            graph[j][i] = sim
        ranks = pagerank(graph)
        rank_max = max(ranks.values()) if ranks else 1.0
        for gid, s in enumerate(sents):
            rank_norm = ranks.get(gid, 0.0) / (rank_max or 1.0)
            pos = math.exp(-s["sent_index"] / max(1, s["doc_sent_count"]) * 2)
            info = min(1.0, s["n_entities"] / 3.0)
            s["score"] = 0.55 * rank_norm + 0.25 * pos + 0.20 * info

        # -- 事实簇（并查集） ---------------------------------------------
        clusters = self._cluster(sents, sims, dedup_threshold)

        # -- 选择 + 覆盖 ---------------------------------------------------
        budget = max_sentences if max_sentences else \
            max(1, round(len(clusters) * ratio))
        budget = max(1, min(budget, len(clusters)))
        by_score = sorted(clusters, key=lambda c: (-c["score"], c["first_gid"]))
        selected = list(by_score[:budget])
        if ensure_coverage:
            # 覆盖保证分两步：
            # 1) 完全未被覆盖的文档优先补其最佳簇（保证每篇至少一条）；
            # 2) 仅被共享簇覆盖、其「独有簇」未入选的文档，补最佳独有簇
            #    （保证各篇独有的关键信息不丢）。
            # 每篇至多补一句；显式 max_sentences 时总数不超过
            # max(budget, 文档数)，自动预算时每篇至多再补一句。
            explicit = bool(max_sentences)
            cap = (max(budget, len(doc_infos)) if explicit
                   else budget + len(doc_infos))
            chosen_ids = {c["cid"] for c in selected}
            covered = {d for c in selected for d in c["docs"]}

            def _add_best(pool) -> bool:
                pool = [c for c in pool if c["cid"] not in chosen_ids]
                if not pool or len(selected) >= cap:
                    return False
                best = max(pool, key=lambda c: c["score"])
                selected.append(best)
                chosen_ids.add(best["cid"])
                covered.update(best["docs"])
                return True

            for info in doc_infos:  # 第一优先：完全未覆盖的文档
                if info["id"] not in covered:
                    _add_best([c for c in clusters if info["id"] in c["docs"]])
            for info in doc_infos:  # 其次：独有信息未入选的文档
                did = info["id"]
                unique = [c for c in clusters if c["docs"] == [did]]
                if unique and not any(c["cid"] in chosen_ids for c in unique):
                    _add_best(unique)

        # -- 排序：导语（最高分）→ 时间线 → 背景 ---------------------------
        lead = max(selected, key=lambda c: c["score"])
        rest = [c for c in selected if c is not lead]
        timed = sorted((c for c in rest if c["time"] is not None),
                       key=lambda c: (c["time"], -c["score"]))
        untimed = sorted((c for c in rest if c["time"] is None),
                         key=lambda c: -c["score"])
        ordered = [lead] + timed + untimed

        # -- 生成输出句 -----------------------------------------------------
        out_sentences = []
        for idx, cluster in enumerate(ordered):
            rep = sents[cluster["rep"]]
            text, compressed = self._compress(rep["text"])
            sources = []
            for gid in cluster["members"]:
                member = sents[gid]
                sources.append({
                    "doc_id": member["doc_id"],
                    "doc_name": member["doc_name"],
                    "sent_index": member["sent_index"],
                    "similarity": round(self._sim_to_rep(sims, gid, cluster["rep"]), 3),
                })
            sources.sort(key=lambda s: (s["doc_id"], s["sent_index"]))
            out_sentences.append({
                "index": idx,
                "text": text,
                "original": rep["text"],
                "doc_id": rep["doc_id"],
                "doc_name": rep["doc_name"],
                "sent_index": rep["sent_index"],
                "time": cluster["time"].isoformat() if cluster["time"] else None,
                "score": round(cluster["score"], 4),
                "compressed": compressed,
                "sources": sources,
            })

        summary = "".join(s["text"] + "。" for s in out_sentences)
        covered_docs = sorted({d for c in selected for d in c["docs"]},
                              key=lambda d: [i["id"] for i in doc_infos].index(d))
        return {
            "summary": summary,
            "sentences": out_sentences,
            "docs": doc_infos,
            "doc_count": len(docs),
            "stats": {
                "input_sentences": len(sents),
                "clusters": len(clusters),
                "selected": len(out_sentences),
                "duplicates_removed": len(sents) - len(clusters),
                "covered_docs": covered_docs,
                "uncovered_docs": [i["id"] for i in doc_infos
                                   if i["id"] not in covered_docs],
            },
        }

    # ------------------------------------------------------------------
    # 句子准备
    # ------------------------------------------------------------------
    def _prepare(self, docs: list[dict]) -> list[dict]:
        sents: list[dict] = []
        for doc_idx, doc in enumerate(docs):
            doc_id = doc.get("id", f"doc_{doc_idx}")
            doc_name = doc.get("name", doc_id)
            ref = self._ref_date(doc.get("created_at"))
            doc_sents = split_sentences(doc["text"])
            for sent_idx, text in enumerate(doc_sents):
                tokens = filter_stopwords(self.segmenter.cut(text))
                if not tokens:
                    continue
                entities = self.ner.recognize(text)
                sents.append({
                    "gid": len(sents),
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "doc_index": doc_idx,
                    "sent_index": sent_idx,
                    "doc_sent_count": len(doc_sents),
                    "text": text,
                    "tokens": tokens,
                    "times": self._extract_times(text, ref),
                    "n_entities": len(entities),
                    # 关键事实要素：用于聚类前的冲突检查（防张冠李戴）
                    "numbers": {e["text"] for e in entities
                                if e["type"] in ("NUMBER", "MONEY", "PERCENT")},
                    "locations": {e["text"] for e in entities
                                  if e["type"] == "LOCATION"},
                    "persons": {e["text"] for e in entities
                                if e["type"] == "PERSON"},
                })
        return sents

    @staticmethod
    def _ref_date(created_at) -> Optional[date]:
        if not created_at:
            return None
        try:
            return date.fromtimestamp(float(created_at))
        except (OverflowError, OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _extract_times(text: str, ref: Optional[date]) -> list[date]:
        """抽取句子中的时间表达，统一解析为 ``date``（无法解析的忽略）。"""
        found: set[date] = set()
        for m in _FULL_DATE_RE.finditer(text):
            try:
                found.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
        for m in _ISO_DATE_RE.finditer(text):
            try:
                found.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
        if ref is not None:
            for m in _MD_DATE_RE.finditer(text):
                try:
                    found.add(date(ref.year, int(m.group(1)), int(m.group(2))))
                except ValueError:
                    pass
            for word, offset in _RELATIVE_DAYS.items():
                if word in text:
                    found.add(ref + timedelta(days=offset))
        return sorted(found)

    # ------------------------------------------------------------------
    # 相似度与聚类
    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_sims(sents: list[dict], vectors: list[dict],
                       threshold: float) -> dict[tuple[int, int], float]:
        """计算句子两两相似度（只保留 >= threshold 的）。

        用词 → 句子的倒排索引预筛候选对，避免 O(n²) 全配对计算。
        """
        inverted: dict[str, list[int]] = {}
        for gid, s in enumerate(sents):
            for word in set(s["tokens"]):
                inverted.setdefault(word, []).append(gid)

        candidates: set[tuple[int, int]] = set()
        for gids in inverted.values():
            for a in range(len(gids)):
                for b in range(a + 1, len(gids)):
                    candidates.add((gids[a], gids[b]))

        sims: dict[tuple[int, int], float] = {}
        for i, j in candidates:
            sim = similarity(vectors[i], vectors[j])
            if sim >= threshold:
                sims[(i, j)] = sim
        return sims

    @staticmethod
    def _compatible(a: dict, b: dict) -> bool:
        """关键事实一致性检查：双方均有且互不相交 → 不同事实，禁止合并。

        防止把同模板但日期 / 数字 / 地名 / 人名不同的句子错误判重
        （如「10月1日宁波停课」与「10月3日福州停课」）。
        只有一方具备该要素时不冲突（信息多寡不同不等于矛盾）。
        """
        for key in ("numbers", "locations", "persons"):
            sa, sb = a[key], b[key]
            if sa and sb and sa.isdisjoint(sb):
                return False
        ta, tb = set(a["times"]), set(b["times"])
        if ta and tb and ta.isdisjoint(tb):
            return False
        return True

    @staticmethod
    def _cluster(sents: list[dict], sims: dict[tuple[int, int], float],
                 threshold: float) -> list[dict]:
        """把相似度 >= threshold 且事实不冲突的句子用并查集聚成事实簇。"""
        parent = list(range(len(sents)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for (i, j), sim in sorted(sims.items(), key=lambda kv: -kv[1]):
            if sim >= threshold and MultiDocFuser._compatible(sents[i], sents[j]):
                union(i, j)

        groups: dict[int, list[int]] = {}
        for gid in range(len(sents)):
            groups.setdefault(find(gid), []).append(gid)

        clusters = []
        for cid, members in enumerate(sorted(groups.values(),
                                             key=lambda g: min(g))):
            members = sorted(members)
            rep = max(members,
                      key=lambda g: (sents[g]["score"],
                                     len(sents[g]["tokens"]), -g))
            doc_ids = sorted({sents[g]["doc_id"] for g in members})
            # 簇时间优先取代表句自己的时间（溯源精确到句），
            # 代表句无时间时退化为成员最早时间
            rep_times = sents[rep]["times"]
            all_times = [t for g in members for t in sents[g]["times"]]
            if rep_times:
                cluster_time = min(rep_times)
            elif all_times:
                cluster_time = min(all_times)
            else:
                cluster_time = None
            support = len(doc_ids)
            clusters.append({
                "cid": cid,
                "members": members,
                "rep": rep,
                "docs": doc_ids,
                "support": support,
                "time": cluster_time,
                "first_gid": members[0],
                "score": sents[rep]["score"] * (1.0 + 0.25 * math.log(support)),
            })
        return clusters

    @staticmethod
    def _sim_to_rep(sims: dict[tuple[int, int], float], gid: int, rep: int) -> float:
        if gid == rep:
            return 1.0
        key = (min(gid, rep), max(gid, rep))
        return sims.get(key, 0.0)

    # ------------------------------------------------------------------
    # 删除式轻压缩
    # ------------------------------------------------------------------
    @staticmethod
    def _compress(text: str) -> tuple[str, bool]:
        """去掉句首引语 / 连接词与句尾署名；只删不增，绝不跨句拼接。"""
        out = text.strip()
        for _ in range(2):  # 「此外，据悉，…」这类叠用最多剥两层
            stripped = False
            for prefix in _ATTRIBUTIONS + _CONNECTIVES:
                if out.startswith(prefix):
                    out = out[len(prefix):].lstrip("，,、：: ")
                    stripped = True
                    break
            if not stripped:
                break
        out = _REPORTER_RE.sub("", out).strip()
        if not out:
            return text, False
        return out, out != text.strip()
