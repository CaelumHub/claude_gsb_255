"""多文档事实对齐与合成摘要。

解决的问题
----------
同一事件常有多篇报道：内容互相重复、各有侧重、甚至口径冲突。逐篇阅读
成本高，而简单拼接又会大段照搬某一篇。本模块在**不生成任何新事实**的
前提下做跨文档的抽取式融合（extractive fusion）：

1. **切句建单元**：每篇报道切句，分词、去停用词、计算 TF-IDF 向量，
   同时抽取句中的日期与数字（口径）信息。
2. **跨文档事实对齐（去重）**：用平均链接凝聚聚类把不同文档中表述同一
   事实的句子聚成一个「事实簇」。聚类除了要求向量相似度，还要求数字 /
   日期口径**相容**——口径冲突的句子不会被合并，而是作为「分歧」上报，
   从根本上防止张冠李戴。
3. **重要度排序**：事实簇之间构建相似度图跑 PageRank（主题中心性），
   叠加多源印证（support）、导语位置、信息具体度（含数字 / 日期）等
   信号。
4. **预算内选择**：MMR 去冗余 + 单篇占比上限（避免整段照搬某一篇）+
   关键事实保底（含数字 / 日期的独有事实优先占名额）。
5. **逻辑组织**：支持逻辑主线（主题贪心游走）、时间线、按来源三种
   排序，并自动分段；每个入选句都带来源引用标记 ``[1][2]``。
6. **增量更新**：事实簇以内容指纹为稳定 ID。增删文档后重新合成为纯
   函数式计算：共享事实不因某篇被删而消失（代表句自动换岗），并输出
   新增 / 消失 / 来源变化的差异说明。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from collections import Counter

from .segmenter import Segmenter
from .text import (compute_tfidf, filter_stopwords, pagerank, similarity,
                   split_sentences)


# 句子结束标点
_ENDINGS = "。！？!?；;"

# 日期：2024年5月1日 / 2024-5-1 / 2024/05/01
_FULL_DATE_RE = re.compile(r"(\d{4})\s*[年/\-](\d{1,2})\s*[月/\-](\d{1,2})\s*日?")
_YEAR_MONTH_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d)")
_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月(?!\s*\d)")
# 显式数字单位（更具体 / 更长的单位放在前面优先匹配，
# 避免「10万人」截成「10万」、「1000万亿次」截成「1000万」）
_NUMBER_UNITS = ["个百分点", "万亿元", "万元", "亿元", "万亿", "千亿", "百亿",
                 "十亿", "万次", "亿次", "万人次", "亿人次", "万台", "亿台",
                 "万公里", "亿吨", "万人", "亿人", "万家", "亿家", "万套",
                 "亿美元", "万美元",
                 "公里", "美元", "欧元", "港元", "港币", "日元", "英镑",
                 "倍", "次", "人", "家", "台", "套", "米", "吨",
                 "万", "亿", "元", "%", "％"]
_NUMBER_RE = re.compile(
    r"百分之[零一二三四五六七八九十百千两]+"
    r"|\d+(?:\.\d+)?\s*(?:" + "|".join(_NUMBER_UNITS) + r")"
    r"|\d+(?:\.\d+)?")


@dataclass
class _Unit:
    """一个句子单元，携带出处与特征。"""

    uid: int
    doc_idx: int
    doc_id: str
    doc_title: str
    sent_index: int
    text: str
    words: list[str]
    pos: float                      # 句内位置 0(首句)~1(末句)
    vec: dict[str, float] = field(default_factory=dict)
    dates: set[str] = field(default_factory=set)
    date_keys: set[tuple] = field(default_factory=set)
    nums: set[str] = field(default_factory=set)
    lead: float = 0.0
    score: float = 0.0
    cluster: int = -1


def _extract_dates(text: str) -> tuple[set[str], set[tuple], list[tuple[int, int]]]:
    """抽取可排序的日期，返回 (规范标签集合, 排序键集合, 字符区间)。

    排序键统一为 ``(year, month, day)``，缺年用 0 占位；
    「5月1日」与「2024年5月1日」在比较时视为兼容（年可缺省）。
    """
    canon: set[str] = set()
    keys: set[tuple] = set()
    spans: list[tuple[int, int]] = []
    for m in _FULL_DATE_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            canon.add(f"{y:04d}-{mo:02d}-{d:02d}")
            keys.add((y, mo, d))
            spans.append(m.span())
    for m in _YEAR_MONTH_RE.finditer(text):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            canon.add(f"{y:04d}-{mo:02d}")
            keys.add((y, mo, 0))
            spans.append(m.span())
    for m in _MONTH_DAY_RE.finditer(text):
        mo, d = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            canon.add(f"{mo:02d}-{d:02d}")
            keys.add((0, mo, d))  # 无年份，可与带年份的同日兼容
            spans.append(m.span())
    for m in _MONTH_RE.finditer(text):
        mo = int(m.group(1))
        s, e = m.span()
        if 1 <= mo <= 12 and not any(ms <= s < me for ms, me in spans):
            canon.add(f"{mo:02d}月")
            keys.add((0, mo, 0))
            spans.append((s, e))
    return canon, keys, spans


def _normalize_num(raw: str) -> str:
    return re.sub(r"\s+", "", raw).replace("％", "%")


def _extract_numbers(text: str, date_spans: list[tuple[int, int]]) -> set[str]:
    """抽取数字口径，跳过落在日期区间内的数字。"""
    nums: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        if any(s <= m.start() < e for s, e in date_spans):
            continue
        token = _normalize_num(m.group())
        if token not in ("", "."):
            nums.add(token)
    return nums


def _dates_compatible(a_keys: set[tuple], b_keys: set[tuple]) -> bool:
    """两组日期是否相容（可缺省年份：``(0,5,1)`` 与 ``(2024,5,1)`` 视为同日）。"""
    for ya, ma, da in a_keys:
        for yb, mb, db in b_keys:
            if ma != mb:
                continue
            if da and db and da != db:
                continue
            if ya and yb and ya != yb:
                continue
            return True
    return False


def _compatible(a: _Unit, b: _Unit) -> Optional[str]:
    """判断两个句子的事实口径是否相容。

    双方都带日期 / 数字却完全对不上时，视为不同事实（返回冲突类型），
    不允许合并，避免把 A 文档的数字安到 B 文档的事实上。
    """
    if a.dates and b.dates and not _dates_compatible(a.date_keys, b.date_keys):
        return "date"
    if a.nums and b.nums and not (a.nums & b.nums) \
            and _has_number_clash(a.nums | b.nums):
        return "number"
    return None


def _number_values(nums: set[str]) -> dict[str, set[str]]:
    """把数字口径按**显式量词**分组：``{量词: {数值}}``。

    只有 ``3万元``、``50亿元`` 这类带明确单位的才进入量词分组；
    裸数字（``1000``、``900`` 及被宽松切出的 ``1000万`` 片段）归到
    空串 ``""``，不参与同量词口径比较，避免误判冲突。
    """
    explicit_units = {
        "%", "％", "个百分点", "万元", "亿元", "万亿元", "万美元", "亿美元",
        "元", "美元", "欧元", "港元", "港币", "日元", "英镑",
        "万人", "亿人", "万台", "亿台", "万家", "亿家", "万套",
        "万次", "亿次", "万人次", "亿人次",
        "倍", "人", "家", "台", "套", "公里", "米", "吨", "次",
    }
    by_unit: dict[str, set[str]] = {}
    for token in nums:
        m = re.match(r"^([0-9.]+)(.*)$", token)
        if m and m.group(2) in explicit_units:
            by_unit.setdefault(m.group(2), set()).add(m.group(1))
    return by_unit


def _has_number_clash(nums: set[str]) -> bool:
    """存在两个同显式量词、不同数值的数字时才算数字口径冲突。"""
    return any(len(values) > 1
               for values in _number_values(nums).values())


def _number_clash_pairs(a_nums: set[str], b_nums: set[str]) -> bool:
    """两句在同一显式量词上给出不同数值（如「3万元」对「2.8万元」）。"""
    for unit, b_values in _number_values(b_nums).items():
        a_values = _number_values(a_nums).get(unit)
        if a_values is not None and a_values != b_values:
            return True
    return False


def _ensure_period(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in _ENDINGS:
        text += "。"
    return text


class MultiDocumentSummarizer:
    """多文档抽取式合成摘要器。"""

    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()

    # -- 对外接口 ---------------------------------------------------------
    def synthesize(self, documents: list[dict], *, ratio: float = 0.35,
                   max_sentences: Optional[int] = None,
                   order: str = "logic",
                   dedup_threshold: float = 0.4,
                   max_source_share: float = 0.5) -> dict:
        """把多篇文档合成一篇带来源标注的连贯摘要。

        :param documents: ``[{"doc_id", "title", "text", "published_at"?}]``
        :param ratio: 入选事实占「去重后事实簇」的比例
        :param max_sentences: 入选句数硬上限（缺省按 ratio 并夹在 3~12）
        :param order: ``logic`` 逻辑主线 / ``timeline`` 时间线 / ``source`` 按来源
        :param dedup_threshold: 跨文档去重相似度阈值（平均链接）
        :param max_source_share: 单一来源最多占入选句的比例（防整段照搬）
        """
        params = {"ratio": ratio, "max_sentences": max_sentences, "order": order,
                  "dedup_threshold": dedup_threshold,
                  "max_source_share": max_source_share}

        docs = self._normalize_documents(documents)
        sources = [{"index": i + 1, "doc_id": d["doc_id"], "title": d["title"],
                    "published_at": d.get("published_at")}
                   for i, d in enumerate(docs)]
        cite_index = {d["doc_id"]: i + 1 for i, d in enumerate(docs)}

        units = self._build_units(docs)
        if not units:
            return {"summary": "", "paragraphs": [], "sentences": [],
                    "clusters": [], "conflicts": [], "sources": sources,
                    "stats": {"documents": len(docs), "sentences": 0,
                              "facts": 0, "selected": 0, "redundancy_rate": 0.0,
                              "per_source": {}},
                    "params": params}

        conflicts_raw = self._find_conflicts(units)
        groups = self._cluster_units(units, dedup_threshold)
        clusters = self._build_clusters(units, groups)
        conflicts = self._format_conflicts(units, clusters, conflicts_raw)
        self._score_clusters(clusters)

        n_facts = len(clusters)
        if max_sentences is None:
            budget = max(3, int(n_facts * ratio))
            budget = min(budget, 12, n_facts)
        else:
            budget = min(max(1, int(max_sentences)), n_facts)

        chosen = self._select_clusters(clusters, budget, max_source_share)
        ordered = self._order_clusters(chosen, order)
        paragraphs = self._paragraphize(ordered, order)
        summary = self._render(paragraphs, cite_index)

        selected_ids = {c["id"] for c in chosen}
        per_source = self._per_source_counts(chosen)
        sentences = self._build_sentence_view(chosen, cite_index)

        return {
            "summary": summary,
            "paragraphs": [[c["id"] for c in para] for para in paragraphs],
            "sentences": sentences,
            "clusters": [self._cluster_view(c, c["id"] in selected_ids, cite_index)
                         for c in clusters],
            "conflicts": conflicts,
            "sources": sources,
            "stats": {
                "documents": len(docs),
                "sentences": len(units),
                "facts": n_facts,
                "selected": len(chosen),
                # 被对齐掉的重复句占比
                "redundancy_rate": round(1 - n_facts / len(units), 3),
                "per_source": {cite_index[k]: v for k, v in per_source.items()},
            },
            "params": params,
        }

    def update(self, previous: dict, documents: list[dict]) -> dict:
        """文档增减后的增量再合成。

        重新对当前文档集合做一次纯函数式合成，并在事实簇 ID 上做**跨版本
        继承**：新增成簇的句子若与旧版某个事实簇高度重合（即便只剩一篇
        报道），沿用旧簇 ID，保证「删一篇不塌一段」。随后在返回值中附带
        ``changes``：

        * ``added`` / ``removed``：事实层面的新增与消失；
        * ``selected_added`` / ``selected_removed``：入选摘要句的变化；
        * ``attribution``：同一事实的印证来源增减。

        被删文档若只是某事实的多个来源之一，代表句会自动换岗，
        该事实仍保留——摘要不会因删掉一篇而整段塌掉。
        """
        params = dict(previous.get("params", {}))
        result = self.synthesize(documents, **params)
        result = self._inherit_cluster_ids(previous, result)

        old_clusters = {c["id"]: c for c in previous.get("clusters", [])}
        new_clusters = {c["id"]: c for c in result["clusters"]}
        old_selected = {c["id"] for c in previous.get("clusters", []) if c.get("selected")}
        new_selected = {c["id"] for c in result["clusters"] if c.get("selected")}

        added = sorted(new_clusters.keys() - old_clusters.keys())
        removed = sorted(old_clusters.keys() - new_clusters.keys())
        attribution = []
        for cid in old_clusters.keys() & new_clusters.keys():
            before = set(old_clusters[cid].get("source_doc_ids", []))
            after = set(new_clusters[cid].get("source_doc_ids", []))
            if before != after:
                attribution.append({
                    "cluster_id": cid,
                    "text": new_clusters[cid]["rep_text"],
                    "support_before": len(before),
                    "support_after": len(after),
                    "added_sources": sorted(after - before),
                    "removed_sources": sorted(before - after),
                })

        result["changes"] = {
            "added": [self._change_item(new_clusters[c]) for c in added],
            "removed": [self._change_item(old_clusters[c]) for c in removed],
            "selected_added": sorted(new_selected - old_selected),
            "selected_removed": sorted(old_selected - new_selected),
            "attribution": attribution,
        }
        return result

    @staticmethod
    def _change_item(cluster_view: dict) -> dict:
        return {"cluster_id": cluster_view["id"], "text": cluster_view["rep_text"],
                "support": cluster_view["support"]}

    # -- 文档与单元 -------------------------------------------------------
    @staticmethod
    def _normalize_documents(documents: list[dict]) -> list[dict]:
        docs = []
        for i, item in enumerate(documents):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            docs.append({
                "doc_id": str(item.get("doc_id") or f"doc_{i + 1}"),
                "title": item.get("title") or f"文档{i + 1}",
                "text": text,
                "published_at": item.get("published_at"),
            })
        return docs

    def _build_units(self, docs: list[dict]) -> list[_Unit]:
        raw: list[_Unit] = []
        uid = 0
        for doc_idx, doc in enumerate(docs):
            sentences = split_sentences(doc["text"])
            total = len(sentences)
            for sent_index, text in enumerate(sentences):
                words = filter_stopwords(self.segmenter.cut(text))
                if len(words) < 2:
                    continue
                dates, date_keys, spans = _extract_dates(text)
                nums = _extract_numbers(text, spans)
                pos = sent_index / max(total - 1, 1)
                raw.append(_Unit(
                    uid=uid, doc_idx=doc_idx, doc_id=doc["doc_id"],
                    doc_title=doc["title"], sent_index=sent_index,
                    text=text, words=words, pos=pos,
                    dates=dates, date_keys=date_keys, nums=nums,
                    lead=math.exp(-2 * pos)))
                uid += 1

        vectors = compute_tfidf([u.words for u in raw])
        for unit, vec in zip(raw, vectors):
            unit.vec = vec

        if len(raw) > 1:
            graph = {u.uid: {} for u in raw}
            for i, a in enumerate(raw):
                for b in raw[i + 1:]:
                    sim = similarity(a.vec, b.vec)
                    if sim > 0.08:
                        graph[a.uid][b.uid] = sim
                        graph[b.uid][a.uid] = sim
            ranks = pagerank(graph)
            n = max(len(raw), 1)
            for u in raw:
                u.score = ranks.get(u.uid, 0.0) * n + 0.08 * u.lead
        elif raw:
            raw[0].score = 1.0
        return raw

    # -- 冲突探测 ---------------------------------------------------------
    def _find_conflicts(self, units: list[_Unit]) -> set[tuple[int, int, str]]:
        """跨文档、同事件但口径不一致的句子对（疑似同一事实的不同说法）。

        判定「同一事件」要求除向量相似外，内容词有足够重合，避免把
        两件不相干事实的日期 / 数字互相比对。
        """
        conflicts: set[tuple[int, int, str]] = set()
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                if a.doc_id == b.doc_id:
                    continue
                sim = similarity(a.vec, b.vec)
                common = set(a.words) & set(b.words)
                shorter = min(len(a.words), len(b.words))

                # 数字口径冲突（同量词不同数值）最典型：售价 3万 vs 2.8万。
                # 即使措辞差异大，只要有一定主题重合就上报
                if _number_clash_pairs(a.nums, b.nums) and sim >= 0.1 \
                        and len(common) >= 2:
                    conflicts.add((a.uid, b.uid, "number"))

                if sim < 0.15:
                    continue
                if len(common) < max(3, shorter * 0.4):
                    continue
                # 日期口径冲突：两句都只围绕各自的一个日期、且指不到一起。
                # 若某句还携带另一句没有的日期，则日期可能只是不同的叙事
                # 锚点（如「发布日」与「推迟到的月份」），不做冲突判定。
                # 同时要求双方在日期之外另有共同锚点（数字/量词一致），
                # 避免把发布与量产这类不同事件点当成同一日期的分歧。
                if a.date_keys and b.date_keys \
                        and not _dates_compatible(a.date_keys, b.date_keys):
                    diff_a = a.date_keys - b.date_keys
                    diff_b = b.date_keys - a.date_keys
                    if len(diff_a) <= 1 and len(diff_b) <= 1 \
                            and self._shared_anchor(a, b, common):
                        conflicts.add((a.uid, b.uid, "date"))
        return conflicts

    @staticmethod
    def _shared_anchor(a: _Unit, b: _Unit, common: set[str]) -> bool:
        """两句除冲突日期外是否锚定同一事实（共享带量词的数字口径或核心事件词）。"""
        # 共同数字必须带非数字量词（售价/百分比等），产品型号（900）不算口径
        def _units(nums: set[str]) -> set[str]:
            out = set()
            for token in nums:
                m = re.match(r"^[0-9.]+(.*)$", token)
                if m and m.group(1) and not m.group(1).isdigit():
                    out.add(m.group(1))
            return out
        if _units(a.nums) & _units(b.nums):
            return True
        # 要求实词高度重合（≥6 且过半），否则只是「同主体的不同事件点」
        threshold = min(len(a.words), len(b.words)) * 0.6
        return len(common) >= max(6, threshold)

    # -- 聚类对齐 ---------------------------------------------------------
    def _cluster_units(self, units: list[_Unit],
                       threshold: float) -> list[list[int]]:
        """平均链接凝聚聚类，合并时要求所有成员两两口径相容。"""
        groups: list[list[int]] = [[u.uid] for u in units]
        by_id = {u.uid: u for u in units}

        while len(groups) > 1:
            best_pair: Optional[tuple[int, int]] = None
            best_sim = threshold
            for ai in range(len(groups)):
                for bi in range(ai + 1, len(groups)):
                    ga, gb = groups[ai], groups[bi]
                    if not self._groups_compatible(ga, gb, by_id):
                        continue
                    sims = [similarity(by_id[i].vec, by_id[j].vec)
                            for i in ga for j in gb]
                    avg = sum(sims) / len(sims)
                    # 数字口径强证据：跨文档两句共享同一带量词数字（如
                    # 「转移2.3万人」），即便分词噪声压低向量相似度，
                    # 且主题仍有实词重合，也视为同一事实对齐
                    anchored = self._groups_number_anchored(ga, gb, by_id, sims)
                    effective = max(avg, threshold + 0.02) if anchored else avg
                    if effective > best_sim:
                        best_sim = effective
                        best_pair = (ai, bi)
            if best_pair is None:
                break
            ai, bi = best_pair
            groups[ai] = groups[ai] + groups[bi]
            groups.pop(bi)

        for cid, group in enumerate(groups):
            for uid_ in group:
                by_id[uid_].cluster = cid
        return groups

    @staticmethod
    def _groups_number_anchored(ga: list[int], gb: list[int],
                                by_id: dict[int, _Unit],
                                sims: list[float]) -> bool:
        """两组之间是否存在「同一带量词数字 + 主题实词重合」的跨文档对。"""
        idx = 0
        for i in ga:
            for j in gb:
                sim = sims[idx]
                idx += 1
                a, b = by_id[i], by_id[j]
                if a.doc_id == b.doc_id or sim < 0.08:
                    continue
                shared_nums = a.nums & b.nums
                if any(re.match(r"^[0-9.]+\S+$", n) for n in shared_nums):
                    common = set(a.words) & set(b.words)
                    if len(common) >= 2:
                        return True
        return False

    @staticmethod
    def _groups_compatible(ga: list[int], gb: list[int],
                           by_id: dict[int, _Unit]) -> bool:
        for i in ga:
            for j in gb:
                if _compatible(by_id[i], by_id[j]):
                    return False
                # 同量词出现不同数值（售价 3万 vs 2.8万）也禁止桥接合并
                if _number_clash_pairs(by_id[i].nums, by_id[j].nums):
                    return False
        return True

    # -- 事实簇 -----------------------------------------------------------
    @staticmethod
    def _cluster_fingerprint(members: list[_Unit], rep: Optional[_Unit] = None) -> str:
        """事实指纹（首版）。

        取成员中「跨文档出现」的核心实词（每篇每词计一次的文档频次
        最高），单篇时退化为代表句实词。首版 ID 由此确定；文档增删后的
        稳定性由 :meth:`_inherit_cluster_ids` 结合上一版结果保证——
        删篇本身损失了「曾与谁共述」的信息，无状态地恢复它在信息论上
        不成立，因此增量更新显式继承旧 ID。
        """
        doc_freq: Counter = Counter()
        per_doc: dict[str, set[str]] = {}
        for u in members:
            per_doc.setdefault(u.doc_id, set()).update(u.words)
        for words in per_doc.values():
            doc_freq.update(words)
        if len(per_doc) >= 2:
            core = [w for w, n in doc_freq.most_common() if n >= 2]
            if core:
                return hashlib.sha1(
                    "|".join(core[:8]).encode("utf-8")).hexdigest()[:10]
        anchor = rep or members[0]
        key = " ".join(w for w in anchor.words if len(w) > 1)[:120] \
            or " ".join(anchor.words)
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]

    def _inherit_cluster_ids(self, previous: dict, result: dict) -> dict:
        """让新合成结果继承旧版中同一事实的簇 ID。

        新版每个簇与旧版簇比较成员代表句的相似度：若高度重合（同一事实
        的表述仍在，哪怕只剩一篇报道），沿用旧簇 ID。匹配是一对一的，
        保证多个新簇不会抢占同一旧 ID。
        """
        old = previous.get("clusters", [])
        if not old:
            return result
        old_rep_vecs = []
        seg = self.segmenter
        for c in old:
            words = filter_stopwords(seg.cut(c.get("rep_text", "")))
            vec = dict(zip(words, [1.0] * len(words)))
            old_rep_vecs.append((c["id"], c.get("source_doc_ids", []), vec))

        # 为新版簇计算代表句向量（用简单二值向量即可，避免重算全量 IDF）
        taken_old: set[str] = set()
        id_map: dict[str, str] = {}
        new_clusters = result["clusters"]
        new_vecs = []
        for c in new_clusters:
            words = filter_stopwords(seg.cut(c.get("rep_text", "")))
            new_vecs.append(dict(zip(words, [1.0] * len(words))))

        # 贪心按最高相似度配对
        pairs = []
        for ni, nvec in enumerate(new_vecs):
            for oi, (oid, _, ovec) in enumerate(old_rep_vecs):
                sim = self._binary_overlap(nvec, ovec)
                if sim >= 0.5:
                    pairs.append((sim, ni, oi))
        for _, ni, oi in sorted(pairs, reverse=True):
            new_id = new_clusters[ni]["id"]
            old_id = old_rep_vecs[oi][0]
            if new_id in id_map or old_id in taken_old:
                continue
            id_map[new_id] = old_id
            taken_old.add(old_id)

        if not id_map:
            return result

        def _map(cluster_id: str) -> str:
            return id_map.get(cluster_id, cluster_id)

        for c in new_clusters:
            c["id"] = _map(c["id"])
        for s in result.get("sentences", []):
            s["cluster_id"] = _map(s["cluster_id"])
        result["paragraphs"] = [[_map(cid) for cid in para]
                                for para in result.get("paragraphs", [])]
        return result

    @staticmethod
    def _binary_overlap(a: dict, b: dict) -> float:
        """二值词向量的归一重合度（Jaccard）。"""
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _build_clusters(self, units: list[_Unit],
                        groups: list[list[int]]) -> list[dict]:
        by_id = {u.uid: u for u in units}
        clusters = []
        for cid, group in enumerate(groups):
            members = [by_id[uid_] for uid_ in group]
            word_sets = [set(m.words) for m in members]
            # 代表句：选簇内最「居中」的句子——其实词与其他成员实词的
            # 对称覆盖率（Jaccard 之和）最高，即最能概括该事实；并列时取
            # 更凝练、含日期/数字、导语位置靠前的。该准则只依赖簇内成员、
            # 与簇外文档无关，删除外围成员时中心成员通常保持稳定。
            # 跨增删的簇 ID 稳定性最终由 update() 的 ID 继承保证。
            def _centrality(idx: int) -> tuple:
                own = word_sets[idx]
                score = sum(len(own & word_sets[j]) /
                            max(len(own | word_sets[j]), 1)
                            for j in range(len(members)) if j != idx)
                u = members[idx]
                return (score, bool(u.dates) + bool(u.nums),
                        u.lead, -len(u.text))
            rep_idx = max(range(len(members)), key=_centrality)
            rep = members[rep_idx]
            doc_ids = sorted({u.doc_id for u in members},
                             key=lambda d: by_id[[m.uid for m in members
                                                  if m.doc_id == d][0]].doc_idx)
            centroid: dict[str, float] = {}
            for u in members:
                for word, weight in u.vec.items():
                    centroid[word] = centroid.get(word, 0.0) + weight
            scale = 1 / len(members)
            centroid = {w: v * scale for w, v in centroid.items()}
            # 稳定簇 ID：锚定到代表句，与成员数无关（见方法文档）
            fp = self._cluster_fingerprint(members, rep)
            date_keys = set().union(*(u.date_keys for u in members))
            date_label = ""
            if date_keys:
                y, mo, d = min(date_keys)
                if y:
                    date_label = f"{y:04d}-{mo:02d}" + (f"-{d:02d}" if d else "")
                elif d:
                    date_label = f"{mo:02d}-{d:02d}"
                else:
                    date_label = f"{mo:02d}月"
            # 簇内只在单篇文档出现的带量词数字（未被其他报道印证的口径）
            num_by_doc: dict[str, set[str]] = {}
            for u in members:
                num_by_doc.setdefault(u.doc_id, set()).update(u.nums)
            unique_numbers: set[str] = set()
            if len(num_by_doc) > 1:
                counts: Counter = Counter()
                for nums in num_by_doc.values():
                    counts.update(_number_values(nums).keys())
                # 某量词仅一篇提到 -> 该篇的该数字是独有口径
                for doc_id, nums in num_by_doc.items():
                    for token in nums:
                        m = re.match(r"^([0-9.]+)(.*)$", token)
                        if m and m.group(2) and counts[m.group(2)] == 1:
                            unique_numbers.add(token)
            clusters.append({
                "id": fp,
                "order_index": cid,
                "members": members,
                "rep": rep,
                "doc_ids": doc_ids,
                "support": len(doc_ids),
                "centroid": centroid,
                "date_label": date_label,
                "date_key": min(date_keys) if date_keys else None,
                "concrete": bool(date_keys)
                            or bool(set().union(*(u.nums for u in members))),
                "unique_numbers": unique_numbers,
                "score": 0.0,
            })
        return clusters

    # -- 打分 -------------------------------------------------------------
    def _score_clusters(self, clusters: list[dict]) -> None:
        k = len(clusters)
        if k > 1:
            graph = {c["id"]: {} for c in clusters}
            for i, a in enumerate(clusters):
                for b in clusters[i + 1:]:
                    sim = similarity(a["centroid"], b["centroid"])
                    if sim > 0.04:
                        graph[a["id"]][b["id"]] = sim
                        graph[b["id"]][a["id"]] = sim
            ranks = pagerank(graph)
        else:
            ranks = {clusters[0]["id"]: 1.0}
        for c in clusters:
            centrality = ranks.get(c["id"], 0.0) * k
            c["score"] = (centrality
                          + 0.12 * math.log(1 + c["support"])
                          + 0.06 * c["rep"].lead
                          + (0.04 if c["concrete"] else 0.0))

    # -- 选择 -------------------------------------------------------------
    def _select_clusters(self, clusters: list[dict], budget: int,
                         max_source_share: float) -> list[dict]:
        ranked = sorted(clusters, key=lambda c: c["score"], reverse=True)
        n_sources = len({c["rep"].doc_id for c in clusters})

        # 单篇占比上限只约束「独有句」（单来源事实）——这才是可能构成
        # 「整段照搬」的部分；多源共识句代表多篇共同事实，不占主导名额。
        # cap 基于**总预算**的占比，确保即使其他来源没有独有事实，
        # 单一文档的独有内容也不超过综述的一半。
        multi = [c for c in ranked if c["support"] > 1]
        single = [c for c in ranked if c["support"] <= 1]
        if n_sources <= 1:
            cap = budget  # 本来就只有一篇，占比限制无意义
        else:
            cap = max(1, math.floor(budget * max_source_share))

        # 含数字/日期的独有事实视为关键事实，给保底名额
        concrete_singles = [c for c in single if c["concrete"]]
        head = concrete_singles[: max(1, int(budget * 0.6))]
        head_ids = {c["id"] for c in head}
        # 共识事实始终优先，独有事实中关键事实保底，其余按重要度排队
        candidates = multi + head + [c for c in single
                                     if c["id"] not in head_ids]

        def greedy(use_cap: bool, mmr_threshold: float) -> list[dict]:
            accepted: list[dict] = []
            used: dict[str, int] = {}
            skipped_by_cap: list[dict] = []
            for c in candidates:
                if len(accepted) >= budget:
                    break
                rep_doc = c["rep"].doc_id
                # 多源印证的共识句不记入单一来源的主导名额——它代表的是
                # 多篇共同事实，而非某一篇的独有内容，不应被占比上限误伤
                single_sourced = c["support"] <= 1
                if (use_cap and single_sourced
                        and used.get(rep_doc, 0) >= cap):
                    skipped_by_cap.append(c)
                    continue
                if any(similarity(c["centroid"], a["centroid"]) >= mmr_threshold
                       for a in accepted):
                    continue
                accepted.append(c)
                if single_sourced:
                    used[rep_doc] = used.get(rep_doc, 0) + 1
            return accepted, skipped_by_cap

        chosen, _ = greedy(True, 0.5)
        if len(chosen) < budget:
            # 先放宽去冗余阈值，但仍守住单篇占比上限（防整段照搬）
            chosen, _ = greedy(True, 0.9)
        # 守住占比上限后若仍填不满预算，说明独有信息高度集中在个别文档。
        # 此时主动保留已选内容、收缩篇幅，而不是用单一文档的句子凑满
        # 配额——多文档综述宁可精炼，也不能退化成某一篇的缩写。
        return chosen

    @staticmethod
    def _per_source_counts(chosen: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in chosen:
            counts[c["rep"].doc_id] = counts.get(c["rep"].doc_id, 0) + 1
        return counts

    # -- 排序 -------------------------------------------------------------
    def _order_clusters(self, chosen: list[dict], order: str) -> list[dict]:
        if order == "timeline":
            return self._order_timeline(chosen)
        if order == "source":
            return sorted(chosen, key=lambda c: (c["rep"].doc_idx,
                                                 c["rep"].sent_index))
        return self._order_logic(chosen)

    @staticmethod
    def _order_logic(chosen: list[dict]) -> list[dict]:
        """从事件总起句出发做主题贪心游走，相邻事实尽量同主题。"""
        # 开场句：以主题中心性（score，PageRank）为主——中心事实最能
        # 概括「发生了什么」；同分时优先导语靠前、多源印证、更凝练的。
        first = max(chosen, key=lambda c: (
            round(c["score"], 2),
            min(c["support"], 3),
            c["rep"].lead,
            -len(c["rep"].text)))
        remaining = [c for c in chosen if c["id"] != first["id"]]
        remaining.sort(key=lambda c: c["score"], reverse=True)
        ordered = [first]
        while remaining:
            cur = ordered[-1]
            best_idx, best_sim = 0, -1.0
            for i, c in enumerate(remaining):
                sim = similarity(cur["centroid"], c["centroid"])
                if sim > best_sim:
                    best_sim, best_idx = sim, i
            if best_sim < 0.08:
                # 主题断开时，回到剩余中最重要的事实开启新段
                best_idx = max(range(len(remaining)),
                               key=lambda i: remaining[i]["score"])
            ordered.append(remaining.pop(best_idx))
        return ordered

    @staticmethod
    def _order_timeline(chosen: list[dict]) -> list[dict]:
        dated = sorted([c for c in chosen if c["date_key"]],
                       key=lambda c: (c["date_key"], -c["score"]))
        undated = [c for c in chosen if not c["date_key"]]
        ordered = list(dated)
        # 无日期事实插入到语义最连贯的位置
        for c in sorted(undated, key=lambda x: x["score"], reverse=True):
            best_pos, best_score = len(ordered), -1.0
            for pos in range(len(ordered) + 1):
                neighbors = []
                if pos > 0:
                    neighbors.append(ordered[pos - 1])
                if pos < len(ordered):
                    neighbors.append(ordered[pos])
                score = sum(similarity(c["centroid"], n["centroid"])
                            for n in neighbors) / max(len(neighbors), 1)
                if score > best_score:
                    best_score, best_pos = score, pos
            ordered.insert(best_pos, c)
        return ordered

    # -- 分段 -------------------------------------------------------------
    @staticmethod
    def _paragraphize(ordered: list[dict], order: str) -> list[list[dict]]:
        if not ordered:
            return []
        if order == "source":
            groups: dict[str, list[dict]] = {}
            for c in ordered:
                groups.setdefault(c["rep"].doc_id, []).append(c)
            paras = [sorted(v, key=lambda c: c["rep"].sent_index)
                     for v in groups.values()]
            return MultiDocumentSummarizer._merge_singletons(paras)
        if order == "timeline":
            paras = [[ordered[0]]]
            for c in ordered[1:]:
                prev = paras[-1][-1]
                if c["date_label"] and prev["date_label"] \
                        and c["date_label"] != prev["date_label"]:
                    paras.append([c])
                else:
                    paras[-1].append(c)
            return MultiDocumentSummarizer._merge_singletons(paras)

        # logic：主题连贯性断点分段
        paras = [[ordered[0]]]
        for c in ordered[1:]:
            sim = similarity(paras[-1][-1]["centroid"], c["centroid"])
            if sim < 0.1 and len(paras[-1]) >= 2 and len(paras[-1]) < 6:
                paras.append([c])
            else:
                paras[-1].append(c)
        return MultiDocumentSummarizer._merge_singletons(paras)

    @staticmethod
    def _merge_singletons(paras: list[list[dict]]) -> list[list[dict]]:
        if len(paras) <= 1:
            return paras
        merged = [paras[0]]
        for para in paras[1:]:
            if len(para) == 1 and len(merged[-1]) < 4:
                merged[-1].extend(para)
            else:
                merged.append(para)
        return merged

    # -- 渲染 -------------------------------------------------------------
    @staticmethod
    def _render(paragraphs: list[list[dict]],
                cite_index: dict[str, int]) -> str:
        lines = []
        for para in paragraphs:
            chunks = []
            for c in para:
                cites = "".join(f"[{cite_index[d]}]" for d in c["doc_ids"])
                chunks.append(_ensure_period(c["rep"].text) + cites)
            lines.append("".join(chunks))
        return "\n".join(lines)

    @staticmethod
    def _build_sentence_view(chosen: list[dict],
                             cite_index: dict[str, int]) -> list[dict]:
        result = []
        for c in chosen:
            result.append({
                "cluster_id": c["id"],
                "text": _ensure_period(c["rep"].text),
                "score": round(c["score"], 4),
                "support": c["support"],
                "date": c["date_label"],
                "concrete": c["concrete"],
                "citations": [cite_index[d] for d in c["doc_ids"]],
                "sources": [{
                    "index": cite_index[u.doc_id],
                    "doc_id": u.doc_id,
                    "title": u.doc_title,
                    "sent_index": u.sent_index,
                    "text": u.text,
                    "is_primary": u.uid == c["rep"].uid,
                } for u in sorted(c["members"],
                                  key=lambda u: (u.doc_idx, u.sent_index))],
            })
        return result

    @staticmethod
    def _cluster_view(c: dict, selected: bool,
                      cite_index: dict[str, int]) -> dict:
        return {
            "id": c["id"],
            "selected": selected,
            "score": round(c["score"], 4),
            "support": c["support"],
            "date": c["date_label"],
            "concrete": c["concrete"],
            "rep_text": _ensure_period(c["rep"].text),
            "source_doc_ids": c["doc_ids"],
            "citations": [cite_index[d] for d in c["doc_ids"]],
            "source_titles": sorted({u.doc_title for u in c["members"]}),
            # 簇内只在单篇出现的数字口径（如某句顺带提及的售价），
            # 便于界面提示「该句还包含未被其他报道印证的数字」
            "unique_numbers": sorted(c["unique_numbers"]),
            "members": [{
                "doc_id": u.doc_id,
                "doc_title": u.doc_title,
                "sent_index": u.sent_index,
                "text": _ensure_period(u.text),
                "is_primary": u.uid == c["rep"].uid,
            } for u in sorted(c["members"],
                              key=lambda u: (u.doc_idx, u.sent_index))],
        }

    # -- 冲突视图 ---------------------------------------------------------
    @staticmethod
    def _format_conflicts(units: list[_Unit], clusters: list[dict],
                          raw: set[tuple[int, int, str]]) -> list[dict]:
        by_id = {u.uid: u for u in units}
        kind_names = {"date": "日期不一致", "number": "数字口径不一致"}
        seen: set[frozenset] = set()
        conflicts = []
        for ua, ub, kind in sorted(raw):
            a, b = by_id[ua], by_id[ub]
            if a.cluster == b.cluster:
                continue
            key = frozenset((a.cluster, b.cluster))
            if key in seen:
                continue
            seen.add(key)
            conflicts.append({
                "kind": kind,
                "kind_name": kind_names[kind],
                "values_a": sorted(a.dates if kind == "date" else a.nums),
                "values_b": sorted(b.dates if kind == "date" else b.nums),
                "claim_a": {"doc_id": a.doc_id, "title": a.doc_title,
                            "text": _ensure_period(a.text)},
                "claim_b": {"doc_id": b.doc_id, "title": b.doc_title,
                            "text": _ensure_period(b.text)},
            })
        return conflicts
