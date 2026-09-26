"""app.digest 的 /infer-partial-digest 纯标准库测试：

python3 tests/test_infer_partial_digest.py
"""

import sys
import os
import random
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.digest import (  # noqa: E402
    DigestError,
    run_digest,
    run_infer_partial_digest,
)

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def expect_error(name, payload, snippet):
    try:
        run_infer_partial_digest(payload)
    except DigestError as e:
        check(name, snippet in e.reason, f"reason={e.reason!r}")
        return
    check(name, False, "未抛出 DigestError")


ECORI = {"name": "EcoRI", "recognition": "GAATTC", "cut_offset": 1}
BSAI = {"name": "BsaI", "recognition": "GGTCTC", "cut_offset": 1}

# --- 线性部分酶切：漏切数最少的全部方案，按保留切点字典序 ---
# EcoRI 在序列上产生切点 [3, 7, 12, 16]；保留 {3,12} 或 {7,16} 都得谱 [3,7,9]
SEQ19 = "AAGAATTCTTTGAATTCGG"
res = run_infer_partial_digest({
    "sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
    "observed_lengths": [9, 7, 3], "tolerance": 0,
})
check("linear status", res["status"] == "solution", str(res))
check("linear effective cuts", res["effective_cuts"] == [3, 7, 12, 16],
      str(res.get("effective_cuts")))
check("linear missed", res["missed_cuts"] == 2, str(res.get("missed_cuts")))
check("linear solution count", res["solution_count"] == 2, str(res))
check("linear solutions lexicographic",
      [s["kept_cuts"] for s in res["solutions"]] == [[3, 12], [7, 16]],
      str(res["solutions"]))
check("linear profiles",
      all(s["fragment_lengths"] == [3, 7, 9] for s in res["solutions"]),
      str(res["solutions"]))

# --- 完全酶切（漏切 0）：观测谱即全部片段 ---
res = run_infer_partial_digest({
    "sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
    "observed_lengths": [5, 4, 3, 4, 3], "tolerance": 0,
})
check("full digest missed 0", res["status"] == "solution"
      and res["missed_cuts"] == 0
      and [s["kept_cuts"] for s in res["solutions"]] == [[3, 7, 12, 16]]
      and res["solutions"][0]["fragment_lengths"] == [3, 3, 4, 4, 5], str(res))

# --- 全部漏切（空保留集）：唯一片段为完整序列 ---
res = run_infer_partial_digest({
    "sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
    "observed_lengths": [19], "tolerance": 0,
})
check("all missed full length", res["status"] == "solution"
      and res["missed_cuts"] == 4
      and [s["kept_cuts"] for s in res["solutions"]] == [[]]
      and res["solutions"][0]["fragment_lengths"] == [19], str(res))

# --- 无匹配：no_solution ---
res = run_infer_partial_digest({
    "sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
    "observed_lengths": [2, 2], "tolerance": 0,
})
check("no solution", res == {"status": "no_solution"}, str(res))

# --- 容差：逐项差 <= tolerance 才算匹配 ---
res = run_infer_partial_digest({
    "sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
    "observed_lengths": [3, 7, 10], "tolerance": 1,
})
check("tolerance boundary match", res["status"] == "solution"
      and res["missed_cuts"] == 2
      and [s["kept_cuts"] for s in res["solutions"]] == [[3, 12], [7, 16]],
      str(res))
res = run_infer_partial_digest({
    "sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
    "observed_lengths": [3, 7, 10], "tolerance": 0,
})
check("tolerance zero rejects", res == {"status": "no_solution"}, str(res))

# --- 环状：单切点线性化为全长；漏切数最少优先于更多漏切的等价解 ---
# BsaI 在环状序列上产生切点 [9, 17]；保留任一切点都得谱 [18]，
# 空保留集也得 [18]，但漏切数 1 < 2，只返回漏切最少的两个方案
CSEQ = "TCTCAAAAGGTCTCAAGG"
res = run_infer_partial_digest({
    "sequence": CSEQ, "topology": "circular", "enzymes": [BSAI],
    "observed_lengths": [18], "tolerance": 0,
})
check("circular min missed preferred", res["status"] == "solution"
      and res["effective_cuts"] == [9, 17]
      and res["missed_cuts"] == 1
      and [s["kept_cuts"] for s in res["solutions"]] == [[9], [17]]
      and all(s["fragment_lengths"] == [18] for s in res["solutions"]),
      str(res))

# 环状完全酶切
res = run_infer_partial_digest({
    "sequence": CSEQ, "topology": "circular", "enzymes": [BSAI],
    "observed_lengths": [10, 8], "tolerance": 0,
})
check("circular full digest", res["status"] == "solution"
      and res["missed_cuts"] == 0
      and [s["kept_cuts"] for s in res["solutions"]] == [[9, 17]]
      and res["solutions"][0]["fragment_lengths"] == [8, 10], str(res))

# --- 线性端点切点（0 与 N）不参与枚举 ---
# AATT 起点 0、offset 0：正向切点 0、回文反向切点 4 -> 有效切点仅 [4]
res = run_infer_partial_digest({
    "sequence": "AATTCCCC", "topology": "linear",
    "enzymes": [{"name": "E", "recognition": "AATT", "cut_offset": 0}],
    "observed_lengths": [4, 4], "tolerance": 0,
})
check("endpoint cut 0 excluded", res["status"] == "solution"
      and res["effective_cuts"] == [4]
      and res["missed_cuts"] == 0
      and [s["kept_cuts"] for s in res["solutions"]] == [[4]], str(res))
res = run_infer_partial_digest({
    "sequence": "AATTCCCC", "topology": "linear",
    "enzymes": [{"name": "E", "recognition": "AATT", "cut_offset": 0}],
    "observed_lengths": [8], "tolerance": 0,
})
check("endpoint cut 0 not enumerated", res["status"] == "solution"
      and res["missed_cuts"] == 1
      and [s["kept_cuts"] for s in res["solutions"]] == [[]], str(res))
# 端点切点 N：AATT 起点 4、offset 4 -> 正向切点 8(=N)、反向切点 4
res = run_infer_partial_digest({
    "sequence": "CCCCAATT", "topology": "linear",
    "enzymes": [{"name": "E", "recognition": "AATT", "cut_offset": 4}],
    "observed_lengths": [4, 4], "tolerance": 0,
})
check("endpoint cut N excluded", res["status"] == "solution"
      and res["effective_cuts"] == [4]
      and res["missed_cuts"] == 0, str(res))

# --- 无酶：有效切点为空，仅可能匹配完整长度 ---
res = run_infer_partial_digest({
    "sequence": "ACGTACGT", "topology": "linear", "enzymes": [],
    "observed_lengths": [8], "tolerance": 0,
})
check("no enzymes full length", res["status"] == "solution"
      and res["effective_cuts"] == []
      and res["missed_cuts"] == 0
      and [s["kept_cuts"] for s in res["solutions"]] == [[]]
      and res["solutions"][0]["fragment_lengths"] == [8], str(res))
res = run_infer_partial_digest({
    "sequence": "ACGTACGT", "topology": "linear", "enzymes": [],
    "observed_lengths": [4, 4], "tolerance": 0,
})
check("no enzymes no solution", res == {"status": "no_solution"}, str(res))

# --- 有效切点超过 12 个：400 ---
# "AA"（反向互补 "TT" 不命中）在 14 个 A 上正向命中起点 0..12 -> 切点 1..13 共 13 个
expect_error("too many effective cuts", {
    "sequence": "AAAAAAAAAAAAAA", "topology": "linear",
    "enzymes": [{"name": "E", "recognition": "AA", "cut_offset": 1}],
    "observed_lengths": [14], "tolerance": 0,
}, "超过上限")
# 环状：14 个起点 -> 切点 0..13 共 14 个
expect_error("too many effective cuts circular", {
    "sequence": "AAAAAAAAAAAAAA", "topology": "circular",
    "enzymes": [{"name": "E", "recognition": "AA", "cut_offset": 1}],
    "observed_lengths": [14], "tolerance": 0,
}, "超过上限")

# --- observed_lengths 校验 ---
BASE = {"sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
        "tolerance": 0}
expect_error("missing observed", BASE, "observed_lengths")
expect_error("observed not list",
             {**BASE, "observed_lengths": "3,7,9"}, "数组")
expect_error("observed empty",
             {**BASE, "observed_lengths": []}, "数量必须在 1..13")
expect_error("observed too many",
             {**BASE, "observed_lengths": [1] * 14}, "数量必须在 1..13")
# 13 个为合法上限：校验通过，但谱不可能含 13 个长度 1 的片段 -> no_solution
res = run_infer_partial_digest({**BASE, "observed_lengths": [1] * 13})
check("observed 13 allowed", res == {"status": "no_solution"}, str(res))
expect_error("observed zero",
             {**BASE, "observed_lengths": [3, 0]}, "正整数")
expect_error("observed negative",
             {**BASE, "observed_lengths": [-2]}, "正整数")
expect_error("observed float",
             {**BASE, "observed_lengths": [3.5]}, "正整数")
expect_error("observed bool",
             {**BASE, "observed_lengths": [True]}, "正整数")
expect_error("observed string",
             {**BASE, "observed_lengths": ["9"]}, "正整数")

# --- tolerance 校验 ---
BASE_OBS = {"sequence": SEQ19, "topology": "linear", "enzymes": [ECORI],
            "observed_lengths": [3, 7, 9]}
expect_error("missing tolerance", BASE_OBS, "tolerance")
expect_error("tolerance negative",
             {**BASE_OBS, "tolerance": -1}, "非负整数")
expect_error("tolerance float",
             {**BASE_OBS, "tolerance": 0.5}, "非负整数")
expect_error("tolerance bool",
             {**BASE_OBS, "tolerance": True}, "非负整数")

# --- 复用序列/拓扑/酶校验 ---
expect_error("payload not dict", ["not", "dict"], "JSON 对象")
expect_error("illegal base", {**BASE_OBS, "sequence": "ACGTX",
                              "tolerance": 0}, "非法碱基")
expect_error("empty seq", {**BASE_OBS, "sequence": " ",
                           "tolerance": 0}, "不能为空")
expect_error("bad topology", {**BASE_OBS, "topology": "plasmid",
                              "tolerance": 0}, "拓扑")
expect_error("bad enzyme offset", {
    "sequence": "AAAA", "topology": "linear",
    "enzymes": [{"name": "E", "recognition": "AA", "cut_offset": 3}],
    "observed_lengths": [4], "tolerance": 0,
}, "越界")


# --- 随机用例：与独立朴素枚举参考完全一致 ---
def reference_profile(n, cuts, topology):
    """独立重写的片段长度谱计算（不调用 app.digest 的内部函数）。"""
    cuts = sorted(cuts)
    if topology == "linear":
        bounds = [0] + [c for c in cuts if 0 < c < n] + [n]
        return sorted(b - a for a, b in zip(bounds, bounds[1:]))
    if not cuts:
        return [n]
    return sorted(
        (b - a) % n or n for a, b in zip(cuts, cuts[1:] + cuts[:1])
    )


def reference_solve(n, effective, topology, observed, tolerance):
    """朴素枚举全部保留子集，返回 (最少漏切数, [(保留切点, 谱), ...]) 或 None。"""
    target = sorted(observed)
    for size in range(len(effective), -1, -1):
        sols = []
        for combo in itertools.combinations(effective, size):
            prof = reference_profile(n, list(combo), topology)
            if len(prof) == len(target) and all(
                abs(a - b) <= tolerance for a, b in zip(prof, target)
            ):
                sols.append((list(combo), prof))
        if sols:
            sols.sort(key=lambda s: s[0])
            return len(effective) - size, sols
    return None


random.seed(20260926)
TRIALS = 300
checked_trials = 0
for _ in range(TRIALS):
    n = random.randint(4, 30)
    seq = "".join(random.choice("ACGT") for _ in range(n))
    topology = random.choice(["linear", "circular"])
    enzymes = []
    for _ in range(random.randint(0, 2)):
        L = random.randint(1, 4)
        rec = "".join(random.choice("ACGT") for _ in range(L))
        enzymes.append(
            {"name": "E", "recognition": rec, "cut_offset": random.randint(0, L)}
        )

    dig = run_digest(
        {"sequence": seq, "topology": topology, "enzymes": enzymes}
    )
    if topology == "linear":
        effective = sorted(c for c in dig["cuts"] if 0 < c < n)
    else:
        effective = sorted(dig["cuts"])

    # 观测谱：随机取一个保留子集生成谱后逐项扰动，容差 0..2
    keep = [c for c in effective if random.random() < 0.7]
    profile = reference_profile(n, keep, topology)
    tolerance = random.randint(0, 2)
    observed = [max(1, x + random.randint(-1, 1)) for x in profile]
    random.shuffle(observed)
    if len(observed) > 13:
        continue

    payload = {
        "sequence": seq, "topology": topology, "enzymes": enzymes,
        "observed_lengths": observed, "tolerance": tolerance,
    }
    if len(effective) > 12:
        try:
            run_infer_partial_digest(payload)
        except DigestError as e:
            assert "超过上限" in e.reason, (seq, effective, e.reason)
            checked_trials += 1
            continue
        raise AssertionError(f"有效切点 {len(effective)} 个未报错: {seq}")

    res = run_infer_partial_digest(payload)
    ref = reference_solve(n, effective, topology, observed, tolerance)
    if ref is None:
        assert res == {"status": "no_solution"}, (payload, res)
    else:
        ref_missed, ref_sols = ref
        assert res["status"] == "solution", (payload, res)
        assert res["effective_cuts"] == effective, (payload, res)
        assert res["missed_cuts"] == ref_missed, (payload, res, ref_missed)
        got = [(s["kept_cuts"], s["fragment_lengths"]) for s in res["solutions"]]
        assert got == ref_sols, (payload, got, ref_sols)
        assert res["solution_count"] == len(ref_sols), (payload, res)
    checked_trials += 1
print(f"  PASS randomized equivalence vs brute-force reference "
      f"({checked_trials} cases)")

print(f"\n{PASS + 1} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
