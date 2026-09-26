"""app.digest 的 /infer-partial-digest 纯标准库测试：python3 tests/test_infer_partial_digest.py"""

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


def ref_profile(n, kept, topology):
    """独立的片段长度谱参考实现（保留切点 kept 已是不含端点的有效切点）。"""
    if topology == "linear":
        bounds = [0, *sorted(kept), n]
        return sorted(b - a for a, b in zip(bounds, bounds[1:]))
    points = sorted(kept)
    if not points:
        return [n]
    return sorted(
        (b - a) % n or n for a, b in zip(points, [*points[1:], points[0]])
    )


def effective_cuts(seq, topology, enzymes):
    """有效切点：线性剔除端点 0/N，环状取模去重（与实现共用同一切点计算）。"""
    res = run_digest({"sequence": seq, "topology": topology, "enzymes": enzymes})
    n = len(seq)
    if topology == "linear":
        return sorted(c for c in res["cuts"] if 0 < c < n)
    return sorted(res["cuts"])


def reference(seq, topology, enzymes, observed, tolerance):
    """朴素枚举参考：逐层枚举保留子集，套用相同匹配与择优规则。"""
    n = len(seq)
    effective = effective_cuts(seq, topology, enzymes)
    k = len(effective)
    if k > 12:
        return "too_many_cuts"
    target = sorted(observed)
    for kept_count in range(k, -1, -1):
        sols = []
        for combo in itertools.combinations(effective, kept_count):
            profile = ref_profile(n, combo, topology)
            if len(profile) == len(target) and all(
                abs(p - o) <= tolerance for p, o in zip(profile, target)
            ):
                sols.append({"kept_cuts": list(combo), "fragment_lengths": profile})
        if sols:
            sols.sort(key=lambda s: s["kept_cuts"])
            return {
                "status": "solution",
                "missed_cuts": k - kept_count,
                "solution_count": len(sols),
                "solutions": sols,
            }
    return {"status": "no_solution"}


# ---------------------------------------------------------------------------
# 核心夹具：线性 AAGAATTCTTTGAATTCGG（19bp）+ EcoRI(GAATTC, off 1)
# 完全酶切切点为 [3, 7, 12, 16]，片段长度 [3,4,5,4,3]
# ---------------------------------------------------------------------------
ECORI = {"name": "EcoRI", "recognition": "GAATTC", "cut_offset": 1}
SEQ = "AAGAATTCTTTGAATTCGG"
BASE = {"sequence": SEQ, "topology": "linear", "enzymes": [ECORI]}

d = run_digest(BASE)
check("fixture cuts", d["cuts"] == [3, 7, 12, 16], str(d["cuts"]))

# --- 完全酶切（无漏切）：观测即全部片段 ---
r = run_infer_partial_digest({**BASE, "observed_lengths": [3, 3, 4, 4, 5],
                              "tolerance": 0})
check("full digest missed 0", r["status"] == "solution"
      and r["missed_cuts"] == 0 and r["solution_count"] == 1, str(r))
check("full digest kept all cuts",
      r["solutions"][0]["kept_cuts"] == [3, 7, 12, 16], str(r["solutions"]))
check("full digest profile",
      r["solutions"][0]["fragment_lengths"] == [3, 3, 4, 4, 5])

# --- 部分酶切唯一解：观测 [3,3,13] 只可能保留切点 3 与 16（漏切 7、12） ---
r = run_infer_partial_digest({**BASE, "observed_lengths": [3, 3, 13],
                              "tolerance": 0})
check("partial unique solution", r == {
    "status": "solution", "missed_cuts": 2, "solution_count": 1,
    "solutions": [{"kept_cuts": [3, 16], "fragment_lengths": [3, 3, 13]}]},
    str(r))
# 观测顺序无关（内部升序排列）
r2 = run_infer_partial_digest({**BASE, "observed_lengths": [13, 3, 3],
                               "tolerance": 0})
check("observed order irrelevant", r2 == r, str(r2))

# --- 同一最少漏切数下多个方案：全部返回，按保留切点列表字典序 ---
# 保留 {3,7} 与 {12,16} 的谱都是 [3,4,12]
r = run_infer_partial_digest({**BASE, "observed_lengths": [3, 4, 12],
                              "tolerance": 0})
check("multiple solutions lexicographic", r == {
    "status": "solution", "missed_cuts": 2, "solution_count": 2,
    "solutions": [
        {"kept_cuts": [3, 7], "fragment_lengths": [3, 4, 12]},
        {"kept_cuts": [12, 16], "fragment_lengths": [3, 4, 12]},
    ]}, str(r))

# --- 容差：逐位差 <= tolerance 才算匹配 ---
# 观测 [3,4,13]：tol 1 时 [3,4,12]（差1）与 [3,3,13]（差1）都匹配 -> 3 个方案
r = run_infer_partial_digest({**BASE, "observed_lengths": [3, 4, 13],
                              "tolerance": 1})
check("tolerance widens matches", r == {
    "status": "solution", "missed_cuts": 2, "solution_count": 3,
    "solutions": [
        {"kept_cuts": [3, 7], "fragment_lengths": [3, 4, 12]},
        {"kept_cuts": [3, 16], "fragment_lengths": [3, 3, 13]},
        {"kept_cuts": [12, 16], "fragment_lengths": [3, 4, 12]},
    ]}, str(r))
r = run_infer_partial_digest({**BASE, "observed_lengths": [3, 4, 13],
                              "tolerance": 0})
check("tolerance 0 rejects off-by-one", r == {"status": "no_solution"}, str(r))

# --- 无保留切点 = 完整序列；漏切数最多也要返回 ---
r = run_infer_partial_digest({**BASE, "observed_lengths": [19], "tolerance": 0})
check("no kept cut -> full sequence", r == {
    "status": "solution", "missed_cuts": 4, "solution_count": 1,
    "solutions": [{"kept_cuts": [], "fragment_lengths": [19]}]}, str(r))

# --- 片段数必须相同：任何子集都给不出 3 段且和为 19 的 [1,1,1] ---
r = run_infer_partial_digest({**BASE, "observed_lengths": [1, 1, 1],
                              "tolerance": 0})
check("impossible observed -> no_solution", r == {"status": "no_solution"},
      str(r))

# --- 无酶（有效切点为空）：只能匹配完整序列 ---
r = run_infer_partial_digest({"sequence": SEQ, "topology": "linear",
                              "enzymes": [], "observed_lengths": [19],
                              "tolerance": 0})
check("no enzymes -> full sequence solution", r == {
    "status": "solution", "missed_cuts": 0, "solution_count": 1,
    "solutions": [{"kept_cuts": [], "fragment_lengths": [19]}]}, str(r))
r = run_infer_partial_digest({"sequence": SEQ, "topology": "linear",
                              "enzymes": [], "observed_lengths": [5, 14],
                              "tolerance": 0})
check("no enzymes count mismatch -> no_solution",
      r == {"status": "no_solution"}, str(r))

# --- 环状：AAG/off0 在 AAGAAGAA 环上切点为 {0,3} ---
# 观测 [8]：保留 {0} 或 {3} 都把环线性化为全长 8 -> 漏切 1，两个方案
CIRC = {"sequence": "AAGAAGAA", "topology": "circular",
        "enzymes": [{"name": "AAG", "recognition": "AAG", "cut_offset": 0}]}
check("circular fixture cuts",
      run_digest(CIRC)["cuts"] == [0, 3], str(run_digest(CIRC)["cuts"]))
r = run_infer_partial_digest({**CIRC, "observed_lengths": [8], "tolerance": 0})
check("circular single-cut linearizes", r == {
    "status": "solution", "missed_cuts": 1, "solution_count": 2,
    "solutions": [
        {"kept_cuts": [0], "fragment_lengths": [8]},
        {"kept_cuts": [3], "fragment_lengths": [8]},
    ]}, str(r))
r = run_infer_partial_digest({**CIRC, "observed_lengths": [3, 5],
                              "tolerance": 0})
check("circular both cuts kept", r == {
    "status": "solution", "missed_cuts": 0, "solution_count": 1,
    "solutions": [{"kept_cuts": [0, 3], "fragment_lengths": [3, 5]}]}, str(r))

# --- 环状切点取模去重：AA/off2 在 TTAA 环上正向切点 4%4=0，反向切点 0，去重为 {0} ---
MOD = {"sequence": "TTAA", "topology": "circular",
       "enzymes": [{"name": "AA2", "recognition": "AA", "cut_offset": 2}]}
check("circular mod dedup cuts", run_digest(MOD)["cuts"] == [0],
      str(run_digest(MOD)["cuts"]))
r = run_infer_partial_digest({**MOD, "observed_lengths": [4], "tolerance": 0})
check("circular mod dedup single cut", r == {
    "status": "solution", "missed_cuts": 0, "solution_count": 1,
    "solutions": [{"kept_cuts": [0], "fragment_lengths": [4]}]}, str(r))

# --- 线性端点切点排除：AAG/off0 切在 0、AGT/off3 切在 4=N，有效切点为空 ---
ENDPOINT = {"sequence": "AAGT", "topology": "linear", "enzymes": [
    {"name": "E0", "recognition": "AAG", "cut_offset": 0},
    {"name": "EN", "recognition": "AGT", "cut_offset": 3},
]}
check("endpoint cuts excluded from digest cuts",
      run_digest(ENDPOINT)["cuts"] == [0, 4], str(run_digest(ENDPOINT)["cuts"]))
r = run_infer_partial_digest({**ENDPOINT, "observed_lengths": [4],
                              "tolerance": 0})
check("endpoint cuts not effective", r == {
    "status": "solution", "missed_cuts": 0, "solution_count": 1,
    "solutions": [{"kept_cuts": [], "fragment_lengths": [4]}]}, str(r))

# --- 有效切点数量边界：12 个接受，13 个拒绝（400） ---
# "A"*13 + AA/off1：切点 1..12 共 12 个；全保留得 13 段长度 1
r = run_infer_partial_digest({
    "sequence": "A" * 13, "topology": "linear",
    "enzymes": [{"name": "E", "recognition": "AA", "cut_offset": 1}],
    "observed_lengths": [1] * 13, "tolerance": 0,
})
check("12 effective cuts accepted", r["status"] == "solution"
      and r["missed_cuts"] == 0
      and r["solutions"][0]["kept_cuts"] == list(range(1, 13))
      and r["solutions"][0]["fragment_lengths"] == [1] * 13, str(r)[:200])
expect_error("13 effective cuts rejected", {
    "sequence": "A" * 14, "topology": "linear",
    "enzymes": [{"name": "E", "recognition": "AA", "cut_offset": 1}],
    "observed_lengths": [1], "tolerance": 0,
}, "有效切点")
# 端点切点不计入上限：原始切点 0..13 共 14 个，有效切点仍为 1..12
r = run_infer_partial_digest({
    "sequence": "A" * 13, "topology": "linear",
    "enzymes": [
        {"name": "E0", "recognition": "A", "cut_offset": 0},
        {"name": "E1", "recognition": "A", "cut_offset": 1},
        {"name": "E2", "recognition": "AA", "cut_offset": 1},
    ],
    "observed_lengths": [1] * 13, "tolerance": 0,
})
check("endpoint cuts not counted toward limit",
      r["status"] == "solution" and r["missed_cuts"] == 0
      and r["solutions"][0]["kept_cuts"] == list(range(1, 13)), str(r)[:200])

# --- 非法输入：全部抛出 DigestError（HTTP 层映射为 400） ---
expect_error("payload not dict", ["x"], "JSON 对象")
expect_error("missing sequence",
             {"topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [1], "tolerance": 0}, "sequence")
expect_error("illegal base",
             {"sequence": "ACGN", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [1], "tolerance": 0}, "非法碱基")
expect_error("bad topology",
             {"sequence": "ACGT", "topology": "ring", "enzymes": [ECORI],
              "observed_lengths": [1], "tolerance": 0}, "拓扑")
expect_error("enzyme bad offset",
             {"sequence": "ACGT", "topology": "linear",
              "enzymes": [{"name": "E", "recognition": "AA", "cut_offset": 9}],
              "observed_lengths": [1], "tolerance": 0}, "越界")
expect_error("missing enzymes",
             {"sequence": "ACGT", "topology": "linear",
              "observed_lengths": [1], "tolerance": 0}, "enzymes")
expect_error("missing observed",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "tolerance": 0}, "observed_lengths")
expect_error("observed not list",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": 5, "tolerance": 0}, "observed_lengths")
expect_error("observed empty",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [], "tolerance": 0}, "数量")
expect_error("observed 14 items",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [1] * 14, "tolerance": 0}, "数量")
expect_error("observed zero",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [3, 0], "tolerance": 0}, "正整数")
expect_error("observed negative",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [-2], "tolerance": 0}, "正整数")
expect_error("observed float",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [1.5], "tolerance": 0}, "正整数")
expect_error("observed bool",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [True], "tolerance": 0}, "正整数")
expect_error("observed string",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": ["3"], "tolerance": 0}, "正整数")
expect_error("missing tolerance",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [4]}, "tolerance")
expect_error("tolerance negative",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [4], "tolerance": -1}, "tolerance")
expect_error("tolerance float",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [4], "tolerance": 0.5}, "tolerance")
expect_error("tolerance bool",
             {"sequence": "ACGT", "topology": "linear", "enzymes": [ECORI],
              "observed_lengths": [4], "tolerance": True}, "tolerance")

# --- 随机等价性：实现结果必须与朴素枚举参考完全一致，并满足各项不变量 ---
random.seed(20260926)
mismatches = []
for case_i in range(300):
    n = random.randint(4, 24)
    seq = "".join(random.choice("ACGT") for _ in range(n))
    topology = random.choice(["linear", "circular"])
    enzymes = []
    for e in range(random.randint(0, 2)):
        L = random.randint(1, 3)
        enzymes.append({
            "name": f"E{e}",
            "recognition": "".join(random.choice("ACGT") for _ in range(L)),
            "cut_offset": random.randint(0, L),
        })

    effective = effective_cuts(seq, topology, enzymes)
    payload = {
        "sequence": seq, "topology": topology, "enzymes": enzymes,
        "observed_lengths": [1], "tolerance": 0,
    }
    if len(effective) > 12:
        # 超限必须报 DigestError（HTTP 400）
        try:
            run_infer_partial_digest(payload)
            mismatches.append((case_i, "expected DigestError for >12 cuts"))
        except DigestError:
            pass
        continue

    # 以一个随机保留子集的谱为基准构造观测，再以一定概率扰动，
    # 使有解与无解情形都被覆盖
    kept = [c for c in effective if random.random() < 0.6]
    observed = ref_profile(n, kept, topology)
    if random.random() < 0.4:
        mode = random.randint(0, 2)
        if mode == 0 and observed:
            i = random.randrange(len(observed))
            observed[i] = max(1, observed[i] + random.choice((-2, -1, 1, 2)))
        elif mode == 1 and len(observed) > 1:
            observed.pop(random.randrange(len(observed)))
        elif len(observed) < 13:
            observed.append(random.randint(1, n))
    tolerance = random.randint(0, 2)
    payload["observed_lengths"] = observed
    payload["tolerance"] = tolerance

    got = run_infer_partial_digest(payload)
    want = reference(seq, topology, enzymes, observed, tolerance)
    if got != want:
        mismatches.append((case_i, payload, got, want))
        continue

    # 不变量：漏切数、谱和、升序、字典序、保留数一致
    if got["status"] == "solution":
        k = len(effective)
        kept_lists = [s["kept_cuts"] for s in got["solutions"]]
        ok = (
            got["missed_cuts"] == k - len(kept_lists[0])
            and got["solution_count"] == len(got["solutions"])
            and all(len(kl) == len(kept_lists[0]) for kl in kept_lists)
            and kept_lists == sorted(kept_lists)
            and all(all(c in effective for c in kl) for kl in kept_lists)
            and all(
                s["fragment_lengths"] == sorted(s["fragment_lengths"])
                and sum(s["fragment_lengths"]) == n
                for s in got["solutions"]
            )
        )
        if not ok:
            mismatches.append((case_i, "invariant", payload, got))

check("randomized equivalence with brute force (300 cases)",
      not mismatches, str(mismatches[:2])[:400])

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
