#!/usr/bin/env python3
"""campaign/patch_f286_ternary_v2.py — the F-284 ternary re-anchored to the C2a-free base
(generate.cpp@9e5366c6 composition, F-285a standing).  Differences from v1: the poll loop
and post-loop finishes target the ORIGINAL simple layout (no fa/fb — that block was the
F-281 relocation that hung the conc lane and is gone); C's stage goes between B's stage and
A's go from the start (v1's capture-under-flight lesson).  Same arm: STRATA_COHORTS=3,
o.batch >= 9, engagement n_active > 2*batch/3; default behavior byte-identical."""
from pathlib import Path

P = Path("/home/svg/LLM/strata-custom-2/src/program/generate.cpp")
src = P.read_text()
n0 = len(src)

def rep(old, new):
    global src
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old[:80]!r}"
    src = src.replace(old, new)

# 1. serve-scope ver_c (beside the local ver_b at :9102)
rep("""        strata::core::Verifier ver_b;
        SplitDrive split_drive_b;
""",
    """        strata::core::Verifier ver_b;
        strata::core::Verifier ver_c;   // F-284/F-286: the third batch-cohort verifier (STRATA_COHORTS=3, >=9 wide)
        SplitDrive split_drive_b;
""")

# 2. round-scope state
rep("    int cohorts = 1;\n    Drive coh_b_drive;\n    bool coh_b_ready = false;\n",
    "    int cohorts = 1;\n    Drive coh_b_drive;\n    bool coh_b_ready = false;\n"
    "    Drive coh_c_drive;                      // F-284/F-286: the third cohort's drive (lazily cloned)\n"
    "    bool coh_c_ready = false;\n")

# 3. init: ver_c inside the ver_b success path
rep("""                    cohorts = 2;
                    std::fprintf(stderr, "strata serve: STRATA_COHORTS=2 on: slots [0,%d) on verifier A, [%d,%d) on B\\n",
                                 o.batch / 2, o.batch / 2, o.batch);
""",
    """                    cohorts = 2;
                    std::fprintf(stderr, "strata serve: STRATA_COHORTS=2 on: slots [0,%d) on verifier A, [%d,%d) on B\\n",
                                 o.batch / 2, o.batch / 2, o.batch);
                    // F-284/F-286 (STRATA_COHORTS=3, o.batch >= 9): the third cohort lane.  3+3+3 keeps every
                    // window 6 rows at 9-wide (5+4 hits w.S+2<=kVerifyMaxT at 4 slots and ROTATES the 5th
                    // idle every round - the F-273 9c~6-parity cap).  Needs STRATA_ELASTIC_S_MIN=9 at b9
                    // (the admission-ramp expert-tail release hangs otherwise - F-285's find, config-fixed).
                    if (std::atoi(v8c) >= 3 && o.batch >= 9) {
                        ver_c.set_stage(0, -1, nullptr, nullptr);
                        if (batch_mtp) ver_c.set_batch_graph_limit(64);
                        ver_c.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
                        if (!ver_c.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr,
                                        batch_mtp ? strata::kernels::kVerifyMaxT : std::max(o.spec, o.batch), cerr_)) {
                            std::fprintf(stderr, "strata serve: STRATA_COHORTS=3 off: the third verifier: %s%s\\n",
                                         cerr_.c_str(), vram_free_note().c_str());
                        } else {
                            std::vector<strata::core::SessionState*> cptrs3;
                            for (auto& u : bslot_ss[0]) cptrs3.push_back(u.get());
                            if (!ver_c.init_slots(cptrs3, cerr_)) {
                                std::fprintf(stderr, "strata serve: STRATA_COHORTS=3 off: %s\\n", cerr_.c_str());
                            } else {
                                cohorts = 3;
                                const int t1 = o.batch / 3, t2 = (2 * o.batch) / 3;
                                std::fprintf(stderr, "strata serve: STRATA_COHORTS=3 on: slots [0,%d) A [%d,%d) B [%d,%d) C\\n",
                                             t1, t1, t2, t2, o.batch);
                            }
                        }
                    }
""")

# 4. round gate
rep("            if (cohorts == 2 && n_active > o.batch / 2) {   // both cohorts must have work; pair scale and below takes the untouched path below\n",
    "            if (cohorts >= 2 && n_active > o.batch / 2) {   // both cohorts must have work; pair scale and below takes the untouched path below\n")

# 5. split + ternary + next_c
rep("                const int split = o.batch / 2;\n                static size_t next_a = 0, next_b = 0;   // each cohort rotates within its own slot span\n",
    "                const int split = o.batch / 2;\n"
    "                const int split2 = (2 * o.batch) / 3;   // F-284/F-286: the ternary's [2b/3, b) span start\n"
    "                const bool ternary = cohorts == 3 && n_active > split2;   // all three cohorts populated\n"
    "                static size_t next_a = 0, next_b = 0, next_c = 0;   // each cohort rotates within its own slot span\n")

# 6. collect + wc
rep("""                CohortWin wa, wb;
                collect(0, split, wa, next_a);
                collect(split, o.batch, wb, next_b);
                if (wa.S + wb.S == 0) return true;
""",
    """                CohortWin wa, wb, wc;
                collect(0, split, wa, next_a);
                if (ternary) collect(split, split2, wb, next_b);
                else collect(split, o.batch, wb, next_b);
                if (ternary) collect(split2, o.batch, wc, next_c);
                if (wa.S + wb.S + wc.S == 0) return true;
""")

# 7. C drive clone
rep("""                if (!coh_b_ready) {   // every drive.d setup write is long done by the first round
                    coh_b_drive = drive;
                    coh_b_drive.d.plan = ver_b.plan_sink();
                    coh_b_drive.calls = 0;
                    coh_b_drive.cpu_ms = 0;
                    coh_b_ready = true;
                }
""",
    """                if (!coh_b_ready) {   // every drive.d setup write is long done by the first round
                    coh_b_drive = drive;
                    coh_b_drive.d.plan = ver_b.plan_sink();
                    coh_b_drive.calls = 0;
                    coh_b_drive.cpu_ms = 0;
                    coh_b_ready = true;
                }
                if (ternary && !coh_c_ready) {   // F-284/F-286: the third drive, at the first ternary round
                    coh_c_drive = drive;
                    coh_c_drive.d.plan = ver_c.plan_sink();
                    coh_c_drive.calls = 0;
                    coh_c_drive.cpu_ms = 0;
                    coh_c_ready = true;
                }
""")

# 8. round log; C STAGE between B's stage and A's go (capture must precede any launch); C GO after B's go
rep("""                std::fprintf(stderr, "cohorts: round A.S=%d B.S=%d\\n", wa.S, wb.S);
""",
    """                if (ternary) std::fprintf(stderr, "cohorts: round A.S=%d B.S=%d C.S=%d\\n", wa.S, wb.S, wc.S);
                else std::fprintf(stderr, "cohorts: round A.S=%d B.S=%d\\n", wa.S, wb.S);
""")
rep("""                if (wb.S > 0 && !ver_b.batch_stage_rows(wb.rows, wb.S, wb.tok, wb.pos, err)) {
                    std::fprintf(stderr, "cohorts: B stage: %s\\n", err.c_str()); std::printf("ERR cohort B stage: %s\\n", err.c_str()); std::fflush(stdout);
                    return false;
                }
""",
    """                if (wb.S > 0 && !ver_b.batch_stage_rows(wb.rows, wb.S, wb.tok, wb.pos, err)) {
                    std::fprintf(stderr, "cohorts: B stage: %s\\n", err.c_str()); std::printf("ERR cohort B stage: %s\\n", err.c_str()); std::fflush(stdout);
                    return false;
                }
                if (ternary && wc.S > 0 && !ver_c.batch_stage_rows(wc.rows, wc.S, wc.tok, wc.pos, err)) {
                    // C stages BEFORE any go: its first-round lazy graph capture must happen while NO
                    // window is in flight (v1's capture-under-flight hung layer 0, #267)
                    std::fprintf(stderr, "cohorts: C stage: %s\\n", err.c_str()); std::printf("ERR cohort C stage: %s\\n", err.c_str()); std::fflush(stdout);
                    return false;
                }
""")
rep("""                if (wb.S > 0 && !ver_b.batch_go(err, false)) {
                    std::fprintf(stderr, "cohorts: B launch: %s\\n", err.c_str()); std::printf("ERR cohort B launch: %s\\n", err.c_str()); std::fflush(stdout);
                    return false;
                }
""",
    """                if (wb.S > 0 && !ver_b.batch_go(err, false)) {
                    std::fprintf(stderr, "cohorts: B launch: %s\\n", err.c_str()); std::printf("ERR cohort B launch: %s\\n", err.c_str()); std::fflush(stdout);
                    return false;
                }
                if (ternary && wc.S > 0 && !ver_c.batch_go(err, false)) {
                    std::fprintf(stderr, "cohorts: C launch: %s\\n", err.c_str()); std::printf("ERR cohort C launch: %s\\n", err.c_str()); std::fflush(stdout);
                    return false;
                }
""")

# 9. poll loop: the ORIGINAL simple loop gains lc (no fa/fb - that block is gone from this base)
rep("""                bool la = wa.S > 0, lb = wb.S > 0;
                while (la || lb) {
                    if (la) {
                        const int r = ver.batch_poll(win_pool_fn, &drive, err);
                        if (r < 0) { std::fprintf(stderr, "cohorts: A poll: %s\\n", err.c_str()); std::printf("ERR cohort A: %s\\n", err.c_str()); std::fflush(stdout); return false; }
                        la = (r == 0);
                    }
                    if (lb) {
                        const int r = ver_b.batch_poll(win_pool_fn, &coh_b_drive, err);
                        if (r < 0) { std::fprintf(stderr, "cohorts: B poll: %s\\n", err.c_str()); std::printf("ERR cohort B: %s\\n", err.c_str()); std::fflush(stdout); return false; }
                        lb = (r == 0);
                    }
                }
""",
    """                bool la = wa.S > 0, lb = wb.S > 0, lc = ternary && wc.S > 0;
                while (la || lb || lc) {
                    if (la) {
                        const int r = ver.batch_poll(win_pool_fn, &drive, err);
                        if (r < 0) { std::fprintf(stderr, "cohorts: A poll: %s\\n", err.c_str()); std::printf("ERR cohort A: %s\\n", err.c_str()); std::fflush(stdout); return false; }
                        la = (r == 0);
                    }
                    if (lb) {
                        const int r = ver_b.batch_poll(win_pool_fn, &coh_b_drive, err);
                        if (r < 0) { std::fprintf(stderr, "cohorts: B poll: %s\\n", err.c_str()); std::printf("ERR cohort B: %s\\n", err.c_str()); std::fflush(stdout); return false; }
                        lb = (r == 0);
                    }
                    if (lc) {
                        const int r = ver_c.batch_poll(win_pool_fn, &coh_c_drive, err);
                        if (r < 0) { std::fprintf(stderr, "cohorts: C poll: %s\\n", err.c_str()); std::printf("ERR cohort C: %s\\n", err.c_str()); std::fflush(stdout); return false; }
                        lc = (r == 0);
                    }
                }
""")

# 10. failed check
rep("""                if (drive.d.failed || coh_b_drive.d.failed) {
                    std::printf("ERR %s\\n", drive.d.failed && drive.d.fail ? drive.d.fail
                              : coh_b_drive.d.failed && coh_b_drive.d.fail ? coh_b_drive.d.fail
                              : "a cohort window failed");
                    return false;
                }
""",
    """                if (drive.d.failed || coh_b_drive.d.failed || (coh_c_ready && coh_c_drive.d.failed)) {
                    std::printf("ERR %s\\n", drive.d.failed && drive.d.fail ? drive.d.fail
                              : coh_b_drive.d.failed && coh_b_drive.d.fail ? coh_b_drive.d.fail
                              : coh_c_ready && coh_c_drive.d.failed && coh_c_drive.d.fail ? coh_c_drive.d.fail
                              : "a cohort window failed");
                    return false;
                }
""")

# 11. post-loop finishes + tail weight
rep("""                if (wa.S > 0 && !finish(ver, wa)) return false;   // an empty cohort has no window to
                if (wb.S > 0 && !finish(ver_b, wb)) return false; // commit (commit_slots would abort)
""",
    """                if (wa.S > 0 && !finish(ver, wa)) return false;   // an empty cohort has no window to
                if (wb.S > 0 && !finish(ver_b, wb)) return false; // commit (commit_slots would abort)
                if (ternary && wc.S > 0 && !finish(ver_c, wc)) return false;   // (the third tail)
""")
rep("""                bt_run += msd2(w0c, w1c);
                bt_commit += msd2(w1c, w2c) * 0.5;   // both tails ran inside this span; split evenly
                bt_emit += msd2(w1c, w2c) * 0.5;
""",
    """                bt_run += msd2(w0c, w1c);
                const double tail_w = (ternary && wc.S > 0) ? (1.0 / 3.0) : 0.5;   // tails inside this span; split evenly
                bt_commit += msd2(w1c, w2c) * tail_w;
                bt_emit += msd2(w1c, w2c) * tail_w;
""")

P.write_text(src)
print(f"F-286 ternary v2 applied: {n0} -> {len(src)} bytes (+{len(src)-n0})")
