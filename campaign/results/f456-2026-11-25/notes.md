# F-456 notes (2026-11-25)
09:05-09:14  A delegate briefed to read AGENTS.md and three F-entries before acting ran out of room and produced nothing
             (trace included).  Briefing error: read-before-write.  Its one finding kept: the MMLU gate is solo-only.
09:15-09:20  The trace is one line: `coh_b_drive = drive;` (generate.cpp:12669), on the first round only.  Guard built.
09:21-09:26  Smoke legs pass on 0aec007e; the armed telemetry leg reads 26.9% on the link = one cohort at 0.50, one at
             0.35: the guard set only A.  Fixed (B and C set every round); 783ecba4 reads 36.6%.
09:28-09:47  Env ladder, control = variable unset: conc -3.5% / +1.8% (mean -0.9%); M1 -2.0% (noise by construction);
             M3b flat.  The ladder's own telemetry: share 19% -> 38-39.5% on armed boots, each CPU expert +18%
             (66 -> 77-78 us), pool per round unchanged, window unchanged.
09:48-09:50  Ruling: closed with mechanism (the tiers share the memory system).  Guard removed; source == pre-F-456;
             rebuild == 7a4d3477b2975d6a (deterministic).
