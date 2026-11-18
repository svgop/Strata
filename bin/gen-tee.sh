#!/usr/bin/env bash
# diagnostic wrapper: tee the engine's stdin (the server's GEN lines) to logs/gen-capture.txt, then exec the
# engine.  The server spawns (exe, "--serve", <engine>, flags...): skip the wrapper's own "--serve" arg.
tee -a /home/svg/LLM/strata-custom-2/logs/gen-capture.txt | exec "${@:2}"
