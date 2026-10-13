#!/bin/sh
cd "/home/svg/LLM/strata-custom-2"
exec "/home/svg/LLM/strata-custom-2/.venv/bin/python" "/home/svg/LLM/strata-custom-2/serve/server.py" "--engine" "strata" "--config" "/home/svg/LLM/strata-custom-2/strata-q2_0.json" "--port" "8080" "--open"
