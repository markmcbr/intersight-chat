SHELL := /bin/bash
COMPOSE ?= docker compose
MODEL ?= mistral-small3.1:24b

.PHONY: up up-gpu down restart logs ps status build pull-model ensure-default-model shell-app shell-ollama clean

## Bring the stack up (CPU only). Works on any Docker host. For GPU servers,
## use `make up-gpu` instead. Automatically pulls + pre-warms $(MODEL) so the
## first user prompt has zero cold-start cost.
up:
	$(COMPOSE) up -d --build
	@$(MAKE) ensure-default-model

## Bring the stack up with NVIDIA GPU passthrough for Ollama.
## Requires the NVIDIA Container Toolkit on the host. This is the recommended
## target for production / lab GPU servers. Automatically pulls + pre-warms
## $(MODEL) into VRAM so the first user prompt has zero cold-start cost.
up-gpu:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
	@$(MAKE) ensure-default-model

## Stop and remove containers (keeps the model volume).
down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build

## Pull a model into the running ollama container. Override with MODEL=...
## (Same MODEL var that ensure-default-model uses, so `make pull-model
## MODEL=foo` and then `make up-gpu MODEL=foo` are self-consistent.)
pull-model:
	$(COMPOSE) exec -T ollama ollama pull $(MODEL)

## Idempotent: wait for Ollama to be ready, pull $(MODEL) if missing, then
## pre-warm it into VRAM via the native /api/generate endpoint with an
## empty prompt and keep_alive=24h. Subsequent runs are near-instant
## (the model is already pulled in the ollama_models volume, and Ollama
## treats the warm-up call as a no-op refresh of the keep-alive timer).
## Called automatically by `make up` and `make up-gpu` — no need to
## invoke directly unless you want to re-warm after switching MODEL.
ensure-default-model:
	@echo "==> Waiting for Ollama to be ready..."
	@for i in $$(seq 1 30); do \
		$(COMPOSE) exec -T ollama ollama list >/dev/null 2>&1 && break; \
		sleep 2; \
	done
	@if ! $(COMPOSE) exec -T ollama ollama list >/dev/null 2>&1; then \
		echo "Ollama did not become ready within 60s. Check 'docker logs intersight-chat-ollama'." >&2; \
		exit 1; \
	fi
	@echo "==> Ensuring $(MODEL) is pulled (first-time download may take several minutes)..."
	@$(COMPOSE) exec -T ollama ollama pull $(MODEL)
	@echo "==> Pre-warming $(MODEL) into VRAM (keep_alive=24h)..."
	@# Kick the warm-up off in detached mode (don't block on the response
	@# stream), then poll `ollama ps` until the model actually appears in
	@# VRAM. This gives the operator a visible progress indicator AND a
	@# hard guarantee that when the prompt returns, the model is loaded.
	@$(COMPOSE) exec --detach -T ollama ollama run $(MODEL) "ok" >/dev/null 2>&1 || true
	@printf "==> Loading"; \
	ready=0; \
	for i in $$(seq 1 60); do \
		if $(COMPOSE) exec -T ollama ollama ps 2>/dev/null | grep -q "$(MODEL)"; then \
			ready=1; break; \
		fi; \
		printf "."; \
		sleep 2; \
	done; \
	if [ $$ready -eq 1 ]; then \
		printf " ready (%ds).\n" $$((i * 2)); \
	else \
		printf " timed out after 120s.\n"; \
		echo "    The model may still be loading. Check 'make status' or"; \
		echo "    'docker logs intersight-chat-ollama' for details."; \
	fi
	@echo "==> App: http://<host>:8501"

## Show what models are currently loaded into VRAM (and their keep-alive
## remaining), and the docker-compose service state. Use this after
## `make up-gpu` to see when the background pre-warm has finished —
## $(MODEL) will appear in the `ollama ps` output once loaded.
status:
	@echo "==> Container status:"
	@$(COMPOSE) ps
	@echo ""
	@echo "==> Models loaded in VRAM (empty list = none loaded yet):"
	@$(COMPOSE) exec -T ollama ollama ps

shell-app:
	$(COMPOSE) exec app bash

shell-ollama:
	$(COMPOSE) exec ollama bash

## Full teardown including downloaded models. Use with care.
clean:
	$(COMPOSE) down -v
