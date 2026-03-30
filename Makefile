.PHONY: sync

sync:
	@git add -A
	@git commit -m "sync: $$(date '+%Y-%m-%d %H:%M')" 2>/dev/null || true
	@git push origin
	@mkdir -p ~/Documents/BerjQuant-Vault/Reference
	@cp ANTHONY/BERJQUANT_STATE.md ~/Documents/BerjQuant-Vault/Reference/BERJQUANT_STATE.md
	@echo "Synced to GitHub + Obsidian at $$(date '+%H:%M')"
