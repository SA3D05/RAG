

all: run

run:
	@ systemd-run --user --scope \
	-p MemoryMax=1G \
	-p CPUQuota=100% \
	uv run python3 -m src

