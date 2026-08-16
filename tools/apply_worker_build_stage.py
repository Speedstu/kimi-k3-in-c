#!/usr/bin/env python3
from pathlib import Path


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 match, got {n}")
    return text.replace(old, new, 1)

root = Path(__file__).resolve().parents[1]

p = root / "Makefile"
s = p.read_text()
s = once(s,
'''CLI_SRC    := src/cli/k3_run.c
CLI_BIN    := $(BIN)/k3
''',
'''CLI_SRC    := src/cli/k3_run.c
CLI_BIN    := $(BIN)/k3
WORKER_SRC := src/cli/k3_worker.c
WORKER_BIN := $(BIN)/k3-worker
''', "make vars")
s = once(s, 'all: $(CLI_BIN)\n', 'all: $(CLI_BIN) $(WORKER_BIN)\n', "make all")
s = once(s,
'''$(CLI_BIN): $(CLI_SRC) $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(CLI_SRC) $(ENGINE_OBJ) -o $@ $(LDFLAGS)
''',
'''$(CLI_BIN): $(CLI_SRC) $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(CLI_SRC) $(ENGINE_OBJ) -o $@ $(LDFLAGS)

$(WORKER_BIN): $(WORKER_SRC) $(CLI_SRC) $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(WORKER_SRC) $(ENGINE_OBJ) -o $@ $(LDFLAGS)
''', "make worker rule")
s = once(s,
'''install: $(CLI_BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(CLI_BIN) $(DESTDIR)$(PREFIX)/bin/k3
''',
'''install: $(CLI_BIN) $(WORKER_BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(CLI_BIN) $(DESTDIR)$(PREFIX)/bin/k3
	install -m 755 $(WORKER_BIN) $(DESTDIR)$(PREFIX)/bin/k3-worker
''', "make install")
p.write_text(s)

p = root / "CMakeLists.txt"
s = p.read_text()
s = once(s,
'''add_executable(k3_cli src/cli/k3_run.c)
set_target_properties(k3_cli PROPERTIES OUTPUT_NAME k3)
target_link_libraries(k3_cli PRIVATE k3)
''',
'''add_executable(k3_cli src/cli/k3_run.c)
set_target_properties(k3_cli PROPERTIES OUTPUT_NAME k3)
target_link_libraries(k3_cli PRIVATE k3)

add_executable(k3_worker src/cli/k3_worker.c)
set_target_properties(k3_worker PROPERTIES OUTPUT_NAME k3-worker)
target_link_libraries(k3_worker PRIVATE k3)
''', "cmake worker target")
s = once(s, 'install(TARGETS k3_cli RUNTIME DESTINATION bin)\n',
         'install(TARGETS k3_cli k3_worker RUNTIME DESTINATION bin)\n', "cmake install")
p.write_text(s)
print('resident worker build wiring materialized')
