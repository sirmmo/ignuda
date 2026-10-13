#!/bin/sh
# Run a command inside the ignuda dev container (Rust + Intel OpenCL CPU runtime).
exec docker run --rm -i -v /srv/ignuda:/src -v ignuda-cargo:/root/.cargo/registry -w /src \
  -e CARGO_TARGET_DIR=/src/target ignuda-dev sh -c "$*"
