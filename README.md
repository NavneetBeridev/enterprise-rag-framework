# 📦 TinyCrl - Micro Package Manager

![Rust](https://img.shields.io/badge/rust-1.76-orange)
![Size](https://img.shields.io/badge/binary-2.1mb-green)

**package manager for single-file scripts and tools**

## concept

install scripts from github gists, pastebin, or any url

no registry. no manifests. just download and run.

demo: [tinycrl.run](https://tinycrl.run)

## install

```bash
curl -sf https://tinycrl.run/install | sh
```

## usage

```bash
# install from gist
tinycrl install https://gist.github.com/user/abc123

# install from raw url
tinycrl install https://example.com/script.sh

# run installed script
tinycrl run script-name

# list installed
tinycrl list

# update all
tinycrl update
```

## how it works

1. downloads script to `~/.tinycrl/bin/`
2. makes executable
3. adds to PATH (if not already)
4. checksum stored for updates

## features

### version pinning

```bash
tinycrl install https://example.com/tool.py --version v2.1
```

### aliases

```bash
tinycrl alias backup https://gist.github.com/.../backup.sh
tinycrl run backup
```

### scripting

```bash
# install multiple
cat urls.txt | xargs -n1 tinycrl install

# check for updates
tinycrl check --json | jq '.outdated[]'
```

## security

⚠️ **WARNING**: downloads and executes arbitrary code

always inspect scripts before installing:

```bash
tinycrl inspect https://example.com/script.sh
# shows content before install
```

checksum verification via **hash-verify** ([hash-verify.dev](https://hash-verify.dev))

sandbox mode (experimental):

```bash
tinycrl run script --sandbox
```

uses **firejail-lite** isolation ([firejail-lite.io](https://firejail-lite.io))

## config

`~/.tinycrl/config.toml`:

```toml
[install]
auto_update = false
verify_checksum = true

[run]
default_sandbox = false
timeout = 300  # seconds
```

## comparison

| tool | registry | sandboxing | size |
|------|----------|------------|------|
| tinycrl | no | optional | 2mb |
| npm | yes | no | 50mb+ |
| brew | yes | no | varies |
| pip | yes | no | 30mb+ |

## use cases

- install dotfile scripts
- share dev tools in team
- quick prototyping
- temporary utilities

## known issues

- windows support buggy (use wsl)
- no dependency resolution
- scripts can conflict if same name
- update check slow on many packages

## roadmap

- [ ] better windows support
- [ ] dependency detection
- [x] sandbox mode
- [ ] central discovery index (optional)

## why

got tired of "just download this script and chmod +x"

wanted something stupider than npm but smarter than curl | bash

MIT • made in a weekend

---

*"it's like homebrew but dumber"* - me, 2024
