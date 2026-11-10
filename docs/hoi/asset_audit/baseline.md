# M0 — baseline

Target: /home/han/pyx/SMHOI (the separate sibling SuperMimic-Reproduced is not the target).

Full plan read. Scope follows user and Part A; Part D numbering conflicts. No training or reference generation.

Initial target: `ba61053`, only untracked user plan. Initial source: `114f766`, two untracked user files recorded in baseline.json. No AGENTS.md found. All input paths, hashes, sizes, branches and full SHAs are in baseline.json. No reset/clean/stash/checkout. Inputs and source repository remain read-only.

Runtime shell sandbox fails at bwrap loopback initialization; commands run through reviewed escalation. Only new audit files in SMHOI are staged.
