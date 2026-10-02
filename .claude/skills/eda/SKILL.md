---
name: eda
description: Inspect a raw or processed sensor file and report statistical properties without loading massive dumps into context.
argument-hint: "[path/to/data_sample]"
---
Write and run a short inline Python check on the file `$ARGUMENTS`:

1. Read only metadata or first 1,000 rows (`head`).
2. Report:
   - Sampling rate / timestamp increments.
   - Total length per run.
   - Missing values / NaN / infinite entries.
   - Channel names, min, max, mean, and standard deviation.
   - Class balance (Normal vs. Fault conditions).
3. Do not print raw arrays. Summarize in a concise markdown table.