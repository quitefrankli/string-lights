* Minimise token usage

* Don't poll or re-read: For background tasks, wait for completion once rather than repeatedly reading output files.

* Skip redundant verification: After a tool succeeds without error, don't re-read the result to confirm.

* Match verbosity to task complexity: Routine ops (merge, deploy, simple file edits) need minimal commentary. Save detailed explanations for complex logic, architectural decisions, or when asked.

* One tool call, not three: Prefer a single well-constructed command over multiple incremental checks.

* Don't narrate tool use: Skip "Let me read the file" or "Let me check the status" ? just do it.

* Minimise comments/docstrings, only add comments when they are truly helpful i.e. deep functions, surprsing logic. Don't add any readme.md files or other documentation files unless explicitly asked to.

* start every new session with "CLAUDE.md read!"