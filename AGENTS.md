# Running the app and tests

Always use `uv run` to run the app, tests, and project Python commands so they use the project's environment and dependencies.

- Run the app with `uv run string-lights <command> [options]`.
- Run the full test suite with `uv run python -m pytest`; add test paths or pytest options as needed.
- Run other Python commands with `uv run python ...`.

Do not invoke the app, `python`, or `pytest` directly using the system environment.

# Project data

Store each input clip and its related files under `data/projects/<project>/`:

- `source.mov` / `source.mp4`: original and prepared video (MP4 is preferred when both exist).
- `profile.json`, `profile.npz`: project-specific annotation and compiled profile. Alternate views also belong at project level, e.g. `profile-neck.npz`.
- `transcription.npy`, `lyrics.json`: string highlights and lyric annotations.
- `tracking/projections.npz`: cached planar tracking.
- `masks/masks.npz`, `masks/metadata.npz`, `masks/seed.png`: hand masks and seed.
- `renders/strings.mp4`, `renders/strings-lyrics.mp4`: finished exports.
- `tmp/`: previews, experimental renders, trial caches, diagnostic output, and other temporary project files.

AI agents must put project experiments in `data/projects/<project>/tmp/`, using explicit CLI input/output overrides. Do not overwrite the project's source, profile, caches, or finished renders for an experiment. Use `data/tmp/` only for temporary work spanning projects. Do not create separate previews, experiments, or archive folders.

Commands accept a project name or an explicit video path. Use `--project-dir` to select the default artifact directory for an external input, and `--output` plus the command's artifact options to give files arbitrary paths/names. `--projects-dir` before the subcommand changes the project collection root, including for the web editors. Inspect `--help` for available overrides. A custom mask output uses `<output-stem>.meta.npz` for its sidecar unless metadata is explicitly specified.

Launch the tabs editor with `uv run string-lights tabs-editor <project>`; a project name is required and the editor is scoped to that project.
