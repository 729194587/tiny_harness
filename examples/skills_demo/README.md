# Minimal Skills demo

This directory is a complete TinyHarness workspace containing one Skill. From
the repository root, run:

```powershell
python -m tiny_harness `
  "Use the review Skill to inspect this workspace" `
  --workspace examples/skills_demo
```

The first model request receives only the `review` name and description. The
full `skills/review/SKILL.md` content enters the conversation only if the model
calls `load_skill` with `{"name":"review"}`.
