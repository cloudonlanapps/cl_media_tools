# INTERNALS.md - Known Issues Registry

This document tracks known linting warnings, type checker issues, and test skips that are intentionally kept in the codebase. Each entry must include justification and removal criteria.

**Last Updated**: 2025-12-16

---

## Format

Each entry must follow this format:

```markdown
### [Category] Issue Title

- **File**: `path/to/file.py`
- **Tool**: ruff | basedpyright | pytest
- **Issue**: Error code or warning message
- **Reason**: Clear, justified explanation for keeping the warning/skip
- **Removal Criteria**: Action required to resolve the issue
- **Added**: YYYY-MM-DD
```

---

## Active Issues

### [Example] Type Ignore for Dynamic Plugin Loading

- **File**: `src/cl_ml_tools/common/plugin_loader.py` (example)
- **Tool**: basedpyright
- **Issue**: `type: ignore[attr-defined]` on line 42
- **Reason**: Dynamic plugin loading via entry points requires runtime attribute access that cannot be statically verified
- **Removal Criteria**: Refactor to use Protocol classes for static type checking of plugin interfaces
- **Added**: 2025-12-16

---

## Resolved Issues

(Issues that were previously documented but have been resolved)

---

## Guidelines

1. **Never skip errors without documentation** - All type ignores, test skips, and ignored linting rules must be documented here
2. **Provide context** - Explain why the issue cannot be immediately resolved
3. **Set removal criteria** - Define what needs to happen to remove the exception
4. **Review regularly** - During Phase 5 QA, review all entries and attempt to resolve
5. **Keep it current** - Remove entries when issues are resolved

---

## Statistics

- **Total Active Issues**: 0
- **Total Resolved Issues**: 0
- **Last Review**: 2025-12-16
