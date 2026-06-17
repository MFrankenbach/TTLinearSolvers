# Repository Rules

## General Best Practices

- Make the smallest change that fully solves the task.
- Prefer explicit assumptions, clear errors, and local verification over silent behavior or broad guesses.
- Keep edits scoped to the code paths the task actually touches.
- Avoid changing logs, notebooks, local outputs, data artifacts, or other research byproducts unless the task explicitly targets them.
- Strongly prefer running tests only specifically for features that changed.
- Don't repeat yourself! Make use of multiple dispatch and shared logic between different cases whenever possible.

## Repository Layout

- Public API or load-order changes may require updating `src/TTLinearSolvers.jl`, not only the file where new code is written.
- Main package tests live under `test/`.

## Error Handling

- Fail early with clear error messages, but avoid redundant checks or checks that would noticeably slow down hot paths.
- Include actual versus expected values in error messages.

## Documentation Requirements

### Docstrings

- The extent of a docstring should match the complexity of the function or type.
- Docstrings should explain the purpose, main usage, important assumptions, and non-obvious pitfalls.
- Internal functions and helpers should also have docstrings, especially when they are technical or complicated.
- When these details are part of the API contract, document conventions such as index or tag expectations, mutating behavior, and important keyword arguments.
- Reference other functions to avoid redundancy and repetitive documentation that easily gets outdated.

## Testing And Verification

- It is **forbidden** to weaken or exclude tests just to make the code pass again.
- Keep tests at least as strict as they already are; extend them when the change introduces meaningful new behavior or risk.
- Use the existing test structure in `test/runtests.jl` as the source of truth for relevant coverage.
- Keep ambiguity checks passing when changing method definitions or dispatch-heavy code.

## Dependency And Scope Discipline

- Add new dependencies only with clear justification.
- Prefer keeping experimental scripts, notebooks, and local analysis artifacts outside core package changes unless the task explicitly targets them.
- Treat untracked data files, logs, archives, and scratch outputs as user or research artifacts, not cleanup candidates.
- Never delete untracked data files, logs, archives, or scratch outputs. Edit them only when the task explicitly targets them.
- By default, edit only source files that the task justifies. Documentation and repository policy files may be edited when the user explicitly requests documentation or policy changes.
