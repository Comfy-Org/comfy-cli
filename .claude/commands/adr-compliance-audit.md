# ADR Compliance Audit

Audit the current diff, or a specified pull request, against `docs/adr/`.

1. Gather the base-to-head diff and changed-file list.
2. List ADR files, excluding `TEMPLATE.md`.
3. Read each ADR's Status, Decision, and Consequences sections.
4. Report only contradictions introduced by the diff.
5. Separate Accepted-ADR violations from Proposed-ADR directional concerns.
6. Identify architecture-significant decisions in the diff that need a new or
   amended ADR.

For each finding, provide the changed file and line, ADR identifier and link,
the violated invariant, and a concrete remediation. End with compliant patterns
and explicitly state when there are no findings.
