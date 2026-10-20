# Security policy

beatlm is a library for running neural networks on a GPU, maintained by one
person. This document says where to send a vulnerability report, what happens
after you do, and what is in scope.

## Reporting

**Use GitHub's private vulnerability reporting on this repository** — the
"Report a vulnerability" button under the Security tab. It opens a private
thread visible only to the maintainer, so the report never sits in a public
issue while a fix is being written.

Please do not open a public issue for a suspected vulnerability. If private
reporting is unavailable to you for any reason, open a public issue that says
only that you have a security report and asks for a contact — no details.

What helps most in a report: what an attacker can reach, the steps to reproduce
it, and the version or commit you saw it on. A proof of concept is welcome but
not required.

## What happens next

This is a single-maintainer project, so these are commitments a single person
can keep rather than a team SLA:

| Stage                                                      | Target                                                              |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| Acknowledge the report                                     | within 7 days                                                       |
| Initial assessment — is it a vulnerability, and how severe | within 30 days                                                      |
| Fix, or an explanation of why it will not be fixed         | discussed with you once assessed                                    |
| Public disclosure                                          | when the fix ships, or 90 days after the report, whichever is first |

The 90-day deadline is the industry norm and it runs regardless of whether a fix
is ready, because an unfixed issue that stays secret indefinitely protects
nobody but the project. If you need it held longer — a coordinated release with
another project, for example — say so and it can be agreed.

Reports are handled through GitHub security advisories, which means you are
credited on the published advisory unless you ask not to be. There is no bug
bounty; nothing here is offered in exchange for a report.

## Scope

beatlm runs inside the application that embeds it, natively or in a browser
page. In scope:

- the model loaders, which parse files the embedding application did not write
  (safetensors and other weight formats) — a malformed file that reaches an
  out-of-bounds read, an unbounded allocation, or a panic the API does not
  document;
- generated shaders — a model or graph that makes beatlm emit a kernel reading
  or writing outside the buffer ranges its plan assigned;
- GPU resource use that an input can drive past the limits the plan promised,
  causing device loss in the embedding application;
- a dependency with a known vulnerability that beatlm actually exercises.

Out of scope: anything that assumes the attacker already controls the embedding
application, and the behaviour of GPU drivers and browsers themselves — report
those upstream. A model whose outputs are merely wrong or harmful is a model
problem, not a beatlm vulnerability.

## Safe harbour

Good-faith research on your own installation is welcome and will not be met with
legal action. "Good faith" means: your own data, no third party's, no attempt to
reach anyone else's machine, and no public disclosure before the window above
closes.

## Handling a report that turns out to be a disclosure

If a report shows that protected content is already published — a credential in
history, personal data in a fixture — it stops being a vulnerability report and
follows [After a disclosure](PUBLIC_DEVELOPMENT.md#after-a-disclosure), which is
where the containment and notification steps live.
