# Security policy

## Supported versions

HFlow is pre-v1. Security fixes are made on the `main` branch and released for
the latest 0.2.x version when applicable.

| Version | Supported |
| --- | --- |
| Latest 0.2.x | Yes |
| Earlier 0.2.x | No; upgrade to the latest patch release |
| 0.1.x | Not part of this project; these releases predate the PyPI name transfer |

## Report a vulnerability privately

Do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability reporting](https://github.com/Hebbian-Robotics/hflow/security/advisories/new)
to share the details with the maintainers.

If that form is unavailable, contact a maintainer through the
[Hebbian Robotics organization](https://github.com/Hebbian-Robotics) without
including vulnerability details, and ask to establish a private channel.

Include, when available:

- the affected commit, version, command, or component;
- the impact and the conditions required to reproduce it;
- a minimal reproduction or proof of concept;
- relevant logs with tokens, credentials, private URLs, and robot data removed;
- any mitigation you have already identified.

Don't upload proprietary robot recordings or data containing people,
screens, badges, credentials, or customer information. Provide a synthetic
reproducer instead.

The maintainers validate the report, coordinate a fix and disclosure, and
credit reporters who want attribution. Response and remediation times depend
on severity and maintainer availability; this pre-v1 project does not promise
a fixed service-level agreement.
