# Releasing `@mupt-ai/dari-router`

Router package releases are published from the public
[`mupt-ai/dari-router`](https://github.com/mupt-ai/dari-router) mirror, not from
the private monorepo.

1. Merge the versioned package changes into the monorepo `main` branch without
   adding an unpublished version to registry consumers.
2. Run the monorepo's **Mirror public packages** workflow on `main`.
3. Confirm the public mirror contains the expected version and commit.
4. Run the mirror's **Publish npm package** workflow from its `main` branch.
5. Confirm the version and its exported subpaths on npm.
6. In a follow-up change, update registry consumers and their lockfiles to the
   published version.

The publish workflow rejects non-`main` refs and versions already present on
npm. It runs typechecking, tests, a packed-package smoke test, and `npm publish`
with provenance. The npm package must have a trusted publisher configured for
repository `mupt-ai/dari-router` and workflow `publish-npm.yml`; no long-lived
npm token is used.

Never update a consumer to an unpublished version. Package publication is not
atomic with a monorepo merge, and npm versions are immutable.
