# Releasing `@mupt-ai/dari-router`

Router package releases are published from the public
[`mupt-ai/dari-router`](https://github.com/mupt-ai/dari-router) mirror, not from
the private monorepo.

1. Update the package version and merge the package changes into the monorepo
   `main` branch.
2. Publish a normal Dari Mono release. The monorepo's **Mirror public packages**
   workflow copies that release's router subtree to the public mirror.
3. The mirror push automatically runs **Publish npm package**. Confirm the
   public mirror contains the expected commit and npm exposes the new version
   and exported subpaths.

For an out-of-band package release, manually run **Mirror public packages** on
monorepo `main`; the resulting mirror push uses the same automatic publish
path. Re-running either workflow is safe because an already-published version
is skipped.

The publish workflow accepts only the mirrored `main` branch. For unpublished
versions it runs typechecking, tests, a packed-package smoke test, and
`npm publish` with provenance. The npm package must have a trusted publisher
configured for repository `mupt-ai/dari-router` and workflow
`publish-npm.yml`; no long-lived npm token is used.

Private monorepo consumers use the local router source and do not wait for npm.
External registry consumers should adopt a version only after it is published;
npm versions are immutable.
