# Eval Router

An automatic managed router that imports an eval scorecard into its selection context.

The eval ID is organization-specific, so replace `evl_your_scorecard` in `router.yml` with an ID returned by:

```bash
dari auth login
dari eval list
```

Then create the router:

```bash
dari router create ./router.yml
```

The scorecard must contain scores for the enabled model and reasoning-level pairs you want the selector to consider. Keep organization-specific IDs and scores out of public example files.
