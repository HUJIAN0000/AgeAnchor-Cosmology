# Migrating from the PRL-era code

The old numbered scripts and `cosmoc.py` are replaced in the current tree. Their
source remains available at commit `d2dcc86173fd9932b6af8245b28c71e3bf6027af`.
Keep previous commits and any existing releases; use an ordinary new commit for
version 2 rather than force-pushing or deleting repository history.

| Previous workflow | Version 2 replacement |
| --- | --- |
| Top-level `01_...` through `09_...` scripts | `RUN_REPRO.py` and common likelihood modules |
| Manual covariance unzip | Automatic unpacking and SHA-256 checks |
| Per-script figure generation | `python RUN_REPRO.py figures` |
| Independent numerical checks | `python RUN_REPRO.py validate` |
| Fixed script execution for new chains | `sample --suite ... --output ...` with frozen settings |
| Single plotting helper at root | Unchanged `basetools/cosmo_tools.py` |

The data and likelihood assumptions have changed, so saved chains from the old
numbered scripts are not interchangeable with version 2 products. In particular,
the current release uses DESI DR2 and the stated stellar-versus-cosmic age
definitions and formation treatment. Refer to the versioned configuration and
source manifests when comparing results.
