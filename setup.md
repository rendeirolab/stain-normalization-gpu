# How to setup the project

1. Install uv

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Setup Environment

```shell
uv sync
```

3. Setup pre-commit hook

```shell
uv run prek install
```


# How to make changes

It's not allowed to push to the main branch directly.
Please open a Pull request for review.

1. Make a new branch
2. Modify your code, make commits
3. Push to remote
4. Draft a pull request and wait for review

# Impelmentation details

1. The `fit` and `normalize` method should take in a cupy array with shape of (B, C, H, W)
2. Add your implementation to the tests
3. Add a benchmark script
4. Add a visualization script to run against the image in the `data` folder, compare it to existing implementation

# About test data

There is one `target.png` for the fit method and five `test_*.png` for the normalize method.
