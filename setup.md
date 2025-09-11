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
