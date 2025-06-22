from typing import NamedTuple


class PRInfo(NamedTuple):
    number: int
    head_repo_url: str
    head_branch: str
    base_repo_url: str
    base_branch: str
    title: str
    user: str
    mergeable: bool

    @property
    def is_fork(self) -> bool:
        return self.head_repo_url != self.base_repo_url