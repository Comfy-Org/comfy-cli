"""Validators for CLI options."""

import typer


class MutuallyExclusiveValidator:
    def __init__(self):
        self.group = []

    def reset_for_testing(self):
        self.group.clear()

    def validate(self, ctx: typer.Context, param: typer.CallbackParam, value):
        # Add cli option to group if it was called with a truthy value (for boolean flags)
        if value and param.name not in self.group:
            self.group.append(param.name)
        if len(self.group) > 1:
            raise typer.BadParameter(f"option `{param.name}` is mutually exclusive with option `{self.group.pop()}`")
        return value
