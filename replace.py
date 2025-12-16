import sys

import libcst as cst
from libcst import matchers as m

ALLOWED_KEYS = {"status", "task_output", "error"}


class DictToTaskResult(cst.CSTTransformer):
    def leave_Dict(self, original_node, updated_node):
        found: dict[str, cst.BaseExpression] = {}

        for element in updated_node.elements:
            # Reject **, comprehensions, etc.
            if element is None or element.key is None:
                return updated_node

            # Only allow string literal keys
            if not isinstance(element.key, cst.SimpleString):
                return updated_node

            key = element.key.evaluated_value
            if key not in ALLOWED_KEYS:
                # ❌ extra key like "job_id"
                return updated_node

            found[key] = element.value

        # Must have "status"
        if "status" not in found:
            return updated_node

        # Build TaskResult(...)
        args = [cst.Arg(keyword=cst.Name("status"), value=found["status"])]

        if "task_output" in found:
            args.append(cst.Arg(keyword=cst.Name("task_output"), value=found["task_output"]))

        if "error" in found:
            args.append(cst.Arg(keyword=cst.Name("error"), value=found["error"]))

        return cst.Call(
            func=cst.Name("TaskResult"),
            args=args,
        )


def transform_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    module = cst.parse_module(source)
    modified = module.visit(DictToTaskResult())

    if modified.code != source:
        print(f"✔ modified: {path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(modified.code)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: replace_taskresult.py <file1.py> <file2.py> ...")
        sys.exit(1)

    for path in sys.argv[1:]:
        transform_file(path)


if __name__ == "__main__":
    main()
