from typing import List

# from nirjas.binder import CommentSyntax, contSingleLines
# from nirjas.languages.java import javaExtractor


def remove_comments(code, preserve_lines=False) -> str:
    """Inspired by Nirja's comment removal code, this function removes comments from the code."""
    copy = True
    f = code.split("\n")
    lines = []
    for line in f:
        content = ""
        found = False
        if "/*" in line:
            pos = line.find("/*")
            while is_inside_string_literal(line, pos):
                pos = line.find("/*", pos + 2)
            if pos != -1:
                content = line[:pos].rstrip()
                line = line[pos:]
                copy = False
                found = True
        if "*/" in line:
            pos = line.rfind("*/")
            while is_inside_string_literal(line, pos):
                pos = line.rfind("*/", 0, pos - 1)
            if pos != -1:
                content = content + line[pos + 2 :]
                line = content
                copy = True
                found = True
        if "//" in line:
            leftmost_double_slash_idx = line.find("//")
            rightmost_double_slash_idx = line.rfind("//")

            if (
                leftmost_double_slash_idx - 1 >= 0
                and line[leftmost_double_slash_idx - 1] != ":"
            ):
                line = line[:leftmost_double_slash_idx]
            elif (
                rightmost_double_slash_idx - 1 >= 0
                and line[rightmost_double_slash_idx - 1] != ":"
            ):
                line = line[:rightmost_double_slash_idx]

            content = line
            found = True
        if not found:
            content = line
        if copy:
            lines.append(content)
        elif preserve_lines:
            lines.append("")

    return "\n".join(lines)


def is_inside_string_literal(s, index):
    in_single_quote = False
    in_double_quote = False
    escape = False

    for i in range(index + 1):
        c = s[i]

        if escape:
            escape = False
            continue

        if c == "\\":
            escape = True
        elif c == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif c == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

    return in_single_quote or in_double_quote


def get_comments(code) -> List[str]:
    """Get comments from the code."""
    in_comment_block = False
    f = code.split("\n")
    comments = []
    for line in f:
        comment = ""
        comment_detected = False
        comment_start = -1
        if "/*" in line:
            pos = line.find("/*")
            comment_start = pos
            comment_end = len(line)
            # line = line[pos:]
            in_comment_block = True
            comment_detected = True

        if "*/" in line:
            comment_end = line.rfind("*/") + 2
            # The comment block has started in previous lines
            if comment_start == -1:
                comment_start = 0
                try:
                    comments[-1] += "\n" + line[:comment_end]
                except IndexError:  # broken comment block
                    comments.append(line[:comment_end])
                    # print(line)
            else:
                comments.append(line[comment_start:comment_end])
            # content = content + line[comment_end:]
            # line = content
            in_comment_block = False
            comment_detected = False
        elif in_comment_block and comment_detected:
            comments.append(line[comment_start:comment_end])
            in_comment_block = True
            comment_detected = False
            continue

        if not comment_detected and "//" in line:
            pos = line.find("//")
            while pos != -1:
                if not is_inside_string_literal(line, pos - 1):
                    break
                pos = line.find("//", pos + 1)

            if pos != -1:
                comment_start = pos
                comment_end = len(line)
                comment_detected = True

        if comment_detected:
            comment = line[comment_start:comment_end]
            if not in_comment_block:
                comments.append(comment)
            else:
                comments[-1] += "\n" + comment
        elif in_comment_block:
            comments[-1] += "\n" + line

    return comments


def get_comments_with_replacement_lines(code):
    """Get comments and their replacement lines.
    Used for dataset generation in RQ2

    Returns:
        List of tuples, where each tuple contains:
        - Line range of the comment in the code
        - The original commented lines
    """
    result = []
    f = code.split("\n")
    start_line = -1
    end_line = -1
    block_type = None

    for line_no, line in enumerate(f):
        comment_detected = False
        if "/*" in line:
            pos = line.find("/*")
            while is_inside_string_literal(line, pos):
                pos = line.find("/*", pos + 2)
            if pos != -1:
                start_line = line_no
                end_line = line_no
                # line = line[pos:]
                block_type = "multi-line"
                comment_detected = True

        if "*/" in line:
            pos = line.rfind("*/")
            while is_inside_string_literal(line, pos):
                pos = line.rfind("*/", 0, pos - 1)
            if pos != -1:
                if start_line == -1:
                    start_line = line_no
                end_line = line_no
                line_range = (start_line, end_line)

                original_lines = f[start_line : end_line + 1]
                result.append((line_range, original_lines))
                block_type = None
                comment_detected = False
                start_line = end_line = -1
                continue

        if block_type != "multi-line" and "//" in line:
            pos = line.find("//")
            while pos != -1:
                if not is_inside_string_literal(line, pos - 1):
                    break
                pos = line.find("//", pos + 1)

            if pos != -1:
                if start_line == -1:
                    start_line = line_no
                end_line = line_no

                if line[:pos].strip() == "":
                    block_type = "single-line"
                    continue
                else:
                    block_type = None

        if block_type == "single-line" and not comment_detected:
            block_type = None

        if block_type == None and start_line != -1:
            line_range = (start_line, end_line)
            original_lines = f[start_line : end_line + 1]
            result.append((line_range, original_lines))
            start_line = end_line = -1

    return result


def get_inline_comments_with_context_lines(code, context_lines=3):
    """Get inline comments with their context lines."""
    lines = code.split("\n")
    comments = []
    in_comment_block = False
    cur_comment = ""
    start = 0
    end = 0

    for i, line in enumerate(lines):
        if "//" in line:
            pos = line.find("//")
            if not is_inside_string_literal(line, pos - 1):
                if in_comment_block:
                    end = min(len(lines), i + context_lines + 1)
                    cur_comment += "\n" + line[pos:]
                else:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    cur_comment = line[pos:]
                    if line[:pos].strip() == "":
                        in_comment_block = True
        elif cur_comment:
            in_comment_block = False
            context = "\n".join(lines[start:end])
            comments.append((context, cur_comment))
            cur_comment = ""
            start = 0
            end = 0
    return comments


def get_comment_types(code):
    """Get the types of comments in the code."""
    comments = get_comments_with_replacement_lines(code)

    comment_types = {
        "inline": False,
        "multiline": False,
        "javadoc": False,
    }


    for _, original_lines in comments:
        if 0 < len(original_lines) < 2:
            comment_types["inline"] = True
        else:
            comment_type = (
                "javadoc"
                if original_lines[0].lstrip().startswith("/*")
                else "multiline"
            )
            comment_types[comment_type] = True

    return comment_types