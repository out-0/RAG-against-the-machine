# TODO: CREATE ONE SPLITER BY THE PYTHON CODE TEXT SPLITER FROM LANGCHAIN
# TODO: CREATE ANOTHER ONE USING AST (ABSTRACT SYNTAX TREE)
# TODO: SET A LIMIT TO SOMETHING LIKE 100 SO IF ITS <= JUST CHUNK 100 DIRECTLY
# TODO: SET A GLOBAL GARDE WHICH IF THE SIZE <= 100 WE JSUT LOOP ON THE FILE AND CHUNK A NONESENCE BLOCK OF 100 CHAR
import ast
from dataclasses import dataclass
import itertools
from documents_loading import Document
import re


@dataclass
class Chunk:
    id: int
    content: str
    start_index: int | None = None
    end_index: int | None = None


class Chunker:
    """Chunker class that manual handle spliting python | markdown files withing
    required max size for chunk.

    For python files its based on AST which a builtin feature instead of external
    library,
    Also spliting lines based for markdown files since one requirement is tracking
    start | end indexes of each chunk which is missing if we relay on external
    chunking liraries.
    """

    def __init__(self, files: list[Document], max_size: int) -> None:
        """Accepting the list of documents targeted and max size
        for splited chunks.
        """
        self.files: list[Document] = files
        # Just for incremental ids
        self.id_generator: itertools.count[int] = itertools.count(start=1)
        self.max_size: int = max_size

    def chunk_python_file(self, source_file: Document) -> list[Chunk]:
        """Spliting a python file based on nodes constracted from
        Abstrac syntax tree, with a basic strategy:

        If node is a class, we check if its oversized, if yes we split
        it based on its internal methods, if method itself is oversized
        we fall back to spliting the method into lines and start shrink
        the lines while respecting max size,
        Anything not class of function is collected isolated as one chunk
        (ex. imports, globals...etc)
        """
        tree: ast.Module = ast.parse(source=source_file.content)

        # This for major functionality of python like (class's, functions)
        splited_chunks: list[Chunk] = []

        # TODO: PUT IMPORTS WITH WITH EVERY CHUNK FOR MORE CONTEXT,
        #
        # edit: Since doing that while handling and respecting the max size
        # will introduce more complexity and miss, we'll skip it regarding
        # that the questions is not relay on the dependencis or imports.
        #
        # imports_tmp: list[str] = [ast.unparse(n) for n in tree.body
        #                      if isinstance(n, (ast.Import, ast.ImportFrom))]
        # imports: str = "\n".join(imports_tmp)

        # Track the start and end indexes of the code
        # that not (class|funs)
        outscoop_start_idx: int | None = None
        outscoop_end_idx: int | None = None

        for node in tree.body:
            node_start_idx, node_end_idx = self._get_chunk_indexes(
                source=source_file.content,
                node=node,
            )

            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if there is already something
                # accumulated so we append it first
                if outscoop_start_idx is not None:
                    splited_chunks.append(
                        Chunk(
                            id=next(self.id_generator),
                            content=(
                                source_file.content[outscoop_start_idx:outscoop_end_idx]
                            ),
                            start_index=outscoop_start_idx,
                            end_index=outscoop_end_idx,
                        )
                    )
                    outscoop_start_idx = None

                # Extracting the real source code of a node
                code_text: str | None = ast.get_source_segment(
                    source=source_file.content, node=node
                )

                if code_text and len(code_text) <= self.max_size:
                    # Extracting some metadata
                    start_idx, end_idx = self._get_chunk_indexes(
                        source=source_file.content, node=node
                    )
                    chunk: Chunk = Chunk(
                        id=next(self.id_generator),
                        content=code_text,
                        start_index=start_idx,
                        end_index=end_idx,
                    )
                    splited_chunks.append(chunk)

                # Too big, split, handle seperatly
                else:
                    splited_chunks.extend(
                        self._split_oversized_node(
                            node=node,
                            source_file=source_file.content,
                            max_size=self.max_size,
                        )
                    )
            # Collect other compenents
            # TODO: Set a minimum limit for chunk size
            else:
                code_text = ast.get_source_segment(
                    source=source_file.content, node=node
                )
                # Just a defensive.
                if not code_text:
                    continue

                if outscoop_start_idx is None:
                    outscoop_start_idx = node_start_idx

                elif (node_end_idx - outscoop_start_idx) > self.max_size:
                    splited_chunks.append(
                        Chunk(
                            id=next(self.id_generator),
                            content=source_file.content[
                                outscoop_start_idx:outscoop_end_idx
                            ],
                            start_index=outscoop_start_idx,
                            end_index=outscoop_end_idx,
                        )
                    )
                    outscoop_start_idx = node_start_idx

            outscoop_end_idx = node_end_idx

        # Register whatever left here
        if outscoop_start_idx:
            splited_chunks.append(
                Chunk(
                    id=next(self.id_generator),
                    content=source_file.content[outscoop_start_idx:outscoop_end_idx],
                    start_index=outscoop_start_idx,
                    end_index=outscoop_end_idx,
                )
            )
        return splited_chunks

    def chunk_markdown_file(self, source_file: Document) -> list[Chunk]:
        """Primary spliting markdown files based on the headers in the markdown
        structure, if a file size is excedding the max size we fallback to
        spiting using Recursicve Characters to maintain and respecting
        max size.

        Arguments:
            - source file: the targeted file after constrcted as Document.

        Return:
            - list of chunks that got splited and maintained max size.
        """

        sections: list[tuple[int, int]] = []
        splited_chunks: list[Chunk] = []

        # Extract the start indexes of headers by maching with regix
        header_positions: list[int] = [
            header.start()
            for header in re.finditer(r"^#{1,6} ", source_file.content, re.MULTILINE)
        ]
        # Add extra entry to benefit later below
        header_positions.append(len(source_file.content))

        # Collect the start / end indexes of each session (header)
        for i in range(len(header_positions) - 1):
            start_idx = header_positions[i]
            end_idx = header_positions[i + 1]

            # Handle if a chunk is oversized
            if (end_idx - start_idx) > self.max_size:
                new_slices: list[tuple[int, int]] = self._split_markdown_section(
                    source_text=source_file.content,
                    chunk_start_idx=start_idx,
                    chunk_end_idx=end_idx,
                )
                sections.extend(new_slices)
                continue
            sections.append((start_idx, end_idx))

        # Iter over slices and build the actual chunks
        for slice in sections:
            start_idx, end_idx = slice
            splited_chunks.append(
                Chunk(
                    id=next(self.id_generator),
                    content=source_file.content[start_idx:end_idx],
                    start_index=start_idx,
                    end_index=end_idx,
                )
            )

        return splited_chunks

    def _split_markdown_section(
        self,
        source_text: str,
        chunk_start_idx: int,
        chunk_end_idx: int,
    ) -> list[tuple[int, int]]:
        """Split an oversized markdown section into smaller slices,
        tracked purely as (start, end) character offsets.

        Arguments:
            - source_text: source file that we working on
            - chunk_start_idx: index of start of that oversized chunk
            - chunk_end_idx: index of end of that oversized chunk

        Return:
            - list of tuples that indicate the slices of that oversized chunk
        """
        section_text: str = source_text[chunk_start_idx:chunk_end_idx]
        lines: list[str] = section_text.splitlines(keepends=True)

        slices: list[tuple[int, int]] = []
        cursor: int = chunk_start_idx
        current_len: int = 0

        for line in lines:
            if current_len + len(line) > self.max_size and current_len > 0:
                slices.append((cursor, cursor + current_len))
                cursor += current_len
                current_len = 0
            current_len += len(line)

        if current_len > 0:
            slices.append((cursor, cursor + current_len))

        return slices

    def process_files(self) -> list[Chunk]:
        """"""
        splited_chunks: list[Chunk] = []

        for file in self.files:
            match file.extension:
                case ".py":
                    splited_chunks.extend(self.chunk_python_file(file))
                case ".md":
                    splited_chunks.extend(self.chunk_markdown_file(file))
                case _:
                    # This default should not reached for the current situation
                    pass

        return splited_chunks

    def _split_oversized_node(
        self,
        node: ast.AST,
        source_file: str,
        max_size: int,
    ) -> list[Chunk]:
        """"""
        # If node is A class, we spit its methods and shrink some of them later
        # for optimization
        # If node i itself a method, we split its lines and shrunk
        #

        splited_chunks: list[Chunk] = []

        if isinstance(node, ast.ClassDef):
            splited_chunks.extend(
                self._split_class_node(
                    node=node,
                    source_file=source_file,
                    max_size=max_size,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            splited_chunks.extend(
                self._split_method_node(
                    source=source_file, node=node, max_size=max_size
                )
            )

        return splited_chunks

    def _split_class_node(
        self, node: ast.ClassDef, source_file: str, max_size: int
    ) -> list[Chunk]:
        """"""

        all_chunks: list[Chunk] = []
        one_chunk: str = ""

        # Not set yet, its will updated at the first
        chunk_start_idx: int = -1

        for method_node in node.body:
            text_code: str | None = ast.get_source_segment(
                source=source_file, node=node
            )

            if text_code is None:
                raise ValueError("Error while extracting source segment")

            node_start_idx: int
            node_start_idx, _ = self._get_chunk_indexes(
                source=source_file, node=method_node
            )

            # Check if the chunk not excedding the max size
            if len(text_code) < max_size:
                # Check if concatinating them will result to oversized
                # This case is not reached at the very first iter
                if len(one_chunk + text_code) > max_size:
                    all_chunks.append(
                        Chunk(
                            id=next(self.id_generator),
                            content=one_chunk,
                            start_index=chunk_start_idx,
                            end_index=chunk_start_idx + len(one_chunk),
                        )
                    )
                    one_chunk = text_code
                    chunk_start_idx = node_start_idx

                else:
                    if one_chunk == "":
                        one_chunk = text_code
                        chunk_start_idx = node_start_idx

                    else:
                        one_chunk = "\n".join([one_chunk, text_code])

            # If method is too long we split it
            else:
                method_chunks: list[Chunk] = self._split_method_node(
                    source=source_file, node=method_node, max_size=max_size
                )
                all_chunks.extend(method_chunks)

        if one_chunk:
            all_chunks.append(
                Chunk(
                    id=next(self.id_generator),
                    content=one_chunk,
                    start_index=chunk_start_idx,
                    end_index=chunk_start_idx + len(one_chunk),
                )
            )

        return all_chunks

    # @staticmethod
    def _get_chunk_indexes(self, source: str, node: ast.stmt) -> tuple[int, int]:
        """For calculating the start index and end index of a chunk
        withing a file we are using some properties that AST already
        using internally (node.lineno | node.col_offset | node.end_lineno
        end_col_offset),
        those are indicate the line number and the column
        position where the chunk starting and ending, so to calculate
        the index of the chunk we need to know how much character
        in the previous lines and then add the col_offset to it.

        Arguments:
            node: node from AST representing a python block of code.

        Return:
            start: char offset where a node is starting withing source file.
            end: char offset where a node is ending withing source file.

        ex.
            if lines_offsets[2] == 12, that mean line 3 is starting at
            character count 12, so to calculate the exact offset where
            a node start, we add the column offset which the node already
            holding to the 12.
        """

        # Each index represent a line and the value is the character
        # count is the character count its start at.
        lines_offsets: list[int] = [0]  # Line 1 start at character 0

        for line in source.splitlines(keepends=True):
            lines_offsets.append(lines_offsets[-1] + len(line))

        if node.end_lineno is None or node.end_col_offset is None:
            raise AttributeError("Error will access node attributes")

        else:
            start: int = lines_offsets[node.lineno - 1] + node.col_offset
            end: int = lines_offsets[node.end_lineno - 1] + node.end_col_offset

            return start, end

    def _split_method_node(
        self, source: str, node: ast.stmt, max_size: int
    ) -> list[Chunk]:
        """Split a method into lines and shrunk them after that while
        keep respecting the max size.

        If the max size was even less than line length or if the code line
        is having a multi quotes, those two cases will add a more complexity
        and honesly they are just a shit cases, so i decide to move on and
        marked it as a effect resulted from a small 'max size',

        cause, honesly why you will choose a very small max size?


        Arguments:
            - source : source file
            - node: method node

        Return:
            - list of small chunks that respect max size
        """
        all_chunks: list[Chunk] = []
        one_chunk: str = ""

        # Since we going to split the string so we track metadata manually
        global_start_idx: int

        global_start_idx, _ = self._get_chunk_indexes(source, node)

        text_code: str | None = ast.get_source_segment(source=source, node=node)
        # Just defensive
        if text_code:
            splited_lines: list[str] = text_code.splitlines(
                keepends=True
            )  # TODO: CHECK CORRECTNESS LATER

        else:
            raise AttributeError("Error while getting source segment")

        cursor: int = global_start_idx

        for line in splited_lines:
            if len(one_chunk + line) <= max_size:
                one_chunk = "".join([one_chunk, line])

            # If its wll exceed then we just take current chunk
            else:
                all_chunks.append(
                    Chunk(
                        id=next(self.id_generator),
                        content=one_chunk,
                        start_index=cursor,
                        end_index=cursor + len(one_chunk),
                    )
                )
                # Update the indexes for the next chunks.
                cursor += len(one_chunk) + 1
                one_chunk = line

        if one_chunk:
            all_chunks.append(
                Chunk(
                    id=next(self.id_generator),
                    content=one_chunk,
                    start_index=cursor,
                    end_index=cursor + len(one_chunk),
                )
            )

        return all_chunks
