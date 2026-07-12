# TODO: CREATE ONE SPLITER BY THE PYTHON CODE TEXT SPLITER FROM LANGCHAIN
# TODO: CREATE ANOTHER ONE USING AST (ABSTRACT SYNTAX TREE)
import ast
from dataclasses import dataclass
import itertools
from documents_loading import Document


@dataclass
class Chunk:
    id: int
    content: str
    start_index: int
    end_index: int

class Chunker:
    """"""

    def __init__(self, files: list[Document]) -> None:
        """"""
        self.files: list[Document] = files

        # Just for incremental ids
        self.id_generator: itertools.count[int] = itertools.count()

    def chunk_python_file(self, file: Document) -> None:
        """"""
        print(file.content)
        tree: ast.Module = ast.parse(source=file.content)

        # This for major functionality of python like (class's, functions)
        chunks: list[Chunk] = []

        # This for the rest of things in a pyhthon code (imports, globals)
        accumulated: list = []

        # This is needed to calculate some metadata of a node
        lines_offsets: list[int] = Chunker.build_lines_count(file)


        # Iterate over the nodes from three (AST)
        for node in ast.iter_child_nodes(tree):

            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extracting the real source code of a node
                code_text: str | None = ast.get_source_segment(file.content, node)

                if code_text and len(code_text) <= args.max_chunk_size:
                    # Extracting some metadata
                    start_idx, end_idx = Chunker.get_node_char_range(
                        node=node, lines_offsets=lines_offsets
                    )
                    chunk: Chunk = Chunk(
                        id=next(self.id_generator),
                        content=code_text,
                        start_index=start_idx,
                        end_indext=end_idx
                    )
                    chunks.append(chunk)

                # Too big, split, handle seperatly
                else:
                    chunks.extend(
                        self.split_oversized_node(
                            node, file.content, args.max_chunk_size
                        )
                    )

            else:
                code_text: str | None = ast.get_source_segment(
                    source=file.content, node=node
                )
                # Just a defensive since logically we wont hit that case
                # since the files parsed is coming as external so info
                # is already set.
                if not code_text:
                    continue
                if len(code_text) + len(accumulated) >= args.max_chunk_size:
                    chunks.append({
                        
                    })

                print(code)
        exit()

    def chunk_markdown_file(self, file: Document) -> None:
        """"""
        pass

    def process_files(self) -> None:
        """"""

        for file in self.files:
            match file.extension:
                case ".py":
                    self.chunk_python_file(file)
                case ".md":
                    pass
                    self.chunk_markdown_file(file)
                case _:
                    # This default should not reached for the current situation
                    pass

    def set_metadata(self) -> None:
        """"""
        pass

    def split_oversized_node(
        self, 
        node: ast.AST,
        source_file: str,
        max_chunk_size:int
    ) -> None:
        """"""
        # If node is A class, we spit its methods and shrink some of them later
        # for optimization
        # If node i itself a method, we split its lines and shrunk

        splited_chunks = []
        pass

    def _split_class_node(self, body: ast.AST, source_file: str) -> list:
        """"""

        all_chunks: list = []
        one_chunk: str = ""
        for method_node in node.body:
            text_code: str | None = ast.get_source_segment(
                    source=file_content, node=node
                    )
            if len(text_code) > max_size:
                method_chunks: list[Chunk] = self._split_method_node()
        pass

    @staticmethod
    def _get_chunk_indexes(source: str, node: ast.stmt) -> tuple[int, int]:
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
        lines_offsets: list[int] = [0] # Line 1 start at character 0

        for line in source.splitlines(keepends=True):
            lines_offsets.append(lines_offsets[-1] + len(line))

        lineno: int | None = node.lineno
        end_lineno: int | None = node.end_lineno
        col_offset: int | None = node.col_offset
        end_col_offset: int | None = node.end_col_offset

        if lineno and end_lineno and col_offset and end_col_offset:
            start: int = lines_offsets[lineno - 1] + col_offset
            end: int = lines_offsets[end_lineno - 1] + end_col_offset

            return start, end
        # This is unreachable case
        else:
            raise AttributeError("Error will access node attributes")

    def _split_method_node(
            self,
            source: str,
            node: ast.AST,
            max_size: int
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

        global_start_idx, _ = Chunker._get_chunk_indexes(node)

        text_code: str | None = ast.get_source_segment(
                source=source, node=node
            )
        # Just defensive
        if text_code:
            splited_lines: list[str] = text_code.splitlines()

        else:
            raise AttributeError("Error while getting source segment")

        cursor: int = global_start_idx

        for line in splited_lines:
            if len(one_chunk + line) < max_size:
                one_chunk = (
                        line
                        if one_chunk == ""
                        else "\n".join([one_chunk, line])
                )

            # If its wll exceed then we just take current chunk
            else:
                all_chunks.append(
                        Chunk(
                            id=next(self.id_generator),
                            content=one_chunk,
                            start_index=cursor,
                            end_index=cursor + len(one_chunk)
                            )
                        )
                # Update the indexes for the next chunks.
                cursor += len(one_chunk) + 1
                one_chunk = ""

        if one_chunk:
            all_chunks.append(
                    Chunk(
                        id=next(self.id_generator),
                        content=one_chunk,
                        start_index=cursor,
                        end_index=cursor + len(one_chunk)
                        )
                    )

        return all_chunks
