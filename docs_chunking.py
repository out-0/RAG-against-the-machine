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

        # Just for incremental ids
        id_generator = itertools.count

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
                        id=next(id_generator),
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

        def split_class_node(body: ast.AST, source_file: str) -> list:
            all_chunks: list = []
            ont_chunk: str = ""
            for method_node in node.body:
                ast.get_source_segment(source=file_content, node=node)


        pass

    @staticmethod
    def build_lines_count(source: str) -> list[int]:
        """For calculating the start index and end index of a chunk
        withing a file we are using some properties that AST already
        using internally (node.lineno | node.col_offset | node.end_lineno
        end_col_offset),
        those are indicate the line number and the column
        position where the chunk starting and ending, so to calculate
        the index of the chunk we need to know how much character
        in the previous lines and then add the col_offset to it.
        """

        # Each index represent a line and the value is the character
        # count is the character count its start at.
        lines_offsets: list = [0] # Line 1 start at character 0

        for line in source.splitlines(keepends=True):
            lines_offsets.append(lines_offsets[-1] + len(line))

        return lines_offsets

    @staticmethod
    def get_node_char_range(
            node:ast.AST,
            lines_offsets: list[int]
    ) -> tuple[int, int]:
        """Based on the lines_offsets we already build now we just map
        the properties of node to extract the characters count of
        previous lines and add the col offset to accumulate the left
        characters and got the result of character count when a node
        start and end.

        Arguments:
            node: node from AST representing a python block of code.
            lines_offsets: list maping lines (indexes) -> char offset


        Return:
            start: char offset where a node is starting withing source file.
            end: char offset where a node is ending withing source file.

        ex.
            if lines_offsets[2] == 12, that mean line 3 is starting at
            character count 12, so to calculate the exact offset where
            a node start, we add the column offset which the node already
            holding to the 12.
        """

        start: int = lines_offsets[node.lineno - 1] + node.col_offset
        end: int = lines_offsets[node.end_lineno - 1] + node.end_col_offset

        return start, end











