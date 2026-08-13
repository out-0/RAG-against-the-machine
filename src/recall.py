def calculate_iou(chunk_a: tuple, chunk_b: tuple) -> float:
    """Calculates Intersection over Union for two character spans."""

    start_max = max(chunk_a[0], chunk_b[0])  # Find where first chunk end
    end_min = min(chunk_a[1], chunk_b[1])  # Find where second chunk start
    # First find the intersection (shared size between two area's)
    intersection = max(0, end_min - start_max)
    # Second find the Union (the full size of the two area's)
    union = (
        (chunk_a[1] - chunk_a[0]) + (chunk_b[1] - chunk_b[0]) - intersection
    )

    # Protect again division by zero
    if union == 0:
        return 0.0
    return intersection / union


def rag_recall_at_k(
    retrieved_results: list,
    ground_truths: list,
    k: int,
    iou_threshold: float = 0.05,
) -> float:
    """
    retrieved_chunks: List of Chunk objects
    ground_truths: List of tuples e.g., [("attention.py", 100, 200), ...]
    """

    # A quick check
    # ret_len: int = len(retrieved_results)
    # if k > ret_len:
    #     print(f"Warning: Found only {ret_len} retrieved result")
    #     print(f"Fallback... {ret_len} result")

    top_k_chunks: list = retrieved_results[:k]
    found_count: int = 0

    for gt in ground_truths:
        gt_file_path = gt["file_path"]
        gt_first_char_idx = gt["first_character_index"]
        gt_last_char_idx = gt["last_character_index"]

        for chunk in top_k_chunks:
            # Rule 1: File path MUST be exact
            if chunk["file_path"] == gt_file_path:
                # Rule 2: Check IoU overlap (intersection over union)
                iou_value: float = calculate_iou(
                    chunk_a=(gt_first_char_idx, gt_last_char_idx),
                    chunk_b=(
                        chunk["first_character_index"],
                        chunk["last_character_index"],
                    ),
                )

                if iou_value >= iou_threshold:
                    found_count += 1
                    break  # Found this ground truth, move to the next one

    # Formula: Found / Total Ground Truths
    if len(ground_truths) == 0:
        return 0.0
    return found_count / len(ground_truths)
