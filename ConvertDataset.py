def convert_df_to_conv_ai_dict(df: pd.DataFrame,
                               personality: List[str],
                               response_columns: List[str],
                               tokenizer: Callable[[str], List[str]],
                               max_tokens: Optional[int] = None,
                               n_candidates: int = 6
                               ) -> Dict[str, List[Any]]:
    # Add one because the index of the dataframe is the 0th position.
    tuple_map = {name: index + 1 for index, name in enumerate(df.columns.tolist())}
    train = []
    val = []
    # Step through every row in the dictionary
    for row in df.itertuples():
        question_text = row[tuple_map["body_1"]]
        for response_column in response_columns:
            candidates = sample_candidates(df, row[tuple_map["id"]], "id", "body", n_candidates)
            # questions = sample_candidates(df, row[tuple_map["id"]], "id", "body_1", n_candidates)
            if max_tokens is not None:
                questions = tokenizer.convert_tokens_to_string(tokenizer.tokenize(question_text)[:max_tokens])
                candidates = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(candidate)[:max_tokens]) for candidate in candidates]
                d = {"personality": personality,
                     "utterances": [{"history": questions,
                                     "candidates": candidates}]}
                if getattr(row, "split") == "train":
                    train.append(d)
                elif getattr(row, "split") == "val":
                    val.append(d)

    data = {"train": train, "valid": val}

    return data

def sample_candidates(df: pd.DataFrame, current_id: Any, id_column: Any, text_column: str, n: int) -> List[str]:
    """Samples candidate responses to a question from the dataframe

    It is aware of data splits and only samples from within the same split.  This avoids
    leaking information between training validation and testing.  The sampled responses are
    also drawn from all rows which do not have the same id as the current_id

    Args:
        df: The dataframe we want to sample responses from
        current_id: The unique identifier we would like to leave out of our sampling
        id_column: The column name in the dataframe with the unique ids.  current_id should
            be an element of this column
        text_column: The column with the text we want to sample
        n: How many samples we want to take.

    Returns:
        A list of samples strings from our dataframe.
    """
    # We must only sample candidates from the correct data split to avoid information leakage across channels
    split = df[df[id_column] == current_id]["split"].tolist()[0]
    candidate_df = df[df["split"] == split]

    # Sample 3 random rows from the dataframe not matching the current id
    sampled_texts = candidate_df[candidate_df[id_column] != current_id].sample(n + 15)[text_column].tolist()

    # join them all
    text = " ".join(sampled_texts)

    # Replace all newlines with spaces...
    text_no_newline = re.sub("\n", " ", text).lower()

    # Split on punctuation
    split_text = re.split('[?.!]', text_no_newline)

    # Remove all empty lines
    filtered_text = [x.strip() for x in split_text if len(x.strip()) > 1]

    # Shuffle the list
    return np.random.choice(filtered_text, n).tolist()
