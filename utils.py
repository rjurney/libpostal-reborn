import time
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd
from langchain.chains import LLMChain
from langchain.globals import set_llm_cache
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.caches import InMemoryCache
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai.chat_models import ChatOpenAI
from openai import APIConnectionError, RateLimitError
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.notebook import tqdm


def gold_label_report(
    s: pd.Series, eval_methods: List[Callable], threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """gold_label_evaluate Evaluate a model on our original gold labels and return a report.
    Returns a Tuple of two pd.DataFrames, one with raw results and one with aggregated results."""
    s = s.copy(deep=True)

    raw_df: pd.DataFrame = pd.DataFrame()
    raw_df["Description"] = s["Description"]
    raw_df["Address1"] = s["Address1"]
    raw_df["Address2"] = s["Address2"]
    raw_df["Label"] = s["Label"]

    agg_funcs = {}
    kwargs = {"threshold": threshold}

    for eval_method in eval_methods:
        # Apply the matching model to the address pair
        func_col_name: str = eval_method.__name__

        if "sbert" in func_col_name:

            def apply_eval_method(row: pd.Series, threshold=threshold) -> Any:
                return eval_method(row, threshold=threshold)

            raw_df[func_col_name] = s.apply(apply_eval_method, axis=1, **kwargs)  # type: ignore
        else:
            raw_df[func_col_name] = s.apply(eval_method, axis=1)

        raw_df[f"{func_col_name}_correct"] = raw_df[func_col_name] == raw_df["Label"]

        agg_funcs[f"{eval_method.__name__}_acc"] = (
            f"{eval_method.__name__}",
            lambda x: x.mean(),
        )

    grouped_df: pd.DataFrame = raw_df.groupby("Description").agg(**agg_funcs)  # type: ignore

    return raw_df, grouped_df


def get_augment_prompt() -> ChatPromptTemplate:

    messages: List[Union[SystemMessagePromptTemplate, HumanMessagePromptTemplate]] = [
        SystemMessagePromptTemplate.from_template(
            "I need your help with a data science, data augmentation task. I am fine-tuning "
            "a sentence transformer paraphrase model to match pairs of addresses. I tried "
            "several embedding models and none of them perform well. They need fine-tuning "
            "for this task. I have created about 100 example pairs of addresses to serve as training "
            "data for fine-tuning a SentenceTransformer model. Each record has the fields "
            "Address1, Address2, a Description of the semantic they express "
            "(ex. 'different street number') and a Label (1 for positive match, 0 for negative)."
            "\n\n"
            "The tasks cover two categories of corner cases or difficult tasks. The first is when similar "
            "addresses in string distance aren't the same, thus have label 0. "
            "The second is the opposite: when dissimilar addresses in string distance are the same, "
            "thus have label 1. The strings you return for Address1 and Address2 should not be literally "
            "the same.\n\n"
            "Your task is to read a pair of Addresses, their Description and their Label and generate {Clones} "
            "different examples that express a similar semantic. Your job is to create variations "
            "of these records that satisfy the semantic expressed in the description but cover "
            "widely varying cases of the meaning covering the entire world. Do not literally copy the "
            "address components. Think methodically. Use what you know about postal addresses to accomplish "
            "this work."
            "\n\n"
            "You should return the result in a valid JSON array of records and nothing else, using the "
            "fields Address1, Address2, Description and Label."
        ),
        # MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            "Please generate {Clones} different examples that express the same or similar semantic as "
            "the pair of addresses below based on its Descripton, Label and the Address pairs.\n\n"
            "Address 1: {Address1}\n"
            + "Address 2: {Address2}\n"
            + "Description: {Description}\n"
            + "Label: {Label}\n"
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    return prompt


def augment_gold_labels(gold_df: pd.DataFrame, runs_per_example: int = 1) -> pd.DataFrame:
    """augment_gold_labels - OpenAI based data augmentation to create variants of gold labeled data."""

    prompt: ChatPromptTemplate = get_augment_prompt()

    set_llm_cache(InMemoryCache())
    llm: ChatOpenAI = ChatOpenAI(model="gpt-4o", temperature=0.5)

    json_output_parser: JsonOutputParser = JsonOutputParser()

    label_chain: LLMChain = LLMChain(
        name="label_chain", prompt=prompt, llm=llm, output_parser=json_output_parser, verbose=True
    )

    augment_results: List[Dict] = []

    total_iterations = len(gold_df) * runs_per_example
    print(
        f"Starting {total_iterations:,} API calls, {runs_per_example}x for each of {len(gold_df):,} hand-labeled records."
    )

    # Setup tqdm to provide progress per API call in our inner loop
    progress_bar = tqdm(total=total_iterations, desc="OpenAI API Calls")

    exception_count: int = 0
    for index, row in gold_df.iterrows():

        for run in range(row["Runs"]):

            # Call the LLMChain with the arguments, repeat on error
            try:
                address_pairs_ary = label_chain.run(**row.to_dict())
                time.sleep(10)
            except APIConnectionError:
                exception_count += 1
            except RateLimitError:
                exception_count += 1
                time.sleep(60)

            # Store the response in the results list
            augment_results += address_pairs_ary

            # Update bar for each API call
            progress_bar.update(1)

    # Close the progress bar when done
    progress_bar.close()

    print(f"Total `APIConnectionErrors`: {exception_count:,}")

    augmented_df: pd.DataFrame = pd.DataFrame(augment_results)
    print(f"Total raw augmentation results: {len(augmented_df):,}")
    augmented_df = augmented_df.dropna()
    print(f"Total clean augmentation results: {len(augmented_df):,}")

    print(f"\nWent from {len(gold_df):,} to {len(augmented_df):,} training examples.\n")

    return augmented_df


def compute_metrics(eval_pred: Tuple[List, List]) -> Dict[str, Number]:
    """compute_metrics - Compute accuracy, precision, recall, f1 and roc_auc"""
    predictions, labels = eval_pred
    metrics = {}
    for metric in accuracy_score, precision_score, recall_score, f1_score, roc_auc_score:
        metrics[metric.__name__] = metric(labels, predictions)

    # plot = wandb.plot.roc_curve(labels, predictions, labels=None, classes_to_plot=None)
    # wandb.log({"roc_curve": plot})

    # wandb.log(metrics)
    return metrics


def to_dict(parsed_address: List[Tuple[str, str]]) -> Dict[str, Union[str, List[str]]]:
    """Convert a parsed address into a dict - where multiple keys result in lists of values."""
    d: Dict[str, Union[str, List[str]]] = {}
    for value, key in parsed_address:
        if key in d:
            if isinstance(d[key], list):
                d[key].append(value)  # type: ignore
            else:
                d[key] = [d[key], value]  # type: ignore
        else:
            d[key] = value
    return d
