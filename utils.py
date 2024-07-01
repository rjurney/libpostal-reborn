import logging
import os
import sys
import time
from numbers import Number
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import pandas as pd
import torch
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
from postal.parser import parse_address  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer  # type: ignore

COLUMN_SPECIAL_CHAR = "[COL]"
VALUE_SPECIAL_CHAR = "[VAL]"

# Setup basic logging
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logger = logging.getLogger(__name__)


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


def compute_sbert_metrics(eval_pred: Tuple[List, List]) -> Dict[str, Number]:
    """compute_metrics - Compute accuracy, precision, recall, f1 and roc_auc"""
    predictions, labels = eval_pred
    metrics = {}
    for metric in accuracy_score, precision_score, recall_score, f1_score, roc_auc_score:
        metrics[metric.__name__] = metric(labels, predictions)

    return metrics


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


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


def parse_match_address(address1: str, address2: str) -> Literal[0, 1]:  # noqa: C901
    """parse_match_address implements address matching using the precise, parsed structure of addresses."""
    parsed_address1: Dict[str, Union[str, List[str]]] = to_dict(parse_address(address1))
    parsed_address2: Dict[str, Union[str, List[str]]] = to_dict(parse_address(address2))

    def match_road(address1: Dict, address2: Dict) -> Literal[0, 1]:
        """match_road - literal road matching, negative if either lacks a road"""
        if ("road" in address1) and ("road" in address2):
            if address1["road"] == address2["road"]:
                logger.debug("road match")
                return 1
            else:
                logger.debug("road mismatch")
                return 0
        logger.debug("road mismatch")
        return 0

    def match_house_number(address1: Dict, address2: Dict) -> Literal[0, 1]:
        """match_house_number - literal house number matching, negative if either lacks a house_number"""
        if ("house_number" in address1) and ("house_number" in address2):
            if address1["house_number"] == address2["house_number"]:
                logger.debug("house_number match")
                return 1
            else:
                logger.debug("house_number mismatch")
                return 0
        logger.debug("house_number mistmatch")
        return 0

    def match_unit(address1: Dict, address2: Dict) -> Literal[0, 1]:
        """match_unit - note a missing unit in both is a match"""
        if "unit" in address1:
            if "unit" in address2:
                logger.debug("unit match")
                return 1 if (address1["unit"] == address2["unit"]) else 0
            else:
                logger.debug("unit mismatch")
                return 0
        if "unit" in address2:
            if "unit" in address1:
                logger.debug("unit match")
                return 1 if (address1["unit"] == address2["unit"]) else 0
            else:
                logger.debug("unit mismatch")
                return 0
        # Neither address has a unit, which is a default match
        return 1

    def match_postcode(address1: Dict, address2: Dict) -> Literal[0, 1]:
        """match_postcode - literal matching, negative if either lacks a postal code"""
        if ("postcode" in address1) and ("postcode" in address2):
            if address1["postcode"] == address2["postcode"]:
                logger.debug("postcode match")
                return 1
            else:
                logger.debug("postcode mismatch")
                return 0
        logger.debug("postcode mismatch")
        return 0

    def match_country(address1: Dict, address2: Dict) -> Literal[0, 1]:
        """match_country - literal country matching - pass if both don't have one"""
        if ("country" in address1) and ("country" in address2):
            if address1["country"] == address2["country"]:
                logger.debug("country match")
                return 1
            else:
                logger.debug("country mismatch")
                return 0
        # One or none countries should match
        logger.debug("country match")
        return 1

    # Combine the above to get a complete address matcher
    if (
        match_road(parsed_address1, parsed_address2)
        and match_house_number(parsed_address1, parsed_address2)
        and match_unit(parsed_address1, parsed_address2)
        and match_postcode(parsed_address1, parsed_address2)
        and match_country(parsed_address1, parsed_address2)
    ):
        logger.debug("overall match")
        return 1
    else:
        logger.debug("overall mismatch")
        return 0


def compute_classifier_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    predictions = (logits > 0.5).long().squeeze()

    if len(predictions) != len(labels):
        raise ValueError(
            f"Mismatch in lengths: predictions ({len(predictions)}) and labels ({len(labels)})"
        )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def structured_encode_address(address: str) -> str:
    """structured_parse_address - encode a parsed address"""
    parsed_address: List[Tuple[str, str]] = parse_address(address)
    sorted_address: List[Tuple[str, str]] = list(
        sorted(parsed_address, key=lambda x: x[1])
    )  # no secondary sort to maek it determinstic?
    encoded_address: str = str()
    for val, col in sorted_address:
        encoded_address += COLUMN_SPECIAL_CHAR + col + VALUE_SPECIAL_CHAR + val
    return encoded_address


def tokenize_function(examples, tokenizer):
    encoded_a = tokenizer(examples["sentence1"], padding="max_length", truncation=True)
    encoded_b = tokenizer(examples["sentence2"], padding="max_length", truncation=True)
    return {
        "input_ids_a": encoded_a["input_ids"],
        "attention_mask_a": encoded_a["attention_mask"],
        "input_ids_b": encoded_b["input_ids"],
        "attention_mask_b": encoded_b["attention_mask"],
        "labels": examples["label"],
    }


def format_dataset(dataset):
    dataset.set_format(
        type="torch",
        columns=["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "labels"],
    )
    return dataset


def save_custom_model(model, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Save the base BERT model and tokenizer
    model.model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)

    # Save the entire state dict of your custom model
    torch.save(model.state_dict(), os.path.join(save_path, "full_model_state_dict.pt"))


def load_custom_model(model_cls, load_path, device="cpu"):
    # Initialize your custom model
    custom_model = model_cls(model_name=load_path)

    # Load the base BERT model and tokenizer
    custom_model.model = AutoModel.from_pretrained(load_path)
    custom_model.tokenizer = AutoTokenizer.from_pretrained(load_path)

    # Load the full state dict
    state_dict = torch.load(
        os.path.join(load_path, "full_model_state_dict.pt"), map_location=device
    )
    custom_model.load_state_dict(state_dict)

    return custom_model.to(device)
