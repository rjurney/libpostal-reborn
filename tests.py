import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from fuzzywuzzy import fuzz  # type: ignore
from postal.expand import expand_address  # type: ignore
from postal.parser import parse_address  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm


@pytest.fixture
def matched_address_pairs() -> List[Tuple[str, str, str]]:
    """matched_address_pairs is a pytest fixture that returns a list of matched address pairs.

    Returns
    -------
    List[Tuple[str, str]]
        A list of pairs of matching address strings
    """
    address_pairs: List[Tuple[str, str, str]] = [
        (
            "Standard Formatting Differences",
            "2024 NW 5th Ave, Miami, FL 33127",
            "2024 Northwest 5th Avenue, Miami, Florida 33127",
        ),
        (
            "Misspellings",
            "1600 Pennsylvna Ave NW, Washington, DC 20500",
            "1600 Pennsylvania Avenue NW, Washington, DC 20500",
        ),
        (
            "Abbreviations",
            "550 S Hill St, Los Angeles, CA",
            "550 South Hill Street, Los Angeles, California",
        ),
        (
            "Incomplete Addresses",
            "1020 SW 2nd Ave, Portland",
            "1020 SW 2nd Avenue, Portland, OR 97204",
        ),
        ("Different Ordering", "221B Baker Street, London, UK", "London, UK, 221B Baker Street"),
        ("Numerical Variations", "Third Ave, New York, NY", "3rd Avenue, New York, New York"),
        (
            "Variant Formats",
            "350 Fifth Avenue, New York, NY 10118",
            "Empire State Bldg, 350 5th Ave, NY, NY 10118",
        ),
        (
            "Variant Formats",
            "Çırağan Caddesi No: 32, 34349 Beşiktaş, Istanbul, Turkey",
            "Ciragan Palace Hotel, Ciragan Street 32, Besiktas, Istanbul, TR",
        ),
        (
            "Different Character Sets",
            "北京市朝阳区建国路88号",
            "Běijīng Shì Cháoyáng Qū Jiànguó Lù 88 Hào",
        ),
        ("Variant Formats", "上海市黄浦区南京东路318号", "上海黄浦南京东路318号"),
        (
            "Variant Formats",
            "Shànghǎi Shì Huángpǔ Qū Nánjīng Dōng Lù 318 Hào",
            "Shànghǎi Huángpǔ Nánjīng Dōng Lù 318 Hào",
        ),
        (
            "Formal and Localized Address Format",
            "B-14, Connaught Place, New Delhi, Delhi 110001, India",
            "B-14, CP, ND, DL 110001",
        ),
        (
            "Different Character Sets",
            "16, MG Road, Bangalore, Karnataka 560001, India",
            "16, एमजी रोड, बैंगलोर, कर्नाटक 560001",
        ),
    ]

    return address_pairs


@pytest.fixture
def mismatched_address_pairs() -> List[Tuple[str, str, str]]:
    """mismatched_address_pairs is a pytest fixture that returns a list of mismatched address pairs.

    Returns
    -------
    List[Tuple[str, str]]
        A list of pairs of mismatching address strings
    """
    address_pairs: List[Tuple[str, str, str]] = [
        (
            "Similar Cities",
            "100 Main Street, Springfield, IL 62701",
            "100 Main Street, Springfield, MA 01103",
        ),
        (
            "Similar Street Names",
            "200 1st Ave, Seattle, WA 98109",
            "200 1st Ave N, Seattle, WA 98109",
        ),
        (
            "Adjacent Building Numbers",
            "4800 Oak Street, Kansas City, MO 64112",
            "4800 W Oak Street, Kansas City, MO 64112",
        ),
        (
            "Similar International Locations",
            "33 Queen Street, Auckland 1010, New Zealand",
            "33 Queen Street, Brisbane QLD 4000, Australia",
        ),
        (
            "Close Numerical Variants",
            "75 West 50th Street, New York, NY 10112",
            "50 West 75th Street, New York, NY 10023",
        ),
        ("Similar Road Names", "北京市朝阳区朝阳门外大街6号", "北京市朝阳区朝阳门内大街6号"),
        (
            "Similar Road Names",
            "Běijīng Shì Cháoyáng Qū Cháoyángmén Wài Dàjiē 6 Hào",
            "Běijīng Shì Cháoyáng Qū Cháoyángmén Nèi Dàjiē 6 Hào",
        ),
        ("Similar Building Names", "上海市徐汇区中山西路200号", "上海市长宁区中山西路200号"),
        (
            "Similar Building Names",
            "Shànghǎi Shì Xúhuì Qū Zhōngshān Xī Lù 200 Hào",
            "Shànghǎi Shì Chángníng Qū Zhōngshān Xī Lù 200 Hào",
        ),
    ]

    return address_pairs


@pytest.fixture
def senzing_addresses() -> pd.DataFrame:
    """addresses pytest fixture that downloads the test addresses from Senzing into a pd.DataFrame

    Returns
    -------
    pandas.DataFrame
        12.9K Senzing test addresses
    """

    test_data_path = "data/test_data.csv"
    test_data_url = (
        "https://github.com/Senzing/libpostal-data/raw/main/files/tests/v1.1.0/test_data.csv"
    )

    if not os.path.exists(test_data_path):
        df = pd.read_csv(test_data_url)
    else:
        df = pd.read_csv(test_data_path)

    return df


def test_string_distance(matched_address_pairs: List[Tuple[str, str, str]]) -> None:
    """test_string_distance tests how similar addresses are using string distsance

    Parameters
    ----------
    address_pairs : List[Tuple[str, str]]
        Fixture that returns a list of address pairs
    """
    for description, address1, address2 in matched_address_pairs:
        distance = fuzz.ratio(address1, address2)
        distance < 0.5


def test_embedding_distance(matched_address_pairs: List[Tuple[str, str, str]]) -> None:
    """test_embedding_distance tests how similar addresses are using sentence embeddings

    Parameters
    ----------
    address_pairs : List[Tuple[str, str]]
        Fixture that returns a list of address pairs
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for description, address1, address2 in matched_address_pairs:

        # Convert addresses to embeddings
        embeddings = model.encode([address1, address2])

        # Calculate the cosine similarity between the two embeddings
        cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        # Print the cosine similarity for each pair
        print(f"Cosine similarity between '{address1}' and '{address2}': {cosine_similarity:.4f}")

        cosine_similarity > 0.5


def test_libpostal_parse_jaccard(matched_address_pairs: List[Tuple[str, str, str]]) -> None:
    """test_libpostal_parse tests how similar addresses are using libpostal and Jaccard Similarity

    Parameters
    ----------
    address_pairs : List[Tuple[str, str]]
        Fixture that returns a list of address pairs
    """

    for description, address1, address2 in matched_address_pairs:

        parsed_address1: Dict[str, str] = parse_address(address1)
        parsed_address2: Dict[str, str] = parse_address(address2)

        print(f"Parsed address 1: {parsed_address1}")
        print(f"Parsed address 2: {parsed_address2}")


def rule_match(address1: str, address2: str) -> float:
    """rule_match match based on sets of parsed address parts matching

    Parameters
    ----------
    address1 : str
        A string address
    address2 : str
        A string address

    Returns
    -------
    float
        1.0 for a match, 0.0 for no match
    """

    # Parse and convert to dictionary
    parsed_address1: Dict[str, str] = {x[1]: x[0] for x in parse_address(address1)}
    parsed_address2: Dict[str, str] = {x[1]: x[0] for x in parse_address(address2)}

    match = 0.0
    # If the house number and postal code match, its a match. In some domains, for some tasks...
    if "postcode" in parsed_address1 and "postcode" in parsed_address2:
        if parsed_address1["postcode"] == parsed_address2["postcode"]:
            if (
                parsed_address1["house_number"] in parsed_address1
                and "house_number" in parsed_address2
            ):
                if parsed_address1["house_number"] == parsed_address2["house_number"]:
                    match = 1.0

    return match


def test_libpostal_parse_rules(matched_address_pairs: List[Tuple[str, str, str]]) -> None:
    """test_libpostal_parse_rules tests how similar addresses are using libpostal and rules

    Parameters
    ----------
    address_pairs : List[Tuple[str, str]]
        Fixture that returns a list of address pairs
    """

    matches = []
    for description, address1, address2 in matched_address_pairs:
        rule_match_score = rule_match(address1, address2)
        matches.append(rule_match_score)

    print(f"Matches: {len([x for x in matches if x])}  matches out of {len(matches)} pairs")


def test_compare_distance_methods(matched_address_pairs: List[Tuple[str, str, str]]) -> None:
    """compare_distance_methods apply all the distance metrics and compare the results.

    Parameters
    ----------
    address_pairs : List[Tuple[str, str]]
        Fixture that returns a list of address pairs
    """
    rows = []
    for description, address1, address2 in tqdm(
        matched_address_pairs, total=len(matched_address_pairs), desc="Comparing addresses"
    ):

        # Compute a string distance that ranges from 0 to 1
        lev_distance = fuzz.ratio(address1, address2)

        # Convert addresses to embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([address1, address2])

        # Calculate the cosine similarity between the two embeddings
        cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        parsed_address1 = parse_address(address1)
        parsed_address2 = parse_address(address2)

        # Compute the Jaccard similarity between the two parsed addresses
        jaccard_similarity = len(set(parsed_address1) & set(parsed_address2)) / len(
            set(parsed_address1) | set(parsed_address2)
        )

        # Get a binary, rule match score
        rule_match_score = rule_match(address1, address2)

        # Build a row with all the values
        row = {
            "address1": address1,
            "address2": address2,
            "lev_distance": lev_distance,
            "cosine_similarity": cosine_similarity,
            "jaccard_similarity": jaccard_similarity,
            "rule_match_score": rule_match_score,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 9999)

    df.to_csv("data/address_similarity.csv", header=True, index=False)


def test_libpostal_expand(matched_address_pairs: List[Tuple[str, str, str]]) -> None:
    """test_libpostal_expand tests how similar addresses are using libpostal and rules

    Parameters
    ----------
    address_pairs : List[Tuple[str, str]]
        Fixture that returns a list of address pairs
    """
    for description, address1, address2 in matched_address_pairs:
        expanded_address1 = expand_address(address1)
        expanded_address2 = expand_address(address2)

        print(f"Expanded address 1: {expanded_address1}")
        print(f"Expanded address 2: {expanded_address2}")
