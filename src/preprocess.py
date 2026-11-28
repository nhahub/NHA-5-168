"""
Load, filter, clean, de-duplicate and split the data (SRS FR-1 to FR-4).

Every setting comes from config.yaml (FR-7). The steps follow the EDA decisions
(notebooks/01_eda.ipynb, reports/EDA_Report.md):

    1  read config and raw CSV                      FR-1
    2  keep English rows                            FR-2
    3  keep the needed columns (no answer, no tags) leakage, EDA section 6
    4  clean subject and body                       DEC-7, EDA 8 (literal "\\n")
    5  build the joined text (subject + ". " + body) FR-2
    6  drop near-empty texts, log how many          DEC-11
    7  check labels against config                  EDA sections 4-5
    8  remove duplicates on normalized text         DEC-13a
    9  paraphrase families (TF-IDF + Union-Find)    DEC-13
   10  stratified GROUP split 70 / 15 / 15          DEC-10, DEC-13
   11  checks: shares per split + leakage check
   12  save splits + split_summary.json with checksums  FR-4

Run from the project root:
    python src/preprocess.py
    python src/preprocess.py --config config.yaml
"""

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("preprocess")

KEEP_COLUMNS = ["subject", "body", "type", "queue", "version"]


# ====================================================================== config
def find_project_root():
    """The project root is the folder that contains config.yaml (one level above src/)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path):
    """Read config.yaml and return it as a dict."""
    with open(config_path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


# ====================================================================== text helpers
def clean_text(text, replace_literal_newlines, normalize_whitespace):
    """
    Input : text                     - one string (NaN allowed).
            replace_literal_newlines - True: the 2-character text "\\n" becomes a space.
            normalize_whitespace     - True: many spaces / tabs / new lines become one space.
    Output: the cleaned string ("" for NaN).
    """
    if pd.isna(text):
        return ""
    text = str(text)
    if replace_literal_newlines:
        text = text.replace("\\n", " ")
    if normalize_whitespace:
        words = text.split()
        text = " ".join(words)
    return text


def build_joined_text(subject, body, separator):
    """FR-2: subject + separator + body; an empty subject gives the body alone."""
    if subject == "":
        return body
    return subject + separator + body


def normalize_text(text):
    """
    Standard form used ONLY to find duplicates (EDA 8.1, DEC-13a):
    lowercase, every non letter/digit becomes a space, single spaces.
    """
    text = str(text).lower()
    cleaned_characters = []
    for character in text:
        if character.isalnum():
            cleaned_characters.append(character)
        else:
            cleaned_characters.append(" ")
    return " ".join("".join(cleaned_characters).split())


def make_vectorizer(tfidf_settings):
    """TF-IDF with the same settings as the baseline (config baseline.tfidf)."""
    return TfidfVectorizer(
        lowercase=tfidf_settings["lowercase"],
        ngram_range=(tfidf_settings["ngram_range"][0], tfidf_settings["ngram_range"][1]),
        min_df=tfidf_settings["min_df"],
        max_features=tfidf_settings["max_features"],
        sublinear_tf=tfidf_settings["sublinear_tf"],
    )


# ====================================================================== steps 1-8
def load_and_clean(config, project_root):
    """Steps 1-7: read, filter English, keep columns, clean, join, drop short texts, check labels."""
    data_config = config["data"]
    cleaning = data_config["cleaning"]

    # Step 1: read the raw CSV
    raw_path = project_root / data_config["raw_path"]
    raw = pd.read_csv(raw_path, encoding="utf-8")
    logger.info("read %s: %d rows x %d columns", raw_path.name, raw.shape[0], raw.shape[1])

    # Step 2: English only
    tickets = raw[raw["language"] == data_config["language"]].copy()
    logger.info("language = %s: %d rows", data_config["language"], len(tickets))

    # Step 3: keep the needed columns; ticket_id = row number in the raw file (traceability)
    tickets["ticket_id"] = tickets.index
    tickets = tickets[["ticket_id"] + KEEP_COLUMNS].reset_index(drop=True)

    # Step 4: clean subject and body
    replace_newlines = cleaning["literal_newlines"] == "replace"
    normalize_spaces = cleaning["normalize_whitespace"]
    clean_subjects = []
    clean_bodies = []
    literal_newline_rows = 0
    for position in range(len(tickets)):
        body_value = tickets.at[position, "body"]
        if not pd.isna(body_value) and "\\n" in str(body_value):
            literal_newline_rows = literal_newline_rows + 1
        clean_subjects.append(clean_text(tickets.at[position, "subject"], replace_newlines, normalize_spaces))
        clean_bodies.append(clean_text(body_value, replace_newlines, normalize_spaces))
    tickets["subject"] = clean_subjects
    tickets["body"] = clean_bodies
    logger.info("bodies with the literal text \\n: %d (replaced: %s)", literal_newline_rows, replace_newlines)

    # Step 5: joined text
    joined_texts = []
    for position in range(len(tickets)):
        joined_texts.append(build_joined_text(tickets.at[position, "subject"],
                                              tickets.at[position, "body"],
                                              data_config["text_separator"]))
    tickets["text"] = joined_texts

    # Step 6: drop near-empty texts (DEC-11: guard, log the count)
    keep_flags = []
    for text in tickets["text"]:
        keep_flags.append(len(text) >= cleaning["min_text_chars"])
    removed_short = len(tickets) - sum(keep_flags)
    tickets = tickets[keep_flags].reset_index(drop=True)
    logger.info("removed texts shorter than %d characters: %d", cleaning["min_text_chars"], removed_short)

    # Step 7: every label must be in the config lists
    label_lists = {data_config["targets"]["type"]: data_config["types"],
                   data_config["targets"]["department"]: data_config["departments"]}
    for column in label_lists:
        allowed = set(label_lists[column])
        unknown = set()
        for label in tickets[column]:
            if label not in allowed:
                unknown.add(label)
        if len(unknown) > 0:
            raise ValueError(f"unknown labels in {column}: {sorted(unknown, key=str)}")
    logger.info("labels checked against config: OK")

    return tickets, {"rows_read": int(raw.shape[0]), "rows_language": int(len(tickets) + removed_short),
                     "literal_newline_bodies": literal_newline_rows, "removed_short": removed_short}


def remove_duplicates(tickets, enabled):
    """Step 8 (DEC-13a): keep the first ticket of each normalized text."""
    if not enabled:
        return tickets, 0
    seen = set()
    keep_flags = []
    for text in tickets["text"]:
        key = normalize_text(text)
        keep_flags.append(key not in seen)
        seen.add(key)
    removed = len(tickets) - sum(keep_flags)
    tickets = tickets[keep_flags].reset_index(drop=True)
    logger.info("removed duplicates (normalized text): %d", removed)
    return tickets, removed


# ====================================================================== step 9: families
def find_nearest_neighbors(vectors, neighbor_count, chunk_size=1000):
    """
    The neighbor_count most similar OTHER tickets of every ticket, with their cosine similarity
    (vectors are L2-normalized, so the dot product is the cosine).
    Returns a list (one entry per ticket) of lists of (position, similarity), most similar first.
    Neighbours with similarity <= 0 (no shared term) are left out.
    """
    ticket_count = vectors.shape[0]
    all_neighbors = []
    for chunk_start in range(0, ticket_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, ticket_count)
        similarity_table = (vectors[chunk_start:chunk_end] @ vectors.T).toarray()
        for row_in_chunk in range(chunk_end - chunk_start):
            ticket_position = chunk_start + row_in_chunk
            row_similarities = similarity_table[row_in_chunk]
            row_similarities[ticket_position] = -1.0           # ignore the ticket itself
            # numpy picks the best positions (a Python loop over 16k values per row is too slow)
            best_positions = np.argsort(-row_similarities)[:neighbor_count]
            neighbors = []
            for position in best_positions:
                similarity = float(row_similarities[position])
                if similarity > 0:
                    neighbors.append((int(position), similarity))
            all_neighbors.append(neighbors)
    return all_neighbors


def find_root(parent, ticket):
    """Union-Find: follow the pointers until a ticket points to itself (the family root)."""
    while parent[ticket] != ticket:
        ticket = parent[ticket]
    return ticket


def union_groups(parent, ticket_a, ticket_b):
    """Union-Find: join the two families (the root of b points to the root of a)."""
    root_a = find_root(parent, ticket_a)
    root_b = find_root(parent, ticket_b)
    if root_a == root_b:
        return False
    parent[root_b] = root_a
    return True


def build_families(texts, tfidf_settings, similarity_threshold, neighbor_count):
    """
    Step 9 (DEC-13): family id for every ticket.
    Link each ticket to each of its neighbor_count nearest neighbours whose cosine >= threshold,
    then Union-Find joins everything that is linked directly or through other tickets.
    """
    vectors = make_vectorizer(tfidf_settings).fit_transform(texts)
    all_neighbors = find_nearest_neighbors(vectors, neighbor_count)

    parent = []
    for ticket in range(len(texts)):
        parent.append(ticket)
    for ticket in range(len(texts)):
        for neighbor in all_neighbors[ticket]:
            neighbor_position = neighbor[0]
            similarity = neighbor[1]
            if similarity >= similarity_threshold:
                union_groups(parent, ticket, neighbor_position)

    family_ids = []
    for ticket in range(len(texts)):
        family_ids.append(find_root(parent, ticket))
    return family_ids


# ====================================================================== step 10: group split
def family_label(labels_in_family):
    """The most frequent label of a family (ties: alphabetical, so the result is fixed)."""
    counts = {}
    for label in labels_in_family:
        counts[label] = counts.get(label, 0) + 1
    best_label = None
    best_count = -1
    for label in sorted(counts):
        if counts[label] > best_count:
            best_label = label
            best_count = counts[label]
    return best_label


def stratified_group_split(tickets, stratify_column, shares, random_seed):
    """
    Step 10 (DEC-10, DEC-13): every family goes WHOLE to train, validation or test,
    and every department keeps about 70 / 15 / 15 of its tickets.

    How: 1) each family gets the department of most of its tickets (98% of families have one)
         2) for each department: shuffle its families with the seed and walk through them,
            counting tickets; a family goes to train while the count is below 70% of the
            department, then to validation below 85%, then to test.
    Output: list with "train" / "validation" / "test" for every ticket.
    """
    # Step 1: tickets and labels of every family
    family_rows = {}
    for position in range(len(tickets)):
        family_id = tickets.at[position, "family_id"]
        family_rows.setdefault(family_id, []).append(position)

    families_by_department = {}
    for family_id in sorted(family_rows):
        labels = []
        for position in family_rows[family_id]:
            labels.append(tickets.at[position, stratify_column])
        department = family_label(labels)
        families_by_department.setdefault(department, []).append(family_id)

    # Step 2: walk through the shuffled families of each department
    random_generator = random.Random(random_seed)
    train_cut = shares["train"]
    validation_cut = shares["train"] + shares["validation"]
    split_of_ticket = [None] * len(tickets)
    for department in sorted(families_by_department):
        family_list = families_by_department[department]
        random_generator.shuffle(family_list)

        department_total = 0
        for family_id in family_list:
            department_total = department_total + len(family_rows[family_id])

        tickets_so_far = 0
        for family_id in family_list:
            done_share = tickets_so_far / department_total
            if done_share < train_cut:
                split_name = "train"
            elif done_share < validation_cut:
                split_name = "validation"
            else:
                split_name = "test"
            for position in family_rows[family_id]:
                split_of_ticket[position] = split_name
            tickets_so_far = tickets_so_far + len(family_rows[family_id])

    return split_of_ticket


# ====================================================================== step 11: checks
def share_table(tickets, column):
    """Percent of each value of column inside each split (rows: values, columns: splits)."""
    table = pd.crosstab(tickets[column], tickets["split"], normalize="columns") * 100
    return table.round(1)


def leakage_check(tickets, tfidf_settings, threshold, chunk_size=1000):
    """
    For every validation/test ticket: similarity to its nearest TRAIN ticket.
    TF-IDF is fitted on train only (as the baseline will be).
    Returns dict: tickets checked, tickets with a train neighbour >= threshold, their share, max.
    """
    train_texts = list(tickets.loc[tickets["split"] == "train", "text"])
    other_texts = list(tickets.loc[tickets["split"] != "train", "text"])
    vectorizer = make_vectorizer(tfidf_settings)
    train_vectors = vectorizer.fit_transform(train_texts)
    other_vectors = vectorizer.transform(other_texts)

    over_threshold = 0
    max_similarity = 0.0
    for chunk_start in range(0, len(other_texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(other_texts))
        similarity_table = (other_vectors[chunk_start:chunk_end] @ train_vectors.T).toarray()
        row_maxima = similarity_table.max(axis=1)
        for value in row_maxima:
            if value >= threshold:
                over_threshold = over_threshold + 1
            if value > max_similarity:
                max_similarity = float(value)

    return {"checked": len(other_texts), "threshold": threshold,
            "with_train_neighbour_over_threshold": over_threshold,
            "share_percent": round(over_threshold / len(other_texts) * 100, 2),
            "max_similarity": round(max_similarity, 4)}


# ====================================================================== step 12: save
def file_checksum(path):
    """SHA-256 of a file, so every run can prove it uses the same data (FR-4, NFR-3)."""
    digest = hashlib.sha256()
    with open(path, "rb") as data_file:
        for block in iter(lambda: data_file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_splits(tickets, output_config, processed_dir):
    """Save one file per split and return {split: {rows, file, sha256}}."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    columns = ["ticket_id", "subject", "body", "text", "type", "queue", "version", "family_id"]
    saved = {}
    for split_name in ["train", "validation", "test"]:
        split_rows = tickets[tickets["split"] == split_name][columns].reset_index(drop=True)
        path = processed_dir / output_config["files"][split_name]
        if output_config["format"] == "parquet":
            split_rows.to_parquet(path, index=False)
        else:
            split_rows.to_csv(path, index=False, encoding="utf-8")
        saved[split_name] = {"rows": int(len(split_rows)), "file": path.name, "sha256": file_checksum(path)}
        logger.info("saved %s: %d rows", path.name, len(split_rows))
    return saved


# ====================================================================== main
def run(config_path):
    project_root = find_project_root()
    config = load_config(config_path)
    data_config = config["data"]
    split_config = data_config["split"]
    tfidf_settings = config["baseline"]["tfidf"]
    seed = config["project"]["random_seed"]

    tickets, counts = load_and_clean(config, project_root)
    tickets, removed_duplicates = remove_duplicates(tickets, data_config["cleaning"]["remove_duplicates"])

    threshold = split_config["family"]["similarity_threshold"]
    neighbor_count = split_config["family"]["neighbors"]
    tickets["family_id"] = build_families(list(tickets["text"]), tfidf_settings, threshold, neighbor_count)
    family_count = tickets["family_id"].nunique()
    largest_family = int(tickets["family_id"].value_counts().max())
    logger.info("families (%d nearest neighbours, similarity >= %.2f): %d, largest %d tickets",
                neighbor_count, threshold, family_count, largest_family)

    shares = {"train": split_config["train"], "validation": split_config["validation"], "test": split_config["test"]}
    tickets["split"] = stratified_group_split(tickets, split_config["stratify_by"], shares, seed)

    # Checks
    families_in_more_than_one_split = int((tickets.groupby("family_id")["split"].nunique() > 1).sum())
    if families_in_more_than_one_split != 0:
        raise RuntimeError(f"{families_in_more_than_one_split} families are in more than one split")
    split_counts = tickets["split"].value_counts()
    for split_name in ["train", "validation", "test"]:
        logger.info("%-10s %6d tickets (%.1f%%)", split_name, split_counts[split_name],
                    split_counts[split_name] / len(tickets) * 100)
    department_shares = share_table(tickets, data_config["targets"]["department"])
    type_shares = share_table(tickets, data_config["targets"]["type"])
    version_shares = share_table(tickets, "version")
    leakage = leakage_check(tickets, tfidf_settings, split_config["leakage_check_threshold"])
    logger.info("leakage check: %d of %d validation/test tickets have a train neighbour >= %.2f (%.2f%%)",
                leakage["with_train_neighbour_over_threshold"], leakage["checked"],
                leakage["threshold"], leakage["share_percent"])

    # Save
    processed_dir = project_root / data_config["processed_dir"]
    output_config = data_config["output"]
    saved = save_splits(tickets, output_config, processed_dir)
    summary = {
        "config": {"random_seed": seed, "split": split_config, "cleaning": data_config["cleaning"]},
        "counts": {**counts, "removed_duplicates": removed_duplicates, "rows_final": int(len(tickets)),
                   "families": int(family_count), "largest_family": largest_family},
        "files": saved,
        "department_share_percent": department_shares.to_dict(),
        "type_share_percent": type_shares.to_dict(),
        "version_share_percent": {str(key): value for key, value in version_shares.to_dict().items()},
        "leakage_check": leakage,
    }
    summary_path = processed_dir / output_config["summary_file"]
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, default=str)
    logger.info("summary written: %s", summary_path.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare train / validation / test splits.")
    parser.add_argument("--config", default=str(find_project_root() / "config.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(arguments.config)


if __name__ == "__main__":
    main()
