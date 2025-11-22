# %%
import json
from copy import deepcopy
from json import JSONDecodeError
from typing import Literal

from pydantic import BaseModel

BASE_JSON_TYPES = str | int | float | None
JSON_TYPES = dict | list | BASE_JSON_TYPES


class JsonChain(BaseModel):
    path: list[BASE_JSON_TYPES]
    chain_type: Literal['key', 'field']

    def with_suffix(self, value: str) -> 'JsonChain':
        clone = self.model_copy()
        clone.path.insert(0, value)
        return clone

    def __hash__(self):
        return hash((tuple(self.path), self.chain_type))


class MatchMetrics(BaseModel):
    true_positives: list[JsonChain]
    false_positives: list[JsonChain]
    false_negatives: list[JsonChain]

    @staticmethod
    def weighted_count(chains: list[JsonChain], key_weight: float = 1, field_weight: float = 1) -> int:
        weights = {
            'key': key_weight,
            'field': field_weight,
        }
        return sum(weights[c.chain_type] for c in chains)

    def f1_score(self, key_weight: float = 1, field_weight: float = 9):
        tp_count = self.weighted_count(self.true_positives, key_weight, field_weight)
        fp_count = self.weighted_count(self.false_positives, key_weight, field_weight)
        fn_count = self.weighted_count(self.false_negatives, key_weight, field_weight)
        return 2 * tp_count / (
                2 * tp_count +
                fp_count +
                fn_count
        )


def json_chain_paths(json_dict: JSON_TYPES, ordered: bool = False) -> list[JsonChain]:
    """
    Generate a list of JSON path chains for a given JSON-like structure.

    This function traverses the provided JSON-like structure recursively to generate
    a comprehensive list of paths (`JsonChain` objects). For dictionaries, the keys
    are used as part of the path. For lists, indices (or placeholders if ordering
    is not required) are used as part of the path. The resulting `JsonChain` objects
    describe the paths to each element (fields or keys) in the JSON structure.

    :param json_dict: The input JSON-like structure. Expected to be a nested
        combination of dictionaries, lists, and primitive values.
    :param ordered: A boolean flag denoting whether list indices are included
        in paths for list elements. If False, placeholders ('[]') replace actual
        indices.
    :return: A list of `JsonChain` objects, each representing a unique path in
        the JSON-like structure.

    """
    json_chains = []
    if isinstance(json_dict, dict):
        for key, value in json_dict.items():
            json_chains.append(JsonChain(path=[key], chain_type='key'))
            json_chains.extend([chain.with_suffix(key) for chain in json_chain_paths(value)])
    elif isinstance(json_dict, list):
        for i, value in enumerate(json_dict):
            key = f'[{i}]' if ordered else '[]'
            json_chains.append(JsonChain(path=[key], chain_type='key'))
            json_chains.extend([chain.with_suffix(key) for chain in json_chain_paths(value)])
    else:
        json_chains.append(JsonChain(path=[json_dict], chain_type='field'))
    return json_chains


def json_match_metrics(
        expected_json: JSON_TYPES,
        predicted_json: JSON_TYPES,
) -> MatchMetrics:
    '''
    Produces all the counts for the match metrics like true positives, false positives, false negatives
    :param expected_json: The true JSON structure.
    :param predicted_json: The predicted JSON structure.
    :return: MatchMetrics (gathers all the counts in a single class)
    '''
    expected_json_chains = json_chain_paths(expected_json)
    predicted_json_chains = json_chain_paths(predicted_json)
    tp = []
    fp = []
    fn = []
    for expected_json_chain in expected_json_chains:
        if expected_json_chain in predicted_json_chains:
            tp += [expected_json_chain]
            predicted_json_chains.remove(expected_json_chain)
        else:
            fn += [expected_json_chain]

    for predicted_json_chain in predicted_json_chains:
        fp += [predicted_json_chain]

    return MatchMetrics(true_positives=tp, false_positives=fp, false_negatives=fn)


def evaluate_json(expected_json: JSON_TYPES, predicted_json_str: str) -> float:
    try:
        predicted_json = json.loads(predicted_json_str)
    except JSONDecodeError:
        return 0

    metrics = json_match_metrics(expected_json, predicted_json)
    return metrics.f1_score()


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    Calculate the mean score of predictions made in a submission against the ground truth solution.

    The function takes a ground truth DataFrame and a submission DataFrame, merges them using the "id" column,
    and computes a score for each row by applying the evaluation function to the true answers
    and the predicted values. The average of all scores is returned.

    Parameters:
    solution : pd.DataFrame
        A DataFrame containing the true answers. Must include a column named "id".
    submission : pd.DataFrame
        A DataFrame containing the predicted answers. Must include a column named "id".

    Returns:
    float
        The mean score computed by comparing the true answers with the predictions.

    Raises:
    AssertionError
        If either the solution or submission DataFrame does not contain the "id" column.
    """
    assert "id" in solution.columns and "id" in submission.columns, "Missing 'id' column"
    df = solution.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))
    df['score'] = df.apply(lambda row: evaluate_json(json.loads(row['answer']), row['prediction']), axis=1)
    return df.score.mean()


def main():
    expected_json = {
        'purchases': [
            {
                'product_name': 'Antitranspirante .Rexona Motionsense Antibacterial Roll Onx50ml',
                'units': 3
            },
            {
                "product_name": "Rey Bolsa Clavo Pepa ",
                "units": 9
            },
            {
                "product_name": "Chocolates Witors selecci\u00f3n especial ",
                "units": 6
            }
        ],
        'buyer': {'name': 'Laura Mueller', 'email': None}
    }
    predicted_json = deepcopy(expected_json)
    predicted_json['buyer'] = []

    metrics = json_match_metrics(expected_json, predicted_json)
    print(metrics)
    print(metrics.f1_score(key_weight=1, field_weight=1))


if __name__ == "__main__":
    main()
