# CSE 132A Homework 3 - 3NF/BCNF Decomposition

## Overview

This assignment involves implementing normalization algorithms for relational databases. You will write code to decompose relations into Third Normal Form (3NF) and Boyce-Codd Normal Form (BCNF). The program must handle both explicit functional dependencies and infer them from sample data when they are not provided.

## Task Description

You are tasked with implementing a decomposition tool that:
1. **Infers functional dependencies** from sample data when explicit FDs are not provided
2. **Computes a minimal cover** of functional dependencies
3. **Synthesizes relations in 3NF** using the synthesis algorithm
4. **Decomposes relations into BCNF** using the decomposition algorithm

The starter code (`decompose.py`) provides the CLI interface, input/output handling, and validation. Your job is to implement the core algorithms in four functions:
- `infer_fds_from_sample()`
- `minimal_cover()`
- `synthesize_3nf()`
- `decompose_bcnf()`

---

## Instructions

### What You Need to Implement

#### 1. `infer_fds_from_sample(attributes, sample_rows)`

**Purpose**: Discover functional dependencies from sample data.

**Algorithm**: 
- For each possible subset `X` of attributes, check if `X → Y` holds for each remaining attribute `Y`
- An FD `X → Y` holds if: whenever two rows agree on all attributes in `X`, they also agree on `Y`
- **Important**: Only count an FD if `X` appears in at least 2 rows (support ≥ 2)
- Return all discovered FDs as a list of tuples: `(frozenset(X), frozenset(Y))`

**Return**: List of `(lhs, rhs)` tuples where both `lhs` and `rhs` are frozensets of attribute names.

#### 2. `minimal_cover(fds)`

**Purpose**: Reduce a set of functional dependencies to a minimal cover.

**Requirements**:
1. **Singleton RHS**: Split each FD so that the right-hand side contains exactly one attribute
2. **Minimize LHS**: Remove redundant attributes from the left-hand side of each FD
3. **Remove redundant FDs**: Eliminate any FD that can be inferred from the remaining FDs

**Algorithm outline**:
- Split FDs into singleton RHS
- For each FD `X → A`, try removing each attribute from `X` and check if the FD still holds
- Remove any FD that is implied by the remaining FDs using closure computation

**Return**: List of `(lhs, rhs)` tuples where `rhs` must be a singleton frozenset.

#### 3. `synthesize_3nf(attributes, minimal_fds)`

**Purpose**: Synthesize relations in Third Normal Form.

**Algorithm** (3NF Synthesis):
1. Start with a minimal cover of FDs
2. Group FDs by their left-hand side
3. For each group with the same LHS `X`, create a relation containing `X ∪ {all attributes on RHS}`
4. If no relation contains a candidate key, add one relation containing a candidate key
5. Remove any relation that is a subset of another relation

**Return**: An iterable of relations, where each relation is an iterable of attribute names (strings).

#### 4. `decompose_bcnf(attributes, minimal_fds)`

**Purpose**: Decompose a relation into Boyce-Codd Normal Form.

**Algorithm** (BCNF Decomposition):
1. Start with the original relation `R` containing all attributes
2. While there exists a relation `Ri` that is not in BCNF:
   - Find a violating FD `X → Y` where `X` is not a superkey of `Ri`
   - Decompose `Ri` into: `R1 = X ∪ Y` and `R2 = Ri - Y`
   - Replace `Ri` with `R1` and `R2`
3. Continue until all relations are in BCNF

**Return**: An iterable of relations, where each relation is an iterable of attribute names (strings).

### Helper Functions and Utilities

The starter code provides several helper functions you may find useful:
- `parse_fd_json()`: Parses FDs from JSON format
- `normalize_relations()`: Validates and normalizes your output relations
- `normalize_fds_for_output()`: Validates and normalizes your output FDs
- `build_output_payload()`: Constructs the final JSON output

---

## File Structure

```
cse132a-hw3/
├── decompose.py          # Main implementation file (you edit this)
├── README.md             # This file
├── data_tester.py        # Lossless join tester (provided)
├── tests/                # Test input files
```

---

## How to Run

### Basic Usage

**Decompose to 3NF**:
```bash
python3 decompose.py --input tests/testcases.json --to 3nf
python3 decompose.py --input tests/multi_testcases.json --to 3nf
```

**Decompose to BCNF**:
```bash
python3 decompose.py --input tests/testcases.json --to bcnf
python3 decompose.py --input tests/multi_testcases.json --to bcnf
```


### Input Format

Input files are in JSON format with the following structure:

```json
{
  "relationName": "R",
  "attributes": ["A", "B", "C"],
  "functionalDependencies": [
    { "left": ["A"], "right": ["B"] },
    { "left": ["B"], "right": ["C"] }
  ],
  "sample_data": [
    { "A": "a1", "B": "b1", "C": "c1" },
    { "A": "a2", "B": "b2", "C": "c2" }
  ]
}
```

**Notes**:
- `functionalDependencies` is optional. If absent, FDs will be inferred from `sample_data`
- `sample_data` is required when `functionalDependencies` is not provided
- `relationName` defaults to "R" if not specified

### Output Format

The program outputs JSON with the following structure:

```json
{
  "relations": [
    { "name": "R1", "attributes": ["A", "B"] },
    { "name": "R2", "attributes": ["B", "C"] }
  ],
  "inferredFDs": [
    { "left": ["A"], "right": ["B"] },
    { "left": ["B"], "right": ["C"] }
  ]
}
```

**Properties**:
- Each relation has a generated name (`R1`, `R2`, etc.) and a sorted list of attributes
- `inferredFDs` contains the minimal cover of functional dependencies
- Each FD has a sorted `left` (LHS) and `right` (RHS), where RHS must be a singleton



## Grading

Each test case is worth **1 point**. Your submission will be evaluated on:

1. **Correctness**: Does your decomposition produce correct 3NF/BCNF relations?
2. **Completeness**: Do you handle all test cases including edge cases?
3. **FD Inference**: Can you correctly infer functional dependencies from sample data?
4. **Minimal Cover**: Is your minimal cover truly minimal and canonical?
5. **Lossless Join**: Do your decompositions preserve all information from the original relation?

**Scoring breakdown**: Only public test case: 1 point each test

**Partial credit**: No partial credit will be given.

**Code and Test Version**: The code version will be the most updated version in the repo.

