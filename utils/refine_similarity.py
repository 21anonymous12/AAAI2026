from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
import ast
import json
import dis

def calculate_text_similarity(text1, text2):
    results = {}

    # 1. Levenshtein Distance
    levenshtein_sim = 1 - levenshtein_distance(text1, text2) / max(len(text1), len(text2))
    results['Levenshtein Similarity'] = levenshtein_sim

    # 2. difflib Similarity
    difflib_sim = SequenceMatcher(None, text1, text2).ratio()
    results['Difflib Similarity'] = difflib_sim

    # 3. Jaccard Similarity
    set1 = set(text1.split())
    set2 = set(text2.split())
    jaccard_sim = len(set1 & set2) / len(set1 | set2)
    results['Jaccard Similarity'] = jaccard_sim

    # 4. Cosine Similarity with TF-IDF
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf_matrix = vectorizer.transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]
    results['Cosine Similarity with TF-IDF'] = cosine_sim

    return results


def calculate_code_similarity(code1, code2):
    results = {}

    # 1. Levenshtein Distance
    levenshtein_sim = 1 - levenshtein_distance(code1, code2) / max(len(code1), len(code2))
    results['Levenshtein Similarity'] = levenshtein_sim

    # 2. difflib Similarity
    difflib_sim = SequenceMatcher(None, code1, code2).ratio()
    results['Difflib Similarity'] = difflib_sim

    # 3. AST (Abstract Syntax Tree) Similarity
    def get_ast_structure(code):
        try:
            return ast.dump(ast.parse(code), annotate_fields=False)
        except Exception as e:
            return ''  # Return empty string if AST parsing fails

    ast1 = get_ast_structure(code1)
    ast2 = get_ast_structure(code2)
    ast_sim = SequenceMatcher(None, ast1, ast2).ratio()
    results['AST Similarity'] = ast_sim

    # 4. Opcode-based Similarity
    def get_opcodes(code):
        try:
            bytecode = dis.Bytecode(code)
            return [instr.opname for instr in bytecode]
        except Exception as e:
            return []

    opcodes1 = get_opcodes(code1)
    opcodes2 = get_opcodes(code2)
    common_opcodes = set(opcodes1) & set(opcodes2)
    total_opcodes = set(opcodes1) | set(opcodes2)
    opcode_sim = len(common_opcodes) / len(total_opcodes) if total_opcodes else 0.0
    results['Opcode Similarity'] = opcode_sim

    return results



