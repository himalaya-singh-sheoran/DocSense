class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.data = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, phrase, data):
        words = phrase.split()
        node = self.root
        for word in words:
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
        node.is_end = True
        node.data = data
    
    def get_longest_matching_phrase(self, tokens, start_idx):
        node = self.root
        current_phrase = []
        longest_phrase = None
        data = None
        for i in range(start_idx, len(tokens)):
            word = tokens[i]
            if word in node.children:
                node = node.children[word]
                current_phrase.append(word)
                if node.is_end:
                    longest_phrase = ' '.join(current_phrase)
                    data = node.data
            else:
                break
        return (longest_phrase, data) if longest_phrase else (None, None)

class NLtoSQLConverter:
    def __init__(self, metadata_trie, manager_lob_trie):
        self.metadata_trie = metadata_trie
        self.manager_lob_trie = manager_lob_trie
        self.operators = {
            'less than': '<', 'greater than': '>',
            'less': '<', 'greater': '>',
            'equal to': '=', 'equals': '=',
            '=': '=', '>': '>', '<': '<',
            '>=': '>=', '<=': '<=',
            'at least': '>=', 'at most': '<='
        }
    
    def convert(self, query):
        tokens = self.preprocess(query)
        consumed = [False] * len(tokens)
        conditions = []
        select_clause = "SELECT *"
        
        # Detect aggregate queries
        number_of_idx = self._find_phrase_indices(tokens, ["number", "of"])
        if number_of_idx != -1:
            select_clause = "SELECT COUNT(*)"
            for j in range(number_of_idx, number_of_idx + 2):
                if j < len(consumed):
                    consumed[j] = True
        
        i = 0
        while i < len(tokens):
            if consumed[i]:
                i += 1
                continue
            # Check for manager/lob value
            phrase, column = self.manager_lob_trie.get_longest_matching_phrase(tokens, i)
            if phrase:
                conditions.append(f"{column} = '{phrase}'")
                phrase_length = len(phrase.split())
                for j in range(i, i + phrase_length):
                    consumed[j] = True
                i += phrase_length
                continue
            # Check for metadata column synonym
            phrase, column = self.metadata_trie.get_longest_matching_phrase(tokens, i)
            if phrase:
                phrase_length = len(phrase.split())
                value_start = i + phrase_length
                # Check for operator
                op_phrase, operator = self._get_operator(tokens, value_start)
                if op_phrase:
                    op_length = len(op_phrase.split())
                    value_start += op_length
                    value = self._extract_value(tokens, value_start)
                    if value is not None:
                        conditions.append(f"{column} {operator} {value}")
                        end_idx = value_start + 1  # assuming value is single token
                        for j in range(i, end_idx):
                            if j < len(consumed):
                                consumed[j] = True
                        i = end_idx
                        continue
                # No operator, check for value
                if value_start >= len(tokens):
                    i += phrase_length
                    continue
                # Check for value in manager_lob trie
                value_phrase, value_column = self.manager_lob_trie.get_longest_matching_phrase(tokens, value_start)
                if value_phrase:
                    conditions.append(f"{column} = '{value_phrase}'")
                    value_length = len(value_phrase.split())
                    for j in range(i, value_start + value_length):
                        consumed[j] = True
                    i = value_start + value_length
                else:
                    # Take the next token as value
                    value = tokens[value_start]
                    conditions.append(f"{column} = '{value}'")
                    for j in range(i, value_start + 1):
                        consumed[j] = True
                    i = value_start + 1
                continue
            i += 1
        
        sql = f"{select_clause} FROM attrition_data"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return sql
    
    def preprocess(self, query):
        query = query.lower()
        query = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in query])
        tokens = query.split()
        # Simple plural handling: remove trailing 's'
        processed = []
        for token in tokens:
            if len(token) > 1 and token.endswith('s'):
                processed.append(token[:-1])
            else:
                processed.append(token)
        return processed
    
    def _get_operator(self, tokens, start_idx):
        max_phrase_length = 2
        for length in range(max_phrase_length, 0, -1):
            if start_idx + length > len(tokens):
                continue
            candidate = ' '.join(tokens[start_idx:start_idx+length])
            if candidate in self.operators:
                return (candidate, self.operators[candidate])
        return (None, None)
    
    def _extract_value(self, tokens, start_idx):
        if start_idx >= len(tokens):
            return None
        # Extract first numeric token
        token = tokens[start_idx]
        if token.replace('.', '', 1).isdigit():
            return token
        return None
    
    def _find_phrase_indices(self, tokens, phrase_words):
        phrase_len = len(phrase_words)
        for i in range(len(tokens) - phrase_len + 1):
            if tokens[i:i+phrase_len] == phrase_words:
                return i
        return -1

# Example usage:
if __name__ == "__main__":
    metadata_trie = Trie()
    metadata_trie.insert("department", "lob")
    metadata_trie.insert("lob", "lob")
    metadata_trie.insert("manager", "manager_name")
    metadata_trie.insert("city", "city")
    metadata_trie.insert("gender", "gender")
    metadata_trie.insert("exit type", "exit_type")
    metadata_trie.insert("country", "country")
    metadata_trie.insert("job level", "job_level")
    metadata_trie.insert("job title", "job_title")
    metadata_trie.insert("tenure", "tenure")
    
    manager_lob_trie = Trie()
    manager_lob_trie.insert("sales", "lob")
    manager_lob_trie.insert("john doe", "manager_name")
    manager_lob_trie.insert("new york", "city")
    manager_lob_trie.insert("retail sales", "sub_lob")
    manager_lob_trie.insert("technology", "lob")
    manager_lob_trie.insert("senior software engineer", "job_title")
    manager_lob_trie.insert("sam williams", "manager_name")
    
    converter = NLtoSQLConverter(metadata_trie, manager_lob_trie)
    
    queries = [
        "Number of employee with tenure less 10 months",
        "Number of Senior Software Engineers working in Technology with tenure greater than 24 months working under Sam Williams",
        "give me gender ratio of employees current working under Roger woods"
    ]
    
    for query in queries:
        print(f"Input: {query}")
        print("SQL:", converter.convert(query))
        print()
