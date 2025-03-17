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
    
    def convert(self, query):
        tokens = self.preprocess(query)
        conditions = []
        consumed = [False] * len(tokens)
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
                if value_start >= len(tokens):
                    i += phrase_length
                    continue
                # Look for value
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
        sql = "SELECT * FROM attrition_data"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return sql
    
    def preprocess(self, query):
        query = query.lower()
        query = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in query])
        tokens = query.split()
        return tokens

# Example usage:
if __name__ == "__main__":
    # Initialize metadata trie with column synonyms
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
    
    # Initialize manager/lob trie with known values and their columns
    manager_lob_trie = Trie()
    manager_lob_trie.insert("sales", "lob")
    manager_lob_trie.insert("john doe", "manager_name")
    manager_lob_trie.insert("new york", "city")
    manager_lob_trie.insert("retail sales", "sub_lob")
    
    converter = NLtoSQLConverter(metadata_trie, manager_lob_trie)
    
    # Sample input queries
    queries = [
        "employees in sales led by john doe",
        "gender female",
        "department retail sales city new york",
        "exit type voluntary and country canada"
    ]
    
    for query in queries:
        print(f"Input: {query}")
        print("SQL:", converter.convert(query))
        print()
