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
        self._update_data(node, data)
    
    def _update_data(self, node, data):
        if node.data is None:
            node.data = data
        elif isinstance(node.data, list):
            node.data.append(data)
        else:
            node.data = [node.data, data]
    
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
    def __init__(self, metadata_trie, value_trie):
        self.metadata_trie = metadata_trie
        self.value_trie = value_trie
        self.operators = {
            'less than': '<', 'greater than': '>', 'equal to': '=',
            'at least': '>=', 'at most': '<=', 'between': 'BETWEEN',
            '!=', 'not equal to': '!=', 'like': 'LIKE'
        }
        self.aggregates = {'count', 'sum', 'avg', 'min', 'max'}
        self.sort_directions = {'asc', 'desc', 'ascending', 'descending'}
        self.clause_keywords = {'where', 'group', 'order', 'having', 'limit', 'top'}

    def convert(self, query):
        tokens = self.preprocess(query)
        state = {
            'select': {'columns': [], 'aggregates': False},
            'where': [],
            'group_by': [],
            'having': [],
            'order_by': [],
            'limit': None,
            'joins': []
        }

        i = 0
        while i < len(tokens):
            # Handle LIMIT
            limit = self._parse_limit(tokens, i)
            if limit:
                state['limit'] = limit['limit']
                i += limit['token_count']
                continue

            # Handle GROUP BY
            group_by = self._parse_group_by(tokens, i)
            if group_by:
                state['group_by'] = group_by['columns']
                i += group_by['token_count']
                continue

            # Handle ORDER BY
            order_by = self._parse_order_by(tokens, i)
            if order_by:
                state['order_by'] = order_by['clauses']
                i += order_by['token_count']
                continue

            # Handle HAVING
            if self._match_phrase(tokens, i, ['having']):
                having_cond = self._parse_condition(tokens, i+1)
                if having_cond:
                    state['having'].append(having_cond['condition'])
                    i += having_cond['token_count'] + 1
                continue

            # Handle BETWEEN
            between_cond = self._parse_between(tokens, i)
            if between_cond:
                state['where'].append(between_cond['condition'])
                i += between_cond['token_count']
                continue

            # Handle standard conditions
            cond_info = self._parse_condition(tokens, i)
            if cond_info:
                target = 'having' if cond_info['type'] == 'aggregate' and state['group_by'] else 'where'
                state[target].append(cond_info['condition'])
                i += cond_info['token_count']
                continue

            # Handle select columns
            col_phrase, col_data = self.metadata_trie.get_longest_matching_phrase(tokens, i)
            if col_phrase:
                state['select']['columns'].append(col_data)
                i += len(col_phrase.split())
                continue

            i += 1

        return self._build_sql(state)

    # Added missing helper methods
    def _match_phrase(self, tokens, start_idx, phrase_words):
        phrase_len = len(phrase_words)
        if start_idx + phrase_len > len(tokens):
            return False
        return tokens[start_idx:start_idx+phrase_len] == phrase_words

    def _find_phrase(self, tokens, start_idx, phrase_words):
        phrase_len = len(phrase_words)
        for i in range(start_idx, len(tokens) - phrase_len + 1):
            if tokens[i:i+phrase_len] == phrase_words:
                return i
        return -1

    def _is_clause_start(self, token):
        return token in self.clause_keywords

    def _parse_limit(self, tokens, start):
        limit_words = ['top', 'first', 'limit']
        for word in limit_words:
            if self._match_phrase(tokens, start, [word]):
                try:
                    limit = int(tokens[start + 1])
                    return {'limit': limit, 'token_count': 2}
                except (IndexError, ValueError):
                    pass
        return None

    def _parse_group_by(self, tokens, start):
        if self._match_phrase(tokens, start, ['group', 'by']) or \
           self._match_phrase(tokens, start, ['per']):
            start += 2 if tokens[start] == 'group' else 1
            columns = []
            while start < len(tokens) and not self._is_clause_start(tokens[start]):
                col_phrase, col_data = self.metadata_trie.get_longest_matching_phrase(tokens, start)
                if col_phrase:
                    columns.append(col_data)
                    start += len(col_phrase.split())
                else:
                    break
            return {'columns': columns, 'token_count': start - (start - len(columns))}

    def _parse_order_by(self, tokens, start):
        if self._match_phrase(tokens, start, ['order', 'by']) or \
           self._match_phrase(tokens, start, ['sort', 'by']):
            start += 2
            clauses = []
            while start < len(tokens) and not self._is_clause_start(tokens[start]):
                col_phrase, col_data = self.metadata_trie.get_longest_matching_phrase(tokens, start)
                if not col_phrase:
                    break

                direction = 'ASC'
                if start + len(col_phrase.split()) < len(tokens):
                    next_word = tokens[start + len(col_phrase.split())]
                    if next_word in self.sort_directions:
                        direction = 'DESC' if next_word in ['desc', 'descending'] else 'ASC'
                        start += 1

                clauses.append((col_data, direction))
                start += len(col_phrase.split())
            return {'clauses': clauses, 'token_count': start - (start - len(clauses))}

    def _parse_between(self, tokens, start_idx):
        col_phrase, col_data = self.metadata_trie.get_longest_matching_phrase(tokens, start_idx)
        if not col_phrase:
            return None

        between_idx = self._find_phrase(tokens, start_idx + 1, ['between'])
        if between_idx == -1:
            return None

        and_idx = self._find_phrase(tokens, between_idx + 1, ['and'])
        if and_idx == -1:
            return None

        lower = self._parse_value(tokens[between_idx + 1:and_idx], col_data)
        upper = self._parse_value(tokens[and_idx + 1:], col_data)

        return {
            'condition': f"{col_data} BETWEEN {lower['value']} AND {upper['value']}",
            'token_count': and_idx + len(upper['tokens']) + 1 - start_idx
        }

    def _build_sql(self, state):
        select_clause = "SELECT "
        if state['select']['columns']:
            select_clause += ', '.join(state['select']['columns'])
        else:
            select_clause += "*"

        sql = [f"{select_clause} FROM attrition_data"]

        if state['where']:
            sql.append(f"WHERE {' AND '.join(state['where'])}")

        if state['group_by']:
            sql.append(f"GROUP BY {', '.join(state['group_by'])}")
            if state['having']:
                sql.append(f"HAVING {' AND '.join(state['having'])}")

        if state['order_by']:
            order_clauses = [f"{col} {dir}" for col, dir in state['order_by']]
            sql.append(f"ORDER BY {', '.join(order_clauses)}")

        if state['limit'] is not None:
            sql.append(f"LIMIT {state['limit']}")

        return ' '.join(sql)

    def preprocess(self, query):
        query = query.lower()
        query = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in query])
        tokens = query.split()
        return [token[:-1] if len(token) > 1 and token.endswith('s') else token for token in tokens]

# Example usage
if __name__ == "__main__":
    metadata_trie = Trie()
    metadata_trie.insert("department", "lob")
    metadata_trie.insert("job title", "job_title")
    metadata_trie.insert("tenure", "tenure")
    metadata_trie.insert("city", "city")
    metadata_trie.insert("avg tenure", "AVG(tenure)")
    metadata_trie.insert("count", "COUNT(*)")
    
    value_trie = Trie()
    value_trie.insert("senior software engineer", "job_title")
    value_trie.insert("technology", "lob")
    value_trie.insert("sam williams", "manager_name")
    value_trie.insert("new york", "city")

    converter = NLtoSQLConverter(metadata_trie, value_trie)

    complex_queries = [
        "Top 5 cities with highest average tenure group by city having avg tenure > 24 order by avg tenure desc",
        "Number of senior software engineers in technology with tenure between 12 and 24 months",
        "Show me managers in new york group by department having count > 10 order by count desc limit 5"
    ]

    for query in complex_queries:
        print(f"Input: {query}")
        print("SQL:", converter.convert(query))
        print()
