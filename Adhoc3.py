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
    def __init__(self, metadata_trie, value_trie):
        self.metadata_trie = metadata_trie
        self.value_trie = value_trie
        self.operators = {
            'less than': '<', 'greater than': '>', 'equal to': '=',
            'at least': '>=', 'at most': '<=', 'between': 'BETWEEN',
            '!=': '!=', 'not equal to': '!=', 'like': 'LIKE'
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
            'limit': None
        }

        i = 0
        while i < len(tokens):
            # Handle LIMIT/TOP
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

    def _parse_condition(self, tokens, start_idx):
        # Try to find column first
        col_phrase, col_data = self.metadata_trie.get_longest_matching_phrase(tokens, start_idx)
        if col_phrase:
            remaining = tokens[start_idx + len(col_phrase.split()):]
            return self._process_operator(col_data, remaining, start_idx + len(col_phrase.split()))
        
        # Try to find value
        val_phrase, val_data = self.value_trie.get_longest_matching_phrase(tokens, start_idx)
        if val_phrase:
            return {
                'condition': f"{val_data} = '{val_phrase}'",
                'token_count': len(val_phrase.split()),
                'type': 'simple'
            }
        return None

    def _process_operator(self, column, remaining_tokens, start_idx):
        op_phrase, operator = self._get_operator(remaining_tokens, 0)
        if operator:
            op_length = len(op_phrase.split())
            value = self._parse_value(remaining_tokens[op_length:], column)
            return {
                'condition': f"{column} {operator} {value['value']}",
                'token_count': start_idx + op_length + len(value['tokens']),
                'type': 'aggregate' if '(' in column else 'simple'
            }
        
        # Handle implicit equals
        if remaining_tokens:
            value = self._parse_value(remaining_tokens, column)
            return {
                'condition': f"{column} = {value['value']}",
                'token_count': start_idx + len(value['tokens']),
                'type': 'simple'
            }
        return None

    def _parse_value(self, tokens, column):
        # Try value trie first
        val_phrase, val_data = self.value_trie.get_longest_matching_phrase(tokens, 0)
        if val_phrase:
            return {'value': f"'{val_phrase}'", 'tokens': val_phrase.split()}
        
        # Numeric value
        if tokens and tokens[0].replace('.', '', 1).isdigit():
            return {'value': tokens[0], 'tokens': [tokens[0]]}
        
        # Default to string literal
        return {'value': f"'{tokens[0]}'", 'tokens': [tokens[0]]} if tokens else None

    def _get_operator(self, tokens, start_idx):
        for length in [2, 1]:  # Check multi-word operators first
            if start_idx + length > len(tokens):
                continue
            phrase = ' '.join(tokens[start_idx:start_idx+length])
            if phrase in self.operators:
                return (phrase, self.operators[phrase])
        return (None, None)

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
        
        lower = self._parse_value(tokens[between_idx+1:and_idx], col_data)
        upper = self._parse_value(tokens[and_idx+1:], col_data)
        
        return {
            'condition': f"{col_data} BETWEEN {lower['value']} AND {upper['value']}",
            'token_count': and_idx + len(upper['tokens']) + 1 - start_idx
        }

    def _build_sql(self, state):
        select_clause = "SELECT " + (', '.join(state['select']['columns']) if state['select']['columns'] else "*")
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

    # Helper methods
    def _match_phrase(self, tokens, start_idx, phrase_words):
        if start_idx + len(phrase_words) > len(tokens):
            return False
        return tokens[start_idx:start_idx+len(phrase_words)] == phrase_words

    def _find_phrase(self, tokens, start_idx, phrase_words):
        for i in range(start_idx, len(tokens) - len(phrase_words) + 1):
            if tokens[i:i+len(phrase_words)] == phrase_words:
                return i
        return -1

    def _parse_group_by(self, tokens, start):
        if self._match_phrase(tokens, start, ['group', 'by']):
            columns = []
            i = start + 2
            while i < len(tokens) and not self._is_clause_start(tokens[i]):
                col_phrase, col_data = self.metadata_trie.get_longest_matching_phrase(tokens, i)
                if col_phrase:
                    columns.append(col_data)
                    i += len(col_phrase.split())
                else:
                    break
            return {'columns': columns, 'token_count': i - start}
        return None

    def _parse_order_by(self, tokens, start):
        if self._match_phrase(tokens, start, ['order', 'by']) or \
           self._match_phrase(tokens, start, ['sort', 'by']):
            clauses = []
            i = start + 2
            while i < len(tokens) and not self._is_clause_start(tokens[i]):
                col_phrase, col_data = self.metadata_trie.get_longest_matching_phrase(tokens, i)
                if not col_phrase:
                    break
                
                direction = 'ASC'
                if i + len(col_phrase.split()) < len(tokens):
                    next_word = tokens[i + len(col_phrase.split())]
                    if next_word in self.sort_directions:
                        direction = 'DESC' if next_word in ['desc', 'descending'] else 'ASC'
                        i += 1
                
                clauses.append((col_data, direction))
                i += len(col_phrase.split())
            return {'clauses': clauses, 'token_count': i - start}
        return None

    def _parse_limit(self, tokens, start):
        if self._match_phrase(tokens, start, ['top']) or \
           self._match_phrase(tokens, start, ['limit']):
            try:
                limit = int(tokens[start+1])
                return {'limit': limit, 'token_count': 2}
            except (IndexError, ValueError):
                pass
        return None

    def _is_clause_start(self, token):
        return token in self.clause_keywords

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
    metadata_trie.insert("count employees", "COUNT(*)")
    
    value_trie = Trie()
    value_trie.insert("senior software engineer", "job_title")
    value_trie.insert("technology", "lob")
    value_trie.insert("sam williams", "manager_name")
    value_trie.insert("new york", "city")

    converter = NLtoSQLConverter(metadata_trie, value_trie)

    queries = [
        "Top 5 cities with highest average tenure group by city having avg tenure > 24 order by avg tenure desc",
        "Number of senior software engineers in technology with tenure between 12 and 24 months",
        "Show me managers in new york group by department having count employees > 10 order by count desc limit 5"
    ]

    for query in queries:
        print(f"Input: {query}")
        print("SQL:", converter.convert(query))
        print()
