class Posting:
    def __init__(self, doc_id, payload):
        self.doc_id = doc_id
        self.payload = payload

    def __eq__(self, other):
        if isinstance(other, Posting):
            return self.doc_id == other.doc_id
        return False

    def __hash__(self):
        return hash(self.doc_id)
