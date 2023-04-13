from functools import wraps


def gather_params(querycls):
    def _accept_query(func):
        @wraps(func)
        def inner(self, **kwargs):
            if len(kwargs) == 1 and 'params' in kwargs:
                return func(self, params=kwargs['params'])
            else:
                query_obj = querycls.parse_obj(kwargs)
                return func(self, params=query_obj)
        return inner
    return _accept_query
