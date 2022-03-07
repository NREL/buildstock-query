
from concurrent.futures import Future


class FutureDf(Future):
    def __init__(self, df, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.df = df
        self.set_result(self.df)

    def running(self):
        return False

    def done(self):
        return True

    def cancelled(self):
        return False

    def result(self, timeout=None):
        return self.df

    def as_pandas(self):
        return self.df
