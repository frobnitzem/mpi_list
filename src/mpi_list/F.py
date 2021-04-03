# Pure python module to provide functional syntactic sugar.
#
# lambda F: F.u(x)
#   can be written as just F.u(x)
#
# lambda F: F.u[x]
#   can be written as just F.u[x]
#
# This works for any chain of lookups followed by a single call () or index []
#   lambda F: F.a.b(x)
#     can be written as just F.a.b(x)
#
#   also F.a.b.c[y], etc.
#
# lambda F: F[x]
#   can be written as just F[x]
#
# lambda F: F.u
#   can be written as just F('u')
#   Note: F.u is incorrect here, since F.u would expect to be called or indexed.
#
# Algebraic combinations of these objects work too, e.g.:
# lambda F: F

class AlgFn:
    # Algebraic function that can be
    # added/subtracted/compared/multiplied, etc.
    # f : elem -> new elem
    def __init__(self, f):
        self.f = f
    def __call__(self, x):
        return self.f(x)
    def __add__(a, b):
        def f(x):
            return a(x) + b(x)
        return AlgFn(f)
    def __sub__(a, b):
        def f(x):
            return a(x) - b(x)
        return AlgFn(f)
    def __mul__(a, b):
        def f(x):
            return a(x) * b(x)
        return AlgFn(f)
    def __div__(a, b):
        def f(x):
            return a(x) / b(x)
        return AlgFn(f)
    def __lt__(a, b):
        def f(x):
            return a(x) < b(x)
        return AlgFn(f)
    def __le__(a, b):
        def f(x):
            return a(x) <= b(x)
        return AlgFn(f)
    def __eq__(a, b):
        def f(x):
            return a(x) == b(x)
        return AlgFn(f)
    def __ne__(a, b):
        def f(x):
            return a(x) != b(x)
        return AlgFn(f)
    def __gt__(a, b):
        def f(x):
            return a(x) > b(x)
        return AlgFn(f)
    def __ge__(a, b):
        def f(x):
            return a(x) >= b(x)
        return AlgFn(f)

# Eventually, we will want to lookup the attr.
# Meanwhile, we don't know if the object
# is a function or class to be indexed.
class CallableAttr:
    def __init__(self, attrs):
        assert isinstance(attrs, list)
        self.attrs = attrs
    def __call__(self, *args, **kws):
        #if len(args) == 1 and len(kws) == 0:
        #    if hasattr(df, self.attr):
        #        print(f"Warning: potentially incorrect use of F.{self.attr}")
        #
        # it was a function, store the call params
        def lm(df):
            # follow the attr chain
            for a in self.attrs:
                df = getattr(df, a)
            return df(*args, **kws)
        return AlgFn(lm)
    def __getattr__(self, attr):
        return CallableAttr(self.attrs + [attr])
    def __getitem__(self, key):
        def lm(df):
            for a in self.attrs:
                df = getattr(df, a)
            return df[key]
        return AlgFn(lm)

class TF:
    def __getattr__(self, attr):
        # e.g. F.attr(key) or F.attr[key]
        return CallableAttr([attr])
    def __getitem__(self, key):
        # e.g. F[key]
        def lm(df):
            return df[key]
        return AlgFn(lm)
    def __call__(self, attr):
        def lm(df):
            return getattr(df, attr)
        return AlgFn(lm)

# This is the useful closure-creating object
F = TF()
