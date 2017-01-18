# A simple round off error estimation of polynomial expressions
# Alexey Solovyev, 2016
# https://github.com/monadius/poly_fp
# MIT License

from fractions import Fraction
from numbers import Rational

# This flag controls string conversion behavior of some internal objects.
# If True then additional information is printed.
verbose_flag = False

# If True then all variables are real-valued variables and hence
# all variables introduce rounding errors.
# If False then all variables are floating-point or fixed-point variables
# depending on their usage context.
real_vars_flag = False

# A template string for printing absolute values.
# Another possible value is "abs({0})".
abs_template = "|{0}|"

# A name of the relative error bound (machine epsilon).
eps_name = "eps"

# A name of the absolute error bound.
delta_name = "delta"


def set_verbose_flag(flag):
    assert(type(flag) is bool)
    global verbose_flag
    verbose_flag = flag


def set_real_vars_flag(flag):
    assert(type(flag) is bool)
    global real_vars_flag
    real_vars_flag = flag


def set_abs_template(s):
    assert(isinstance(s, basestring))
    global abs_template
    abs_template = s


def set_eps_name(name):
    assert(isinstance(name, basestring))
    global eps_name
    eps_name = name


def set_delta_name(name):
    assert(isinstance(name, basestring))
    global delta_name
    delta_name = name


class Variable:
    """Defines a variable """

    name = None

    def __init__(self, name):
        assert(isinstance(name, basestring))
        self.name = name

    def __repr__(self):
        return "Variable('{0}')".format(self.name)

    def __str__(self):
        if verbose_flag:
            return "var:" + self.name
        else:
            return self.name


class Constant:
    """Defines a constant """

    value = None

    def __init__(self, val):
        if isinstance(val, basestring) or isinstance(val, Rational):
            self.value = Fraction(val)
        else:
            raise TypeError("argument should be a string "
                            "or a Rational instance")

    def __str__(self):
        if verbose_flag:
            return "const:" + str(self.value)
        else:
            return str(self.value)


def convert_to_expr(val):
    """Converts a given value to an expression.
    Accepted values: Expr, Constant, Variable, string, Rational
    """
    if isinstance(val, Expr):
        return val
    elif isinstance(val, Constant):
        return ConstExpr(val)
    elif isinstance(val, Variable):
        return VarExpr(val)
    elif isinstance(val, basestring) or isinstance(val, Rational):
        return ConstExpr(Constant(val))
    else:
        raise TypeError("argument should be an instance of: "
                        "Expr, Constant, Variable, basestring, Rational")


class Expr:
    """A base class of expressions.

    This class overloads '+', '-' (unary and binary), and '*'.
    """

    def __neg__(self):
        return NegExpr(self)

    def __add__(self, other):
        return AddExpr(self, convert_to_expr(other))

    def __radd__(self, other):
        return AddExpr(convert_to_expr(other), self)

    def __sub__(self, other):
        return SubExpr(self, convert_to_expr(other))

    def __rsub__(self, other):
        return SubExpr(convert_to_expr(other), self)

    def __mul__(self, other):
        return MulExpr(self, convert_to_expr(other))

    def __rmul__(self, other):
        return MulExpr(convert_to_expr(other), self)


class NegExpr(Expr):
    """Represents a negation of an expression"""

    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return "-({0})".format(self.expr)
        

class AddExpr(Expr):
    """Represents a sum of two expressions """

    left = None
    right = None

    def __init__(self, left, right):
        assert(isinstance(left, Expr) and isinstance(right, Expr))
        self.left = left
        self.right = right

    def __str__(self):
        return "({0}) + ({1})".format(self.left, self.right)


class SubExpr(Expr):
    """Represents a difference of two expressions """

    left = None
    right = None

    def __init__(self, left, right):
        assert(isinstance(left, Expr) and isinstance(right, Expr))
        self.left = left
        self.right = right

    def __str__(self):
        return "({0}) - ({1})".format(self.left, self.right)


class MulExpr(Expr):
    """Represents a product of two expressions """

    left = None
    right = None

    def __init__(self, left, right):
        assert(isinstance(left, Expr) and isinstance(right, Expr))
        self.left = left
        self.right = right

    def __str__(self):
        return "({0}) * ({1})".format(self.left, self.right)


class VarExpr(Expr):
    """Represents an expression associated with a variable """

    var = None

    def __init__(self, var):
        assert(isinstance(var, Variable))
        self.var = var

    def __str__(self):
        if verbose_flag:
            return "VarExpr({0})".format(self.var)
        else:
            return str(self.var)


class ConstExpr(Expr):
    """Represents an expression associated with a constant """

    const = None

    def __init__(self, const):
        assert(isinstance(const, Constant))
        self.const = const

    def __str__(self):
        if verbose_flag:
            return "ConstExpr({0})".format(self.const)
        else:
            return str(self.const)


def mk_var_expr(name):
    """Creates a VarExpr from a given name"""
    var = Variable(name)
    return VarExpr(var)


def mk_const_expr(c):
    """Creates a ConstExpr from a given constant (string or number)"""
    const = Constant(c)
    return ConstExpr(const)


class ErrorTerm:
    """Represents an error term.

    Error terms appear due to absolute and relative rounding errors.
    The rounding model gives rnd(x) = x(1 + e) + d where
    e is a relative error and d is an absolute error of rnd.
    The current implementation is very simple and the role of each
    error term should be derived from its context.
    """

    global_index = 0

    # error term index (different error terms have different indices)
    index = None

    # True if relative
    relative = None

    def __init__(self, index, relative):
        assert (type(index) is int)
        assert (type(relative) is bool)
        self.index = index
        self.relative = relative

    def __repr__(self):
        return "ErrorTerm({0}, {1})".format(self.index, self.relative)

    def __str__(self):
        if verbose_flag:
            return self.__repr__()
        else:
            if self.relative:
                return "e_" + str(self.index)
            else:
                return "d_" + str(self.index)

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        assert (isinstance(other, ErrorTerm))
        return self.index == other.index


def get_error_term(e=None, rel=True):
    ErrorTerm.global_index += 1
    return ErrorTerm(ErrorTerm.global_index, rel)


class Monomial:
    """Represents a monomial in the form c * (x * y * ...) * rel_error * abs_error.

    Here c is a constant (fraction) and x, y, ... are variables;
    rel_error = (1 + e1)(1 + e2)... is an accumulated relative error;
    abs_error = d1 * d2 * ... is an accumulated absolute error.
    """

    # constant coefficient (Fraction)
    c = None

    # list of variables ([Variable])
    vars = None

    # list of relative error terms ([ErrorTerm])
    rel_errs = None

    # list of absolute error terms ([ErrorTerm])
    abs_errs = None

    def __init__(self):
        self.c = Fraction(1)
        self.vars = []
        self.rel_errs = []
        self.abs_errs = []

    def copy(self):
        """Creates a copy of itself"""
        m = Monomial()
        m.c = self.c
        m.vars = list(self.vars)
        m.rel_errs = list(self.rel_errs)
        m.abs_errs = list(self.abs_errs)
        return m

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        c_str = str(self.c)
        vars_str = "*".join([str(v) for v in self.vars])
        rel_str = "*".join(["(1 + {0})".format(e) for e in self.rel_errs])
        abs_str = "*".join(["{0}".format(e) for e in self.abs_errs])
        return "*".join([s for s in [c_str, vars_str, rel_str, abs_str] if s != ""])


def var_expr_to_poly(expr):
    """Converts VarExpr to a polynomial (a list of monomials: [Monomial])"""
    assert(isinstance(expr, VarExpr))
    m = Monomial()
    m.vars.append(expr.var)
    return [m]


def const_expr_to_poly(expr):
    """Converts ConstExpr to a polynomial (a list of monomials)"""
    assert(isinstance(expr, ConstExpr))
    m = Monomial()
    m.c = expr.const.value
    return [m]


def rnd_poly(poly, rel_error, abs_error):
    """Rounds a given polynomial (a list of monomials) and returns
    a new polynomial for the rounded result
    """
    result = [m.copy() for m in poly]
    if rel_error:
        for m in result:
            m.rel_errs.append(rel_error)
    if abs_error:
        abs_m = Monomial()
        abs_m.abs_errs.append(abs_error)
        result.append(abs_m)
    return result


def neg_poly(p):
    """Returns a negation of a polynomial"""
    result = [m.copy() for m in p]
    for m in result:
        m.c = -m.c
    return result
        

def add_poly(p, g):
    """Returns a sum of two polynomials"""
    return [m.copy() for m in p + g]


def sub_poly(p, g):
    """Returns a difference of two polynomials"""
    result = [m.copy() for m in p]
    for m in g:
        k = m.copy()
        k.c = -k.c
        result.append(k)
    return result


def mul_poly(p, g):
    """Returns a product of two polynomials"""
    result = []
    for m in p:
        for n in g:
            k = Monomial()
            k.c = m.c * n.c
            k.vars = m.vars + n.vars
            k.rel_errs = m.rel_errs + n.rel_errs
            k.abs_errs = m.abs_errs + n.abs_errs
            result.append(k)
    return result


def float_poly(expr):
    """Converts an expression (Expr) to a polynomial (a list of monomials)
    which represents the corresponding floating-point expression.

    It is assumed that all computations are done with the same floating-point format.

    The standard floating-point rounding model is used:
    VarExpr(x) if real_vars_flag = True --> rnd(x) = x * (1 + e) + d
    VarExpr(x) if real_vars_flag = False --> rnd(x) = x
    ConstExpr(c) --> rnd(c) = (1 + e) * c (it is assumed that all constants are normal)
    AddExpr(e1, e2) --> rnd(e1 + e2) = (e1 + e2) * (1 + e) (subnormal results are exact and hence d = 0)
    SubExpr(e1, e2) --> rnd(e1 - e2) = (e1 - e2) * (1 + e) (subnormal results are exact and hence d = 0)
    MulExpr(e1, e2) --> rnd(e1 * e2) = (e1 * e2) * (1 + e) + d
    """
    if isinstance(expr, VarExpr):
        v = var_expr_to_poly(expr)
        if real_vars_flag:
            e = get_error_term(expr)
            d = get_error_term(expr, rel=False)
            return rnd_poly(v, e, d)
        else:
            return v
    elif isinstance(expr, ConstExpr):
        e = get_error_term(expr)
        return rnd_poly(const_expr_to_poly(expr), e, None)
    elif isinstance(expr, NegExpr):
        p = float_poly(expr.expr)
        return neg_poly(p)
    elif isinstance(expr, AddExpr):
        p1 = float_poly(expr.left)
        p2 = float_poly(expr.right)
        e = get_error_term(expr)
        return rnd_poly(add_poly(p1, p2), e, None)
    elif isinstance(expr, SubExpr):
        p1 = float_poly(expr.left)
        p2 = float_poly(expr.right)
        e = get_error_term(expr)
        return rnd_poly(sub_poly(p1, p2), e, None)
    elif isinstance(expr, MulExpr):
        p1 = float_poly(expr.left)
        p2 = float_poly(expr.right)
        e = get_error_term(expr)
        d = get_error_term(expr, rel=False)
        return rnd_poly(mul_poly(p1, p2), e, d)


def fixed_poly(expr):
    """Converts an expression (Expr) to a polynomial (a list of monomials)
    which represents the corresponding fixed-point expression.

    It is assumed that all computations are done with the same fixed-point format.

    The standard fixed-point rounding model is used:
    VarExpr(x) if real_vars_flag = True --> rnd(x) = x + d
    VarExpr(x) if real_vars_flag = False --> rnd(x) = x
    ConstExpr(c) --> rnd(c) = c + d
    AddExpr(e1, e2) --> rnd(e1 + e2) = e1 + e2 (exact)
    SubExpr(e1, e2) --> rnd(e1 - e2) = e1 - e2 (exact)
    MulExpr(e1, e2) --> rnd(e1 * e2) = (e1 * e2) + d
    """
    if isinstance(expr, VarExpr):
        v = var_expr_to_poly(expr)
        if real_vars_flag:
            d = get_error_term(expr, rel=False)
            return rnd_poly(v, None, d)
        else:
            return var_expr_to_poly(expr)
    elif isinstance(expr, ConstExpr):
        d = get_error_term(expr, rel=False)
        return rnd_poly(const_expr_to_poly(expr), None, d)
    elif isinstance(expr, NegExpr):
        p = fixed_poly(expr.expr)
        return neg_poly(p)
    elif isinstance(expr, AddExpr):
        p1 = fixed_poly(expr.left)
        p2 = fixed_poly(expr.right)
        return add_poly(p1, p2)
    elif isinstance(expr, SubExpr):
        p1 = fixed_poly(expr.left)
        p2 = fixed_poly(expr.right)
        return sub_poly(p1, p2)
    elif isinstance(expr, MulExpr):
        p1 = fixed_poly(expr.left)
        p2 = fixed_poly(expr.right)
        d = get_error_term(expr, rel=False)
        return rnd_poly(mul_poly(p1, p2), None, d)
    else:
        raise TypeError("argument should be an Expr instance")


def get_real_part(poly):
    """Returns a real-valued part of a polynomial
    (the part corresponding to the ideal real-valued computations without round off errors)
    """
    result = []
    for m in poly:
        if not m.abs_errs:
            t = Monomial()
            t.c = m.c
            t.vars = list(m.vars)
            result.append(t)
    return result


def get_rel_part(poly):
    """Returns a part of a polynomial which contains relative errors only (no absolute errors)"""
    result = [m.copy() for m in poly if not m.abs_errs and m.rel_errs]
    return result


def get_abs_part(poly):
    """Returns a part of a polynomial which contains absolute errors"""
    result = [m.copy() for m in poly if m.abs_errs]
    return result


def get_rel_error_bound(poly):
    """Returns a simple relative error bound of a polynomial.

    The result is in the form [(Monomial, n)] where n is the number of relative error terms
    corresponding to the Monomial.

    Example:
    poly = [m1(x) * (1 + e1) * (1 + e2), m2(x) * (1 + e3)]
    get_rel_error_bound(poly) = [(m1(x), 2), (m2(x), 1)]
    """
    result = []
    r = get_rel_part(poly)
    for m in r:
        k = Monomial()
        k.c = m.c
        k.vars = m.vars
        result.append((k, len(m.rel_errs)))
    return result


def combine_rel_error(poly_rel_err):
    """Returns a simplified expression for a given relative error bound.

    The input should be of the type [(Monomial, n)] where n's are integers.
    This function multiplies all monomials by corresponding n's and finds
    the maximum value of n's.
    The result is ([Monomial], int).
    """
    err = []
    max_n = 0
    for (m, n) in poly_rel_err:
        if n > max_n:
            max_n = n
        k = m.copy()
        k.c *= n
        err.append(k)
    return (err, max_n)


def get_lin_rel_error(poly):
    """Returns a linear part of the relative error.

    This function combines monomials corresponding to the same error terms together.
    The result of this function is a list of polynomials: [[Monomial]].
    """
    result = {}
    r = get_rel_part(poly)
    for m in r:
        k = Monomial()
        k.c = m.c
        k.vars = m.vars
        for e in m.rel_errs:
            if e in result:
                result[e].append(k.copy())
            else:
                result[e] = [k.copy()]
    return result.values()


def get_abs_error_bound(poly):
    """Returns a simple absolute error bound of a polynomial.

    The result is in the form [(Monomial, k, n)] where 
    k is the number of absolute error terms and
    n is the number of relative error terms
    corresponding to the Monomial.

    Example:
    poly = [m1(x) * (1 + e1) * (1 + e2) * d4, m2(x) * (1 + e3) * d4 * d5 * d4]
    get_abs_error_bound(poly) = [(m1(x), 2, 1), (m2(x), 1, 3)]
    """
    result = []
    r = get_abs_part(poly)
    for m in r:
        k = Monomial()
        k.c = m.c
        k.vars = m.vars
        result.append((k, len(m.abs_errs), len(m.rel_errs)))
    return result


def combine_abs_error(poly_abs_err):
    """Returns a simplified expression for a given absolute error bound.

    The input should be of the type [(Monomial, k, n)] where k's and n's are integers.
    All k's should be at least 1.
    This function returns two polynomials: one with monomials for which k == 1 and another
    with monomials for which k >= 2.
    The result also contains maximum values of n's for both polynomials.
    """
    err1 = []
    err2 = []
    max_n1 = 0
    max_n2 = 0
    for (m, a, n) in poly_abs_err:
        assert (a >= 1)
        if a >= 2:
            err2.append(m.copy())
            if n > max_n2:
                max_n2 = n
        else:
            err1.append(m.copy())
            if n > max_n1:
                max_n1 = n
    return (err1, max_n1, err2, max_n2)


def poly_to_str(poly):
    """Converts a polynomial (a list of monomials) into a string"""
    if not poly:
        return "0"
    else:
        return " + ".join([str(m) for m in poly])


def poly_to_str_abs(poly):
    """Returns a string corresponding to a polynomial where all monomials
    are replaced by their absolute values"""
    if not poly:
        return "0"
    else:
        return " + ".join([abs_template.format(m) for m in poly])


def poly_err_to_str(poly_err, err_template):
    """Converts a polynomial error ([(Monomial, int)]) into a string"""
    if not poly_err:
        return "0"
    strs = ["{0} * {1}".format(abs_template.format(m),
                               err_template.format(n)) for (m, n) in poly_err]
    return " + ".join(strs)


def analyze_float(expr):
    """Analyzes a given expression and prints out all floating-point error bounds"""
    fp = float_poly(expr)
    err0_rel = get_rel_error_bound(fp)
    err0_rel_combined, max_rel_n = combine_rel_error(err0_rel)
    err1_rel = get_lin_rel_error(fp)
    err2_rel = [(m, n) for (m, n) in err0_rel if n >= 2]
    err2_rel2 = [(m, n * n) for (m, n) in err2_rel]

    err_abs = get_abs_error_bound(fp)
    err1_abs, max_abs1_n, err2_abs, max_abs2_n = combine_abs_error(err_abs)

    v0_str = poly_to_str(get_real_part(fp))

    err0_rel_str = poly_err_to_str(err0_rel,
                                   "((1 + e)^{0} - 1)".replace("e", eps_name))
    err0_rel_combined_str = poly_to_str_abs(err0_rel_combined)

    template0 = " * d^{0}".replace("d", delta_name)
    template1 = " * (1 + e)^{0}".replace("e", eps_name)
    err_abs_strs = []
    for (m, k, n) in err_abs:
        s = abs_template.format(m) + template0.format(k)
        if n > 0:
            s += template1.format(n)
        err_abs_strs.append(s)

    if err_abs_strs:
        err_abs_str = " + ".join(err_abs_strs)
    else:
        err_abs_str = "0"
    err12_abs_str = poly_to_str_abs(err1_abs + err2_abs)

    err1_rel_strs = [abs_template.format(poly_to_str(p)) for p in err1_rel]
    err2_rel_str = poly_err_to_str(err2_rel,
                                   "((1 + e)^{0} - 1 - {0}*e)".replace("e", eps_name))
    err2_rel_str_combined = poly_to_str_abs(combine_rel_error(err2_rel2)[0])

    print("float({0}) = v0 + error".format(expr))
    print("v0 = {0}\n".format(v0_str))

    print("error = err_rel + err_abs\n")

    print("|err_rel| <= {0}".format(err0_rel_str))
    print("|err_rel| <= ({0}) * eps / (1 - {1}*eps)\n"
          .replace("eps", eps_name)
          .format(err0_rel_combined_str, max_rel_n))

    print("|err_abs| <= {0}".format(err_abs_str))
    print("|err_abs| <= ({0}) * (1 + eps)^{1} * delta\n"
          .replace("eps", eps_name)
          .replace("delta", delta_name)
          .format(err12_abs_str, max(max_abs1_n, max_abs2_n)))

    if err1_rel:
        print("err_rel = err_rel1 + err_rel2\n")

        print("|err_rel1| <= ({0}) * eps"
              .replace("eps", eps_name)
              .format(" + ".join(err1_rel_strs)))

        print("|err_rel2| <= {0}".format(err2_rel_str))
        print("|err_rel2| <= ({0}) * eps^2 / (1 - {1}*eps)\n"
              .replace("eps", eps_name)
              .format(err2_rel_str_combined, max_rel_n))


def analyze_fixed(expr):
    """Analyzes a given expression and prints out all fixed-point error bounds"""
    fx = fixed_poly(expr)

    err_abs = get_abs_error_bound(fx)
    err1_abs, max_abs1_n, err2_abs, max_abs2_n = combine_abs_error(err_abs)

    v0_str = poly_to_str(get_real_part(fx))

    template0 = " * d^{0}".replace("d", delta_name)
    err_abs_strs = []
    for (m, k, n) in err_abs:
        assert(n == 0)
        s = abs_template.format(m) + template0.format(k)
        err_abs_strs.append(s)

    if err_abs_strs:
        err_abs_str = " + ".join(err_abs_strs)
    else:
        err_abs_str = "0"

    err1_abs_str = poly_to_str_abs(err1_abs)
    err2_abs_str = poly_to_str_abs(err2_abs)

    print("fixed({0}) = v0 + error".format(expr))
    print("v0 = {0}\n".format(v0_str))

    print("|error| <= {0}\n".format(err_abs_str))

    print("error = error1 + error2\n")

    print("|error1| <= ({0}) * delta"
          .replace("delta", delta_name)
          .format(err1_abs_str))

    print("|error2| <= ({0}) * delta^2\n"
          .replace("delta", delta_name)
          .format(err2_abs_str))
